import jax
from jax import random
import haiku as hk
import optax
import jax.numpy as jnp
import jraph
import collections
import numpy as np
import pickle
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(pickle.dumps(experience))

    def sample(self, batch_size):
        indices = jax.random.choice(jax.random.PRNGKey(0), len(self.buffer), (batch_size,), replace=False)
        batch = [pickle.loads(self.buffer[idx]) for idx in indices]
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buffer)

class SharedQNetwork(hk.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim  # 6 possible actions

    def __call__(self, graph):
        """Processes the entire graph and returns Q-values per unit"""
        def update_fn(node_features):
            return hk.nets.MLP([64, 64, self.output_dim])(node_features)

        gnn = jraph.GraphNetwork(
            update_node_fn=update_fn,
            update_edge_fn=None,
            update_global_fn=None,
            aggregate_edges_for_nodes_fn=jraph.segment_mean
        )

        updated_graph = gnn(graph)
        return updated_graph.nodes  # Q-values per node
    
class QNetwork:
    def __init__(self, grid_size, max_units):
        self.grid_size = grid_size
        self.max_units = max_units
        self.network = hk.transform(self.q_network)
        self.params = self.network.init(random.PRNGKey(42), self.create_dummy_graph())
        self.optimizer = optax.adam(1e-3)
        self.opt_state = self.optimizer.init(self.params)

    def q_network(self, graph):
        net = jraph.GraphNetwork(
            update_node_fn=lambda nodes, sent_edges, received_edges, globals_: jax.nn.relu(hk.Linear(128)(nodes)),
            update_edge_fn=lambda edges, senders, receivers, globals_: jax.nn.relu(hk.Linear(128)(edges)),
            update_global_fn = lambda nodes, edges, globals_: jax.nn.relu(hk.BatchApply(hk.Linear(128))(jnp.asarray(globals_)))
        )
        with open("log.txt", "a") as f:
            f.write(f"globals_shape: {globals_}\n")
        q_values = net(graph)
        # Ensure the output is constrained to the 6 possible actions
        q_values = hk.Linear(6)(q_values.nodes)
        return q_values

    def create_dummy_graph(self):
        num_tiles = self.grid_size * self.grid_size
        num_units = self.max_units * 2  # Both teams' units

        # Dummy tile nodes
        tile_nodes = jnp.ones((num_tiles, 4))  # 4 features: [energy, tile_type, relic_presence, unit_count]

        # Dummy unit nodes
        unit_nodes = jnp.ones((num_units, 6))  # 6 features: [energy, team_id, relic_proximity, energy_nearby, nebula_nearby, asteroid_nearby]

        # Pad tile nodes to match the number of features of unit nodes
        tile_nodes_padded = jnp.pad(tile_nodes, ((0, 0), (0, 2)), mode='constant')

        # Dummy edges
        senders = jnp.array([0])
        receivers = jnp.array([0])
        edge_features = jnp.ones((1, 1))  # Dummy edge feature

        return jraph.GraphsTuple(
            nodes=jnp.concatenate([tile_nodes_padded, unit_nodes], axis=0),  # All nodes combined
            edges=edge_features,
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([num_tiles + num_units]),  # Total number of nodes
            n_edge=jnp.array([1]),  # Total number of edges
            globals=jnp.array([[1.0]])  # Now shape is (1, 1)
        )

    def apply(self, params, graph):
        return self.network.apply(params, None, graph)
class Agent:
    def __init__(self, player: str, env_cfg: dict):
        self.player = player
        self.team_id = 0 if player == "player_0" else 1
        self.env_cfg = env_cfg
        self.max_units = env_cfg["max_units"]
        self.unit_sap_range = env_cfg["unit_sap_range"]
        self.unit_sap_cost = env_cfg["unit_sap_cost"]
        self.unit_move_cost = env_cfg["unit_move_cost"]
        self.unit_sensor_range = env_cfg["unit_sensor_range"]
        self.grid_size = 24
        self.max_steps = env_cfg["max_steps_in_match"] * env_cfg["match_count_per_episode"]
        #self.max_steps = env_cfg["max_steps_in_match"]
        #self.q_networks = [QNetwork(self.grid_size, self.max_units) for _ in range(self.max_units)]
        #self.target_networks = [QNetwork(self.grid_size, self.max_units) for _ in range(self.max_units)]
        self.q_network = QNetwork(self.grid_size, self.max_units)  # shared network
        self.target_network = QNetwork(self.grid_size, self.max_units)
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.update_target_steps = 1000
        self.last_obs = None
        self.last_actions = None
        self.visited_positions = set()
          
    def find_target_to_sap(self, unit_pos, observed_relic_node_positions, observed_relic_nodes_mask):
        # Dummy implementation, replace with actual logic
        target_pos = observed_relic_node_positions[0]
        return target_pos

    def create_bipartite_graph_from_obs(self, obs):
        num_tiles = self.grid_size * self.grid_size
        num_units = self.max_units * 2  # Both teams' units
        unit_sensor_range = self.unit_sensor_range
        unit_sap_range = self.unit_sap_range

        # === Tile Nodes === #
        tile_nodes = jnp.zeros((num_tiles, 4))  # [energy, tile_type, relic_presence, unit_count]

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                idx = x * self.grid_size + y
                if obs["sensor_mask"][x][y]:  # Only update visible tiles
                    tile_energy = obs["map_features"]["energy"][x][y]
                    tile_type = obs["map_features"]["tile_type"][x][y]  # 0 = empty, 1 = nebula, 2 = asteroid
                    relic_presence = any((x, y) == tuple(relic) for relic in obs["relic_nodes"] if relic[0] != -1)
                    unit_count = sum(
                        (obs["units_mask"][t][u] and tuple(obs["units"]["position"][t][u]) == (x, y))
                        for t in range(2) for u in range(self.max_units)
                    )
                    tile_nodes = tile_nodes.at[idx].set(jnp.array([tile_energy, tile_type, relic_presence, unit_count]))

        # === Unit Nodes === #
        unit_nodes = jnp.zeros((num_units, 6))  # [energy, team_id, relic_proximity, energy_nearby, nebula_nearby, asteroid_nearby]

        unit_positions = {}  # Track unit positions for unit-unit edges
        for team in range(2):
            for unit_id in range(self.max_units):
                if obs["units_mask"][team][unit_id]:  # Check if unit exists and is visible
                    pos_x, pos_y = obs["units"]["position"][team][unit_id]
                    energy = obs["units"]["energy"][team][unit_id]
                    unit_idx = team * self.max_units + unit_id
                    tile_idx = pos_x * self.grid_size + pos_y

                    # Relic proximity
                    relic_nearby = any(
                        abs(pos_x - relic[0]) <= unit_sensor_range and abs(pos_y - relic[1]) <= unit_sensor_range
                        for relic in obs["relic_nodes"] if relic[0] != -1
                    )

                    # Aggregate nearby tile features
                    energy_nearby, nebula_nearby, asteroid_nearby = 0, 0, 0
                    for dx in range(-unit_sensor_range, unit_sensor_range + 1):
                        for dy in range(-unit_sensor_range, unit_sensor_range + 1):
                            nx, ny = pos_x + dx, pos_y + dy
                            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and obs["sensor_mask"][nx][ny]:
                                tile_type = obs["map_features"]["tile_type"][nx][ny]
                                tile_energy = obs["map_features"]["energy"][nx][ny]

                                energy_nearby += tile_energy
                                nebula_nearby += (tile_type == 1)
                                asteroid_nearby += (tile_type == 2)

                    unit_nodes = unit_nodes.at[unit_idx].set(
                        jnp.array([energy, team, relic_nearby, energy_nearby, nebula_nearby, asteroid_nearby])
                    )

                    unit_positions[unit_idx] = (pos_x, pos_y)

        # Pad tile nodes to match the number of features of unit nodes
        tile_nodes_padded = jnp.pad(tile_nodes, ((0, 0), (0, 2)), mode='constant')

        # === Edges === #
        senders, receivers, edge_features = [], [], []

        # Unit-Tile Edges (Each unit connects to its tile)
        for unit_idx, (x, y) in unit_positions.items():
            tile_idx = x * self.grid_size + y
            senders.append(unit_idx + num_tiles)  # Unit index in bipartite graph
            receivers.append(tile_idx)
            edge_features.append(self.unit_move_cost)  # Moving to a tile has a cost

        # Tile-Tile Movement Edges (adjacency)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                tile_idx = x * self.grid_size + y
                neighbors = [
                    ((x-1, y), self.unit_move_cost),  # Up
                    ((x+1, y), self.unit_move_cost),  # Down
                    ((x, y-1), self.unit_move_cost),  # Left
                    ((x, y+1), self.unit_move_cost)   # Right
                ]
                for (nx, ny), move_cost in neighbors:
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        neighbor_idx = nx * self.grid_size + ny
                        senders.append(tile_idx)
                        receivers.append(neighbor_idx)
                        edge_features.append(move_cost)

        # Unit-Unit Edges (within sap range)
        for u1, (x1, y1) in unit_positions.items():
            for u2, (x2, y2) in unit_positions.items():
                if u1 != u2 and abs(x1 - x2) <= unit_sap_range and abs(y1 - y2) <= unit_sap_range:
                    senders.append(u1 + num_tiles)
                    receivers.append(u2 + num_tiles)
                    edge_features.append(self.unit_sap_cost)  # Sapping has a cost

        # Convert lists to JAX arrays
        senders = jnp.array(senders)
        receivers = jnp.array(receivers)
        edge_features = jnp.array(edge_features).reshape(-1, 1)  # Edge features as a column vector

        return jraph.GraphsTuple(
            nodes=jnp.concatenate([tile_nodes_padded, unit_nodes], axis=0),  # All nodes combined
            edges=edge_features,
            senders=senders,
            receivers=receivers,
            n_node=jnp.array([num_tiles + num_units]),  # Total number of nodes
            n_edge=jnp.array([len(senders)]),  # Total number of edges
            globals=jnp.array([1.0])  # Dummy global feature
        )
    
    def get_valid_actions(self, unit_pos):
        action_to_direction = {
        0: "STAY",     
        1: "UP",
        2: "RIGHT",
        3: "DOWN",
        4: "LEFT",
        5: "sap" 
        }
        # Map directions to position offsets (dx, dy)
        direction_to_offset = {
        "STAY": (0, 0),
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0)
        }
        x, y = unit_pos
        map_width = 24
        map_height = 24
        valid_actions = []
        for action_index in range(6):
            direction = action_to_direction.get(action_index)
            if direction in direction_to_offset:
                dx, dy = direction_to_offset[direction]
                new_x = x + dx
                new_y = y + dy
                # Check if new position is within the map boundaries
                if 0 <= new_x < map_width and 0 <= new_y < map_height:
                    valid_actions.append(action_index)
            else:
                # Actions that don't involve movement are always valid
                valid_actions.append(action_index)
        return valid_actions
    
    def train(self):
        batch = self.replay_buffer.sample(self.batch_size)
        last_obs_batch, last_actions_batch, reward_batch, obs_batch, done_batch = batch

        # Convert only the numeric data to JAX arrays
        last_actions_batch = jnp.array(last_actions_batch, dtype=jnp.int32)
        reward_batch = jnp.array(reward_batch, dtype=jnp.float32)
        done_batch = jnp.array(done_batch, dtype=jnp.float32).reshape(-1, 1)

        # Ensure obs_batch and last_obs_batch are dictionaries
        if isinstance(obs_batch[0], bytes):
            obs_batch = [pickle.loads(obs) for obs in obs_batch]
        if isinstance(last_obs_batch[0], bytes):
            last_obs_batch = [pickle.loads(obs) for obs in last_obs_batch]

        # Process each dictionary in the list
        obs_batch_graphs = [self.create_bipartite_graph_from_obs(obs) for obs in obs_batch]
        last_obs_batch_graphs = [self.create_bipartite_graph_from_obs(obs) for obs in last_obs_batch]
        # Batch graphs using jraph.batch
        last_obs_graph = jraph.batch([self.create_bipartite_graph_from_obs(obs) for obs in last_obs_batch])
        next_obs_graph = jraph.batch([self.create_bipartite_graph_from_obs(obs) for obs in obs_batch])

        # Compute Q-values for batched graphs from shared networks
        current_q_values = self.q_network.apply(self.q_network.params, last_obs_graph)  # shape: (total_nodes, 6)
        next_q_values = self.target_network.apply(self.target_network.params, next_obs_graph)

        # IMPORTANT: You must have a way to extract unit-specific node indices from each experience.
        # For example, if your replay buffer also stored 'unit_node_indices' (shape: (batch_size, max_units))
        # then gather the Q-values for your controlled units.:

        def gather_unit_q(q_vals, unit_indices):
            # q_vals: (total_nodes, 6), unit_indices: (batch_size, max_units)
            # Returns: (batch_size, max_units, 6)
            return jnp.stack([q_vals[unit_indices[i]] for i in range(unit_indices.shape[0])])

        # (You need to add unit_node_indices into your experience tuples.
        # For illustration, assume unit_node_indices is computed as below:)
        num_tiles = self.grid_size * self.grid_size
        # For each experience, assume unit i's node index = num_tiles + i
        if self.team_id == 0:
            unit_node_indices = jnp.tile(jnp.arange(num_tiles, num_tiles + self.max_units)[None, :], (self.batch_size, 1))
        else:
            unit_node_indices = jnp.tile(jnp.arange(num_tiles + self.max_units, num_tiles + self.max_units + self.max_units)[None, :], (self.batch_size, 1))
        current_unit_q = gather_unit_q(current_q_values, unit_node_indices)  # shape: (batch_size, max_units, 6)
        next_unit_q = gather_unit_q(next_q_values, unit_node_indices)

        # Now, select Q-values for the actions taken (last_actions_batch assumed shape (batch_size, max_units))
        selected_q = jnp.take_along_axis(current_unit_q, last_actions_batch[..., None], axis=2).squeeze(axis=2)

        # Compute targets per unit
        max_next_q = jnp.max(next_unit_q, axis=2)
        targets = reward_batch.reshape(-1, 1) + self.gamma * max_next_q * (1 - done_batch)

        loss = jnp.mean((selected_q - targets) ** 2)

        grads = jax.grad(lambda p: loss)(self.q_network.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.q_network.params = optax.apply_updates(self.q_network.params, updates)

        '''
        for unit_id in range(self.max_units):
            # Calculate the target Q-values using the target network
            next_q_values = jnp.array([self.target_networks[unit_id].apply(self.target_networks[unit_id].params, graph) for graph in obs_batch_graphs])
            target_q_values = reward_batch + self.gamma * jnp.max(next_q_values, axis=1) * (1 - done_batch)
            target_q_values = target_q_values.reshape(-1, 1)

            def loss_fn(params):
                # Calculate the Q-values using the current network with the given params
                current_q_values = jnp.array([self.q_networks[unit_id].apply(params, graph) for graph in last_obs_batch_graphs])
                selected_q_values = current_q_values[jnp.arange(self.batch_size), last_actions_batch]
                loss = jnp.mean((selected_q_values - target_q_values) ** 2)
                return loss

            # Compute gradients and update the network parameters
            grads = jax.grad(loss_fn)(self.q_networks[unit_id].params)
            updates, self.q_networks[unit_id].opt_state = self.q_networks[unit_id].optimizer.update(grads, self.q_networks[unit_id].opt_state)
            self.q_networks[unit_id].params = optax.apply_updates(self.q_networks[unit_id].params, updates)
        '''
    def get_reward(self, obs, actions):
        # Dummy implementation, replace with actual logic
        unit_positions = jnp.array(obs["units"]["position"][self.team_id])
        reward = 0.0
        for pos in unit_positions:
            pos_tuple = tuple(pos.tolist())
            if pos_tuple not in self.visited_positions:
                # Reward for exploring new positions
                reward += 1.0 
                self.visited_positions.add(pos_tuple)
        return reward
    
    def is_done(self, obs):
        # Dummy implementation, replace with actual logic
        step = obs["steps"]  
        if step >= self.max_steps:
            return True
        else:
            return False
    
    def update_target_networks(self):
        '''
        for unit_id in range(self.max_units):
            self.target_networks[unit_id].params = self.q_networks[unit_id].params
        '''
        self.target_network.params = self.q_network.params
         
    def act(self, step: int, obs: dict, remainingOverageTime: int):
        # Convert observation to graph structure
        #graph = self.create_graph_from_obs(obs)
        graph = self.create_bipartite_graph_from_obs(obs)
        actions = jnp.zeros((self.env_cfg["max_units"], 3), dtype=int)

        for unit_id in range(self.max_units):
            if obs["units_mask"][self.team_id][unit_id]:
                unit_pos = obs["units"]["position"][self.team_id][unit_id]
                #q_values = self.q_networks[unit_id].apply(graph)
                #q_values = self.q_networks[unit_id].apply(self.q_networks[unit_id].params, graph)
                #q_value = q_values[unit_pos[0] * self.grid_size + unit_pos[1]]
                q_values = self.q_network.apply(self.q_network.params, graph)
                if self.team_id == 0:
                    unit_node_index = (self.grid_size * self.grid_size) + unit_id
                else:
                    unit_node_index = (self.grid_size * self.grid_size) + self.max_units + unit_id
                q_value = q_values[unit_node_index]
                valid_actions = jnp.array(self.get_valid_actions(unit_pos))
                mask = jnp.full(6, -jnp.inf)
                mask = mask.at[valid_actions].set(0)
                masked_q_values = q_value + mask
                
                if jax.random.uniform(jax.random.PRNGKey(step)) < self.epsilon:
                    action = jax.random.choice(jax.random.PRNGKey(step), valid_actions)
                else:
                    action = jnp.argmax(masked_q_values)
                    
                if action == 5:  # sap action
                    target_pos = self.find_target_to_sap(unit_pos, obs["relic_nodes"], obs["relic_nodes_mask"])
                    actions = actions.at[unit_id].set(jnp.array([action, target_pos[0], target_pos[1]]))
                else:
                    actions = actions.at[unit_id].set(jnp.array([action, 0, 0]))
                
        # Store the current observation and action in the replay buffer
        if self.last_obs is not None and self.last_actions is not None:
            reward = self.get_reward(self.last_obs, self.last_actions)
            done = self.is_done(self.last_obs)
            self.replay_buffer.add((self.last_obs, self.last_actions, reward, obs, done))

        # Update the last observation and actions
        self.last_obs = obs
        self.last_actions = actions

        # Train the network
        if len(self.replay_buffer) > self.batch_size:
            self.train()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if step % self.update_target_steps == 0:
            self.update_target_networks()
            
        return actions