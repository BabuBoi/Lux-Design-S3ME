import jax
from jax import random
import haiku as hk
import optax
import jax.numpy as jnp
import jraph

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
            update_global_fn=lambda nodes, edges, globals_: jax.nn.relu(hk.Linear(128)(globals_))
        )
        return net(graph)

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
            globals=jnp.array([1.0])  # Dummy global feature
        )

    def apply(self, graph):
        return self.network.apply(self.params, None, graph)
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
        self.q_networks = [QNetwork(self.grid_size, self.max_units) for _ in range(self.max_units)]

    def act(self, step: int, obs: dict, remainingOverageTime: int):
        # Convert observation to graph structure
        #graph = self.create_graph_from_obs(obs)
        graph = self.create_bipartite_graph_from_obs(obs)
        actions = jnp.zeros((self.env_cfg["max_units"], 3), dtype=int)

        for unit_id in range(self.max_units):
            if obs["units_mask"][self.team_id][unit_id]:
                unit_pos = obs["units"]["position"][self.team_id][unit_id]
                q_values = self.q_networks[unit_id].apply(graph)
                q_value = q_values.nodes[unit_pos[0] * self.grid_size + unit_pos[1]]
                action = jnp.argmax(q_value)
                if action == 5:  # sap action
                    target_pos = self.find_target_to_sap(unit_pos, obs["relic_nodes"], obs["relic_nodes_mask"])
                    actions = actions.at[unit_id].set(jnp.array([action, target_pos[0], target_pos[1]]))
                else:
                    actions = actions.at[unit_id].set(jnp.array([action, 0, 0]))

        return actions
    
    
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
