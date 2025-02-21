import pickle
import os
import jax.numpy as jnp
import numpy as np
import jax
import jraph
import flax.linen as nn
import jax.random as random
import optax
from collections import deque
from jax import jit, grad
from lux.utils import direction_to
import functools

class GCNLayer(nn.Module):
    """A single Graph Convolutional Network (GCN) layer."""
    output_dim: int

    @nn.compact
    def __call__(self, node_features, adj_matrix):
        """Applies the GraphConv layer followed by ReLU activation"""
        node_features = GraphConv(self.output_dim)(node_features, adj_matrix)
        return nn.relu(node_features)
    
class GraphConv(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, node_features, adj_matrix):
        """Applies a simple Graph Convolution"""
        w = self.param('w', nn.initializers.lecun_normal(), (node_features.shape[-1], self.output_dim))
        node_features = jnp.dot(node_features, w)  # Linear transformation
        return jnp.matmul(adj_matrix, node_features)  # Aggregate neighbor features
    
class GCN(nn.Module):
    """A multi-layer Graph Convolutional Network."""
    """A multi-layer Graph Convolutional Network."""
    hidden_dim: int 
    output_dim: int
    num_layers: int = 2

    def setup(self):
        self.gcn_layers = [GCNLayer(self.hidden_dim) for _ in range(self.num_layers - 1)]
        self.output_layer = GCNLayer(self.output_dim)

    def __call__(self, node_features, adj_matrix):
        #assert adj_matrix.shape[0] == adj_matrix.shape[1] == node_features.shape[0], "Adjacency matrix and node features shape mismatch"
        # Pass through hidden layers
        for layer in self.gcn_layers:
            node_features = layer(node_features, adj_matrix)

        # Final layer
        node_features = self.output_layer(node_features, adj_matrix)

        return node_features

# ---- GNN-based Q-Network ----
class GNNQNetwork(nn.Module):
    """Q-Network using a GCN-based graph encoder."""
    hidden_dim: int
    num_actions: int = 6
    num_gcn_layers: int = 2

    def setup(self):
        self.gcn = GCN(self.hidden_dim, self.hidden_dim, self.num_gcn_layers)
        self.policy_head = nn.Dense(self.num_actions)  # Output action Q-values

    def __call__(self, node_features, adj_matrix):
       # Ensure node_features and adj_matrix have a batch dimension
        if node_features.ndim == 2:
            node_features = node_features[None, :]  # Add batch dimension
        if adj_matrix.ndim == 2:
            adj_matrix = adj_matrix[None, :]  # Add batch dimension

        # Repeat adj_matrix to match the batch size of node_features
        batch_size = node_features.shape[0]
        adj_matrix = jnp.tile(adj_matrix, (batch_size, 1, 1))

        # Process the entire batch at once
        node_embeddings = jax.vmap(self.gcn)(node_features, adj_matrix)  # Apply GCN to each batch element
        global_embeddings = jnp.mean(node_embeddings, axis=1)  # Mean pooling over nodes for each batch element
        q_values = self.policy_head(global_embeddings)  # Compute Q-values for each batch element

        return q_values
    """
    num_actions: int = 6
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, node_features, edge_index):
        # First GCN Layer
        x = nn.Dense(self.hidden_dim)(node_features)
        x = nn.relu(x)

        # Second GCN Layer (Simple Aggregation)
        # Message Passing
        row, col = edge_index
        x = x.at[row].add(x[col])

        # Second layer
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Global Pooling
        x = jnp.mean(x, axis=0)  

        # Ensure it has a batch dimension
        if x.ndim == 1:
            x = x[None, :]  # Convert (features,) → (1, features)

        return nn.Dense(self.num_actions)(x)
        """
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.1):
        self.grid_size = 24
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize GNN Q-network
        self.q_network = GNNQNetwork(hidden_dim=576, num_actions=action_dim)
        self.target_network = GNNQNetwork(hidden_dim=576, num_actions=action_dim)
        self.edges = self.generate_edges(self.grid_size)
        self.q_params = self.q_network.init(random.PRNGKey(0), jnp.ones((self.grid_size**2, self.state_dim)),self.edges)
        self.target_params = self.q_params
        
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.q_params)
        
        self.memory = deque(maxlen=10000)
        
    def generate_edges(self,grid_size):
        """Creates an 4-neighborhood edge index for a grid."""
        num_nodes = grid_size * grid_size
        adj_matrix = jnp.zeros((num_nodes, num_nodes))
        for x in range(grid_size):
            for y in range(grid_size):
                idx = x * grid_size + y  # Convert (x, y) to 1D index
                neighbors = [
                    (x-1, y), (x+1, y), (x, y-1), (x, y+1),  # 4-neighbor connectivity
                ]
                for nx, ny in neighbors:
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        n_idx = nx * grid_size + ny
                        adj_matrix = adj_matrix.at[idx, n_idx].set(1)
                        adj_matrix = adj_matrix.at[n_idx, idx].set(1)

        return adj_matrix  # Convert to shape (2, E)
    
    def generate_edges8(self,grid_size):
        edges = []
        for x in range(grid_size):
            for y in range(grid_size):
                idx = x * grid_size + y  # Convert (x, y) to 1D index
                neighbors = [
                    (x-1, y), (x+1, y), (x, y-1), (x, y+1),  # Up, Down, Left, Right
                    (x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1)  # Diagonals
                ]
                for nx, ny in neighbors:
                    if 0 <= nx < grid_size and 0 <= ny < grid_size:
                        n_idx = nx * grid_size + ny
                        edges.append((idx, n_idx))

        return jnp.array(edges).T
    
    def generate_edgesF(num_nodes):
        """Creates a fully connected graph (all nodes communicate)."""
        edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
        return jnp.array(edges).T

    def store_transition(self, state, action, reward, next_state, done):
        # Convert inputs to JAX arrays
        state = jnp.array(state)
        next_state = jnp.array(next_state)
        reward = jnp.array(reward)
        # Append to memory
        self.memory.append((state, action, reward, next_state, done))
        
    def select_action(self, state, unit_pos, rng):
        valid_actions = jnp.array(self.get_valid_actions(unit_pos))
        idx = unit_pos[0] * self.grid_size + unit_pos[1]  # Convert (x, y) to 1D index
        if random.uniform(rng, ()) < self.epsilon:
            out = random.choice(rng, valid_actions)
            """
            with open('log.txt', 'a') as f:
                f.write(f"Unit pos:{unit_pos}; ")
                f.write(f"Valid actions:{valid_actions}")
                f.write(f"Random: {out}")
                f.write("\n")
                f.write("-"*30)
                f.write("\n")
            """
            return out
        q_values = self.q_network.apply(self.q_params, state, self.edges)
        mask = jnp.full(self.action_dim, -jnp.inf)
        mask = mask.at[valid_actions].set(0)
        masked_q_values = q_values[idx] + mask
        #masked_q_values = q_values + mask
        """
        with open('log.txt', 'a') as f:
            #f.write(f"{masked_q_values}; ")
            #f.write(f"{type(masked_q_values)}; ")
            f.write(f"Unit pos:{unit_pos}; ")
            f.write(f"Valid actions:{valid_actions}")
            f.write(f"Masked:{jnp.argmax(masked_q_values)};\n")
            f.write("\n")
            f.write("-"*30)DQNAgent
            f.write("\n")
        """
        return jnp.argmax(masked_q_values)
    
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
        for action_index in range(self.action_dim):
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

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        # Sample a batch of transitions from memory using jax.random
        rng = random.PRNGKey(0)
        indices = random.choice(rng, jnp.arange(len(self.memory)), (batch_size,), replace=False)
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to JAX arrays
        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        next_states = jnp.array(next_states)
        dones = jnp.array(dones)

        # Compute target Q-values
        next_q_values = self.target_network.apply(self.target_params, next_states, self.edges)
        next_q_values = jnp.max(next_q_values, axis=1)
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute current Q-values
        q_values = self.q_network.apply(self.q_params, states, self.edges)

        # Extract Q-values for the specific actions taken
        action_indices = actions[:, 0].astype(int)  # Extract the action indices
        unit_positions = actions[:, 1:3].astype(int)  # Extract the unit positions (x, y)
        unit_indices = jnp.array([pos[0] * self.grid_size + pos[1] for pos in unit_positions])
        q_values = q_values[jnp.arange(batch_size), unit_indices, action_indices]

        # Compute loss
        loss = jnp.mean((q_values - targets) ** 2)

        # Compute gradients and update parameters
        grads = jax.grad(lambda params: jnp.mean((self.q_network.apply(params, states, self.edges) - targets) ** 2))(self.q_params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.q_params = optax.apply_updates(self.q_params, updates)

        # Update target network parameters
        self.update_target_network()
        
    def update_target_network(self):
        self.target_params = self.q_params

class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        self.previous_observation = None
        self.previous_action = None
        self.previous_team_points = 0
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.visited_positions = set()
        # Auto-load model if it exists
        model_path = "gnndqn.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.params = pickle.load(f)
        
        # Initialize DQN agent
        self.state_dim = 6  # Number of node features
        self.action_dim = 6  # 5 possible directions (up, down, left, right, stay) and one attack action 
        self.dqn_agent = DQNAgent(self.state_dim, self.action_dim)
    
    def prepare_state(self, obs):
        # Extract features
        """
        unit_positions = jnp.array(obs["units"]["position"][self.team_id], dtype=jnp.float32).flatten()
        unit_energies = jnp.array(obs["units"]["energy"][self.team_id], dtype=jnp.float32).flatten()
        unit_mask = jnp.array(obs["units_mask"][self.team_id],dtype=jnp.float32).flatten()
        team_points = jnp.array(obs["team_points"][self.team_id],dtype=jnp.float32).flatten()
        observed_relic_nodes_mask = jnp.array(obs["relic_nodes_mask"],dtype=jnp.float32).flatten()
        # Combine features into a single state vector
        state = jnp.concatenate([unit_positions, unit_energies, unit_mask, team_points, observed_relic_nodes_mask], axis=1)
        return state
        """
        grid_size = 24
        node_features = jnp.zeros((grid_size**2, self.state_dim))

        unit_positions = jnp.array(obs["units"]["position"][self.team_id], dtype=jnp.int32)
        unit_energies = jnp.array(obs["units"]["energy"][self.team_id], dtype=jnp.float32)
        relic_positions = jnp.array(obs["relic_nodes"], dtype=jnp.int32)
        unit_mask = jnp.array(obs["units_mask"][self.team_id],dtype=jnp.float32)
        team_points = jnp.array(obs["team_points"][self.team_id],dtype=jnp.float32)
        observed_relic_nodes_mask = jnp.array(obs["relic_nodes_mask"],dtype=jnp.float32)
        """
        with open('log.txt', 'a') as f:
                f.write(f"team_points:{team_points}; ")
        """
        # Set features for units
        for unit_pos, energy in zip(unit_positions, unit_energies):
            idx = unit_pos[0] * grid_size + unit_pos[1]
            node_features = node_features.at[idx].set(jnp.array([1, energy, 0, 0, team_points, 0]))  # [unit_present, energy, relic_present, unit_mask, team_points, observed_relic_nodes_mask]

        # Set features for relics
        for relic_pos in relic_positions:
            idx = relic_pos[0] * grid_size + relic_pos[1]
            node_features = node_features.at[idx, 2].set(1)  # Mark relic presence

        return node_features
    
    def compute_reward(self, unit_positions):
        reward = 0.0

        # For each unit
        for pos in unit_positions:
            pos_tuple = tuple(pos.tolist())
            if pos_tuple not in self.visited_positions:
                # Reward for exploring new positions
                reward += 1.0 
                self.visited_positions.add(pos_tuple)
            """
            if len(self.relic_node_positions) >= 1:
                # After finding a relic, encourage staying near relics
                for relic_pos in self.relic_node_positions:
                    distance = jnp.linalg.norm(pos - relic_pos)
                    if distance < self.env_cfg["relic_proximity_radius"]:
                        # Reward for being close to a relic
                        reward += 2.0  
                    else:
                        # Penalty for moving away from relics
                        reward -= 0.5  
            """

        return reward

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit. 
        
        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        if remainingOverageTime <= 0:
            done = True
        else:
            done = False # Set to True if the episode ends;
        unit_mask = jnp.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_positions = jnp.array(obs["units"]["position"][self.team_id]) # shape (max_units, 2)
        unit_energys = jnp.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_node_positions = jnp.array(obs["relic_nodes"]) # shape (max_relic_nodes, 2)
        observed_relic_nodes_mask = jnp.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = jnp.array(obs["team_points"][self.team_id]) # points of each team, team_points[self.team_id] is the points of the your team
        
        # ids of units you can control at this timestep
        available_unit_ids = jnp.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.array(jnp.where(observed_relic_nodes_mask)[0]))
        
        actions = jnp.zeros((self.env_cfg["max_units"], 3), dtype=int)
        actionsn = jnp.zeros((24**2, 3), dtype=int)  # Shape (num_nodes, 3)

        # basic strategy here is simply to have some units randomly explore and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match
        
        # save any new relic nodes that we discover for the rest of the game.
        for id in visible_relic_node_ids:
            if id not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id)
                self.relic_node_positions.append(observed_relic_node_positions[id])
        
        rng = random.PRNGKey(step)
        
        current_state = self.prepare_state(obs)
        reward = self.compute_reward(unit_positions)
        if self.previous_observation is not None and self.previous_action is not None:
            previous_state = self.prepare_state(self.previous_observation)
            self.dqn_agent.store_transition(previous_state, self.previous_action, reward, current_state, done)
            # Train the agent
            self.dqn_agent.train()
        # unit ids range from 0 to max_units - 1
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]
            idx = unit_pos[0] * 24 + unit_pos[1]  # Convert (x, y) to 1D index
            # Convert to graph representation
            #node_features = jnp.array(unit_positions, dtype=jnp.float32)
            #edge_index = jnp.array([[i, j] for i in range(len(unit_positions)) for j in range(len(self.relic_node_positions))], dtype=jnp.int32).T
            #node_features = unit_positions  # Shape: (num_units, 2)
            #edge_index = self.dqn_agent.generate_edges(len(node_features))
            #edge_index = self.dqn_agent.generate_edges(24)
            # Select action using GNN-DQN
            action = self.dqn_agent.select_action(current_state, unit_pos, rng)
            actions = actions.at[unit_id].set(jnp.array([action, 0, 0]))
            actionsn = actionsn.at[idx].set(jnp.array([action, 0, 0]))
        self.previous_action = actionsn
        self.previous_observation = obs
        return actions
        """
        use milti agent DQN
        centralize the updates 
        take all gradendest and take avrage after converging 
        save model in pickle model takes in inputs give outputs
        eaither use centralized or decentralized 
        certralized harder to converge     
        """
     