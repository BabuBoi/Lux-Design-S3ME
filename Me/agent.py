import pickle
import os
import jax.numpy as jnp
import numpy as np
import jax
import jraph
import flax.linen as nn
import random
class GNN(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple):
        x = nn.Dense(self.hidden_dim)(graph.nodes)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return x  # New node embeddings

class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()
        
        self.epsilon = 1
        self.gnn_model = GNN(hidden_dim=32)
        key = jax.random.PRNGKey(0)
        dummy_graph = self.build_graph(np.zeros((1, 2)), np.zeros((1, 2)))
        self.params = self.gnn_model.init(key, dummy_graph)
        # Auto-load model if it exists
        model_path = "gnn_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self.params = pickle.load(f)

    def build_graph(self, unit_positions, relic_positions):
        num_units = unit_positions.shape[0]
        num_relics = relic_positions.shape[0]
        num_nodes = num_units + num_relics

        nodes = np.concatenate((unit_positions, relic_positions), axis=0)
        senders, receivers = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    senders.append(i)
                    receivers.append(j)

        return jraph.GraphsTuple(
            nodes=jnp.array(nodes),
            edges=None,
            senders=jnp.array(senders),
            receivers=jnp.array(receivers),
            n_node=jnp.array([num_nodes]),
            n_edge=jnp.array([len(senders)]),
            globals=None
        )

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_positions = np.array(obs["units"]["position"][self.team_id])  
        observed_relic_positions = np.array(obs["relic_nodes"])
        unit_mask = np.array(obs["units_mask"][self.team_id]) # shape (max_units, )
        unit_energys = np.array(obs["units"]["energy"][self.team_id]) # shape (max_units, 1)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"]) # shape (max_relic_nodes, )
        team_points = np.array(obs["team_points"]) # points of each team, team_points[self.team_id] is the points of the your team
        # ids of units you can control at this timestep
        available_unit_ids = np.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.where(observed_relic_nodes_mask)[0])
        
        graph = self.build_graph(unit_positions, observed_relic_positions)
        gnn_output = self.gnn_model.apply(self.params, graph)

        # Convert actions to a JAX array
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for unit_id in range(unit_positions.shape[0]):
            if random.random() < self.epsilon:
                action_direction = np.random.randint(0, 5)
            else:
                action_direction = np.argmax(gnn_output[unit_id]) % 5
            actions[unit_id] = [action_direction, 0, 0]

        return jnp.array(actions)  # Ensure actions are JAX arrays