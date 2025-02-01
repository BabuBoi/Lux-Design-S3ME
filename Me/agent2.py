from lux.utils import direction_to
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph
import pickle  # For saving/loading model parameters

class GNN(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple):
        x = nn.Dense(self.hidden_dim)(graph.nodes)  # Initial node embedding
        x = nn.relu(x)
        
        # Apply a message-passing layer
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        return x  # New node embeddings
    
class Agent():
    def __init__(self, player: str, env_cfg, model_path: str = None) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg
        
        self.gnn_model = GNN(hidden_dim=32)  # Initialize GNN model
        key = jax.random.PRNGKey(0)  # JAX requires a key for randomness
        dummy_graph = self.build_graph(np.zeros((1, 2)), np.zeros((1, 2)))  # Dummy input for initialization
        self.params = self.gnn_model.init(key, dummy_graph)  # Initialize parameters

        if model_path:
            self.load_model(model_path)

    def build_graph(self, unit_positions, relic_positions):
        """Converts environment data into a graph structure."""
        num_units = unit_positions.shape[0]
        num_relics = relic_positions.shape[0]
        num_nodes = num_units + num_relics
        
        # Node features: unit positions + relic positions
        nodes = np.concatenate((unit_positions, relic_positions), axis=0)

        # Fully connected graph (edges between all nodes)
        senders, receivers = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    senders.append(i)
                    receivers.append(j)

        graph = jraph.GraphsTuple(
            nodes=jnp.array(nodes),
            edges=None,  # No explicit edge features
            senders=jnp.array(senders),
            receivers=jnp.array(receivers),
            n_node=jnp.array([num_nodes]),
            n_edge=jnp.array([len(senders)]),
            globals=None
        )
        return graph

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # (max_units, 2)
        observed_relic_positions = np.array(obs["relic_nodes"])  # (max_relic_nodes, 2)
        
        graph = self.build_graph(unit_positions, observed_relic_positions)
        
        # Process the graph with the GNN to get embeddings
        gnn_output = self.gnn_model.apply(self.params, graph)
        
        # Select actions based on the GNN embeddings
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for unit_id in range(unit_positions.shape[0]):
            action_direction = np.argmax(gnn_output[unit_id]) % 5
            actions[unit_id] = [action_direction, 0, 0]
        if isinstance(actions, dict):  # Assuming dict of numpy arrays
            actions = {k: jnp.array(v) for k, v in actions.items()}
        else:
            actions = jnp.array(actions)

        return actions
    
    def save_model(self, path="gnn_model.pkl"):
        """Save model parameters to a file."""
        with open(path, "wb") as f:
            pickle.dump(self.params, f)
        print(f"Model saved to {path}")

    def load_model(self, path="gnn_model.pkl"):
        """Load model parameters from a file."""
        try:
            with open(path, "rb") as f:
                self.params = pickle.load(f)
            #print(f"Model loaded from {path}")
        except FileNotFoundError:
            #print(f"No model found at {path}, starting with random parameters.")