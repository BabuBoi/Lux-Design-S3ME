import jax.numpy as jnp
import jraph
from flax import linen as nn

class GNN(nn.Module):
    latent_dim: int = 128

    def setup(self):
        self.node_encoder = nn.Dense(self.latent_dim)
        self.edge_encoder = nn.Dense(self.latent_dim)
        self.gnn_layer = jraph.GraphNetwork(
            update_node_fn=nn.Dense(self.latent_dim),
            update_edge_fn=nn.Dense(self.latent_dim),
            update_global_fn=nn.Dense(self.latent_dim),
        )
        self.output_layer = nn.Dense(5)  # 5 possible actions

    def __call__(self, graph):
        nodes = self.node_encoder(graph.nodes)
        edges = self.edge_encoder(graph.edges)
        graph = graph._replace(nodes=nodes, edges=edges)
        graph = self.gnn_layer(graph)
        return self.output_layer(graph.nodes)  # Output per unit