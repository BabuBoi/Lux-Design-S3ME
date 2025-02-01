import jax.numpy as jnp
import jraph
from lux.utils import direction_to
def state_to_graph(state):
    nodes = jnp.array([0.0, 1.2, -0.5, 3.4])  # Extract features from state
    edges = jnp.array([[dist / max_dist, shared / max_shared]  # Normalize values
    for (dist, shared) in edge_data])  # Define relationships
    senders = jnp.array([...])  # Source nodes
    receivers = jnp.array([...])  # Target nodes

    graph = jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        globals=None,
        n_node=jnp.array([len(nodes)]),
        n_edge=jnp.array([len(edges)]),
    )
    return graph