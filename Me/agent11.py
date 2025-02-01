from model import GNN
from utils import state_to_graph
import jax.numpy as jnp
import pickle

# Load trained model parameters
with open("trained_params.pkl", "rb") as f:
    trained_params = pickle.load(f)

model = GNN()

def agent_fn(observation, config):
    graph = state_to_graph(observation)
    logits = model.apply(trained_params, graph)
    actions = {unit_id: jnp.argmax(logits[unit_id]) % 5 for unit_id in observation.units}
    return dict(action=actions)