import jax
import jax.numpy as jnp
import optax  # Optimizer
import numpy as np
from agent import Agent, GNN
import pickle

def loss_fn(params, model, graph):
    """Compute loss (example: MSE loss on node features)."""
    output = model.apply(params, graph)
    target = jnp.ones_like(output)  # Dummy target for illustration
    return jnp.mean((output - target) ** 2)

def train(agent, num_epochs=100, lr=0.01, save_path="gnn_model.pkl"):
    key = jax.random.PRNGKey(0)
    dummy_graph = agent.build_graph(np.random.rand(5, 2), np.random.rand(3, 2))  # Dummy input
    params = agent.params  # Initial parameters

    # Define optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Training step function
    @jax.jit
    def step(params, opt_state, graph):
        loss, grads = jax.value_and_grad(loss_fn)(params, agent.gnn_model, graph)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Training loop
    for epoch in range(num_epochs):
        params, opt_state, loss = step(params, opt_state, dummy_graph)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # Save trained model
    with open(save_path, "wb") as f:
        pickle.dump(params, f)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    agent = Agent(player="player_0", env_cfg={"max_units": 5})
    train(agent)