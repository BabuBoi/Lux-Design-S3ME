import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from model import GNN
from utils import state_to_graph
from src.luxai_s3.env import LuxAIS3Env

# Initialize environment
env = LuxAIS3Env()

# Initialize model
model = GNN()
key = jax.random.PRNGKey(0)  # Initialize a random key
obs, state = env.reset(key)  # Pass the key to reset()
params = model.init(key, state_to_graph(obs))

# Optimizer
optimizer = optax.adam(learning_rate=3e-4)

# Training state
train_state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer,
)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    state = env.reset(key)
    done = False
    while not done:
        graph = state_to_graph(state)
        logits = model.apply(train_state.params, graph)
        
        # Select actions
        actions = {unit_id: jnp.argmax(logits[unit_id]) % 5 for unit_id in state.units}
        
        # Step environment
        next_state, reward, done, _ = env.step(actions)
        
        # Compute loss (Policy Gradient loss)
        loss, grads = jax.value_and_grad(compute_loss)(
            train_state.params, model, graph, actions, reward
        )
        
        # Update model
        train_state = train_state.apply_gradients(grads=grads)
    
    print(f"Epoch {epoch}, Loss: {loss}")