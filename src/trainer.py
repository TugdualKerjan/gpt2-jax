import jax
import jax.numpy as jnp
import optax
import equinox as eqx

@eqx.filter_value_and_grad(has_aux=True)
def calculate_loss(model, x, keys):
    logits = jax.vmap(model, in_axes=(0,0,0))(x["input_ids"], x["attention_mask"], keys)
    loss = jax.vmap(optax.softmax_cross_entropy_with_integer_labels, in_axes=(0, 0))(logits[:, :-1, :], x["input_ids"][:, 1:])
    return jnp.mean(loss), logits


@eqx.filter_jit
def make_step(model, optimizer, optimizer_state, x, keys):
    (loss, predictions), grads = calculate_loss(model, x, keys)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, model)
    model = eqx.apply_updates(model, updates)
    return model, optimizer_state, loss, predictions