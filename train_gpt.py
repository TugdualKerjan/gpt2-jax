import datetime
import datasets
from transformers import AutoTokenizer
import optax
import jax.numpy as jnp
import equinox as eqx
from src import GPT, GPTConfig, make_step
from tensorboardX import SummaryWriter
from jax import config
import jax.random as jr
from tqdm import tqdm

config.update("jax_debug_nans", True)

DATASET_PATH = "dataset"
CONFIG = GPTConfig()
BATCH_SIZE = 64
NUM_EPOCHS = 2
LEARNING_RATE = 5e-4
RANDOM = jr.PRNGKey(79)


dataset = datasets.load_dataset("roneneldan/TinyStories")
dataset = dataset["train"]
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize(example):
    return tokenizer.batch_encode_plus(example["text"], padding="max_length", truncation=True, max_length=CONFIG.max_position_embeddings, return_tensors="pt")

tokenized_data = dataset.map(
    tokenize, remove_columns=["text"], batched=True, batch_size=8
)

tokenized_data = tokenized_data.with_format("jax")
model = GPT(CONFIG, RANDOM)
# model = GPT.create_instance(CONFIG, RANDOM) #TODO: Weight inited has much higher loss for some reason


trainable = eqx.filter(model, eqx.is_array)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients to a maximum norm of 1.0
    optax.adamw(LEARNING_RATE)
)
optimizer_state = optimizer.init(trainable)

writer = SummaryWriter(
    log_dir="./runs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
)

def collate_fn(batch):
    input_ids = [x["input_ids"] for x in batch["tokenized"]]
    attention_mask = [x["attention_mask"] for x in batch["tokenized"]]
    return {
        "input_ids": jnp.array(input_ids),
        "attention_mask": jnp.array(attention_mask),
    }

# Training loop
step = 0
for epoch in range(NUM_EPOCHS):
    for i, batch in tqdm(enumerate(tokenized_data.iter(batch_size=BATCH_SIZE)), colour="green"):
        keys = jr.split(RANDOM, BATCH_SIZE+1)  # Shape: (BATCH_SIZE+1, 2)
        k, RANDOM = keys[:-1], keys[-1]  
        model, optimizer_state, loss, predictions = make_step(model, optimizer, optimizer_state, batch, k)

        writer.add_scalar("Cross entropy loss", loss, step)
        step += 1
        if step % 5 == 0:
            writer.add_text("Input text", tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True), step)
            writer.add_text("Predicted text", tokenizer.decode(jnp.argmax(predictions[0], axis=-1), skip_special_tokens=True), step)
        if step % 150 == 0:
            eqx.tree_serialise_leaves("./gpt2.eqx", model)