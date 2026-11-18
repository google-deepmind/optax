# Character-level Transformer on Tiny Shakespeare

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.sandbox.google.com/github/google-deepmind/optax/blob/main/examples/nanolm.ipynb)

This example demonstrates how to train a small-scale transformer-based language model (inspired by NanoGPT) on the Tiny Shakespeare dataset.  The core idea is to train a model that can predict the next character in a sequence of text based on the characters that came before it.

**Why the Tiny Shakespeare Dataset?**

* **Manageable Size:**  Since we're building a small-scale model, the Tiny Shakespeare dataset provides a suitable training corpus without overwhelming computational resources.
* **Linguistic Complexity:** Shakespeare's works offer a rich vocabulary and interesting grammatical patterns, making the dataset a good testbed for our model's language learning abilities.
* **Accessibility:** Easily accessible through [TensorFlow Datasets](https://www.tensorflow.org/datasets).

**Libraries Used**

* **JAX:** Provides the foundation for numerical computations and automatic differentiation.
* **Tensorflow Datasets (`tfds`)** Offers easy access to the Tiny Shakespeare dataset.
* **Flax's NNX Module:** Provides the stateful building blocks for defining our neural network architecture.
* **Optax:** Contains a library of optimization algorithms for training the model's parameters. In this example we'll use the {py:func}`optax.adamw` solver.

```python
import functools

from flax import nnx
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import optax
import tensorflow_datasets as tfds

# platform check
print("JAX running on", jax.devices()[0].platform.upper())
```

<!-- #region -->
# Hyperparameters and dataset download

Next, we set some important hyperparameters. This includes hyperparameters for the training process such as the learning rate `LEARNING_RATE` and the batch size `BATCH_SIZE`, as well as model parameters such as the context window size `BLOCK_SIZE` and the number of layers `NUM_LAYERS`.


After setting these, we load the Tiny Shakespeare dataset and print the length of the training set, which is around one million characters, and that of the validation set (around 50k characters). Finally, we print a small snippet of the train set.
<!-- #endregion -->

```python
# @markdown Random seed:
SEED = 42  # @param{type:"integer"}
# @markdown Learning rate passed to the optimizer:
LEARNING_RATE = 2e-3 # @param{type:"number"}
# @markdown Batch size:
BATCH_SIZE = 128  # @param{type:"integer"}
# @markdown Number of training iterations:
N_ITERATIONS = 50_000  # @param{type:"integer"}
# @markdown Number of training iterations between two consecutive evaluations:
N_FREQ_EVAL = 2_000 # @param{type:"integer"}
# @markdown Rate for dropout in the transformer model
DROPOUT_RATE = 0.2  # @param{type:"number"}
# @markdown Context window for the transformer model
BLOCK_SIZE = 64  # @param{type:"integer"}
# @markdown Number of layer for the transformer model
NUM_LAYERS = 6  # @param{type:"integer"}
# @markdown Size of the embedding for the transformer model
EMBED_SIZE = 256  # @param{type:"integer"}
# @markdown Number of heads for the transformer model
NUM_HEADS = 8  # @param{type:"integer"}
# @markdown Size of the heads for the transformer model
HEAD_SIZE = 32  # @param{type:"integer"}
```

```python
ds = tfds.load("tiny_shakespeare")

# combine train and test examples into a single string
text_train = ""
for example in ds["train"].concatenate(ds["test"]).as_numpy_iterator():
  text_train += example["text"].decode("utf-8")

# similarly, create a single string for validation
text_validation = ""
for example in ds["validation"].as_numpy_iterator():
  text_validation += example["text"].decode("utf-8")
```

```python
print(f"Length of text for training: {len(text_train):_} characters")
print(f"Length of text for validation: {len(text_validation):_} characters")
```

```python
# small sample of the train set
print(text_train[:1000])
```

# Data preparation

To prepare the data for the model, we first create a vocabulary consisting of all the unique characters in the dataset. We print that vocabulary and its size.

We then define encoding and decoding functions to convert text into sequences of integers (representing our characters) and vice versa.

Finally, we define a function `get_batch` that returns random mini-batches of data. This function uses JAX's {py:func}`jax.lax.dynamic_slice` function to efficiently handle sequences of varying lengths within batches. The `@jax.jit` decorator compiles this function for faster execution. The function randomly samples a batch from the data and prepares input sequences (`x`) and target sequences (`y`). The target sequence is simply the input sequence shifted by one position, as the goal of the language model is to predict the next character given the previous ones.


```python
vocab = sorted(list(set(text_train)))
print("Vocabulary:, ", "".join(vocab))
print("Length of vocabulary: ", len(vocab))
```

```python
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# encode train and validation data
train_data = jnp.array(encode(text_train))
eval_data = jnp.array(encode(text_validation))
```

```python
dynamic_slice_vmap = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))


@jax.jit
def get_batch(random_key, data):
  """Prepares a random batch of training data.

  Args:
      random_key: A random seed for sampling a batch.
      data: The complete training dataset.

  Returns:
      x: Input sequences.
      y: Target sequences (shifted inputs).
  """
  ix = jax.random.randint(
      random_key, shape=(BATCH_SIZE, 1), minval=0, maxval=len(data) - BLOCK_SIZE
  )
  x = dynamic_slice_vmap(data, ix, (BLOCK_SIZE,))
  y = dynamic_slice_vmap(data, ix + 1, (BLOCK_SIZE,))
  return x, y
```

# NanoLM Model Definition

The NanoLM model itself is defined as a Flax NNX module. The core of the model is a Transformer architecture, designed for sequence-to-sequence tasks like language modeling. Key parameters of the model, such as the number of layers, attention heads, and embedding size, are specified here.

Inside the model's `__call__` method, we first embed our input characters into vector representations. Positional embeddings are added to provide the model with a sense of order in the sequence. The core of the Transformer consists of multiple layers. Each layer has two main components:

 * **Multi-Head Attention**: This mechanism allows the model to "attend" to different parts of the input sequence, improving its understanding of context and relationships within the text. In the code this is implemented through the {py:class}`flax.nnx.MultiHeadAttention` class.

 * **Feedforward Network**: This network processes the output of the attention layer, applying non-linear transformations to further learn complex patterns in the data.

Normalization and dropout (for regularization) are used within the layers to improve training stability.  Finally, a dense layer maps the model's output to the vocabulary size, producing probabilities for each character as the next potential character.

The generate function enables the model to create new text sequences. It iteratively generates one character at a time, conditioned on the previously generated text.

```python
class NanoLM(nnx.Module):
  """NanoLM model using flax.nnx."""

  def __init__(
      self,
      vocab_size: int,
      num_layers: int = 6,
      num_heads: int = 8,
      head_size: int = 32,
      dropout_rate: float = 0.2,
      embed_size: int = 256,
      block_size: int = 64,
      rngs: nnx.Rngs = None,
  ):
    self.vocab_size = vocab_size
    self.block_size = block_size
    self.embed_size = embed_size

    self.embed = nnx.Embed(vocab_size, embed_size, rngs=rngs)
    self.pos_embed = nnx.Embed(block_size, embed_size, rngs=rngs)

    layers = []
    for _ in range(num_layers):
      layers.append(
          nnx.Dict({
              'ln_1': nnx.LayerNorm(embed_size, rngs=rngs),
              'attn': nnx.MultiHeadAttention(
                  num_heads=num_heads,
                  in_features=embed_size,
                  qkv_features=head_size,
                  out_features=head_size * num_heads,
                  dropout_rate=dropout_rate,
                  decode=False,
                  rngs=rngs,
              ),
              'ln_2': nnx.LayerNorm(embed_size, rngs=rngs),
              'mlp_fc': nnx.Linear(embed_size, 4 * embed_size, rngs=rngs),
              'mlp_dropout': nnx.Dropout(dropout_rate, rngs=rngs),
              'mlp_proj': nnx.Linear(4 * embed_size, embed_size, rngs=rngs),
          })
      )
    self.layers = nnx.List(layers)
    self.ln_f = nnx.LayerNorm(embed_size, rngs=rngs)
    self.head = nnx.Linear(embed_size, vocab_size, rngs=rngs)

  def __call__(self, x, deterministic: bool = False):
    seq_len = x.shape[1]
    x = self.embed(x) + self.pos_embed(jnp.arange(seq_len))
    mask = jnp.tril(jnp.ones((x.shape[-2], x.shape[-2])))

    for layer in self.layers:
      x_norm = layer['ln_1'](x)
      attn_out = layer['attn'](
          x_norm,
          x_norm,
          x_norm,
          mask=mask,
          deterministic=deterministic,
      )
      x = x + attn_out

      x_norm = layer['ln_2'](x)
      x_mlp = layer['mlp_fc'](x_norm)
      x_mlp = jax.nn.relu(x_mlp)
      x_mlp = layer['mlp_dropout'](x_mlp, deterministic=deterministic)
      x_mlp = layer['mlp_proj'](x_mlp)
      x = x + x_mlp

    x = self.ln_f(x)
    return self.head(x)

  @functools.partial(nnx.jit, static_argnames=('length',))
  def generate(self, rng, length):
    def _scan_generate(carry, _):
      random_key, context = carry
      logits = self(context, deterministic=True)
      rng, rng_subkey = jax.random.split(random_key)
      new_token = jax.random.categorical(
          rng_subkey, logits[:, -1, :], axis=-1, shape=(1, 1)
      )
      context = jnp.concatenate([context[:, 1:], new_token], axis=1)
      return (rng, context), new_token

    _, new_tokens = jax.lax.scan(
        _scan_generate,
        (rng, jnp.zeros((1, self.block_size), dtype=jnp.int32)),
        (),
        length=length,
    )
    return new_tokens
```

# State, Optimizer, and Loss Definition

This section initializes the model's parameters, defines the loss function used for language modeling, and sets up the training and evaluation processes.

In this case the loss function `loss_fun` is the cross-entropy. It uses dropout for regularization, enabled via `deterministic=False` using the RNG state passed to the model (`rngs=nnx.Rngs(SEED)`). We also define a function for evaluating the model's performance on unseen data (`eval_step`) with `deterministic=True`.

```python
model = NanoLM(
    vocab_size=len(vocab),
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    head_size=HEAD_SIZE,
    dropout_rate=DROPOUT_RATE,
    embed_size=EMBED_SIZE,
    block_size=BLOCK_SIZE,
    rngs=nnx.Rngs(SEED),
)

def loss_fun(model, x, y):
  logits = model(x, deterministic=False)
  return optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=y
  ).mean()

@nnx.jit
def eval_step(model, x, y):
  logits = model(x, deterministic=True)
  return optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=y
  ).mean()
```

We've now instantiated a NanoLM model with the following number of parameters

```python
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key)
```

```python
n_params = optax.tree.size(nnx.state(model, nnx.Param))

print(f"Total number of parameters: {n_params:_}")
```

<!-- #region -->
# Model training

We start by creating an optimizer and instantiating its state using {py:class}`flax.nnx.Optimizer`. In this case we'll use {py:func}`optax.adamw` but it's possible to replace this with other optax optimizers.


We then proceed to the training loop. For maximum efficiency we extract the most computationally intensive tasks inside the `step` function and just-in-time compile this function using `@nnx.jit`. This allows JAX to perform some optimizations in our code and generally achieve a much higher efficiency than without.

Inside the training loop, we call the aforementioned `step` function, as well as computing the loss on a validation set every `N_FREQ_EVAL` iterations.
<!-- #endregion -->

```python
# To run with SGD instead of adam, replace `adam` with `sgd`
# We wrap our model and optax optimizer in an nnx.Optimizer
optimizer = nnx.Optimizer(
    model,
    optax.adamw(learning_rate=LEARNING_RATE),
    wrt=nnx.Param
)
```

```python
%%time

all_train_losses = []
all_eval_losses = []

# we define one iteration of the optimizer and JIT this function using nnx.jit
@nnx.jit
def step(model, optimizer, key):
  key, subkey = jax.random.split(key)
  batch = get_batch(key, train_data)
  loss, grad = nnx.value_and_grad(loss_fun)(model, *batch)
  optimizer.update(model, grad)
  return key, loss

for i in range(N_ITERATIONS):
  key, loss = step(model, optimizer, key)
  all_train_losses.append(loss)

  # once every N_FREQ_EVAL we compute loss on the validation set
  if i % N_FREQ_EVAL == 0:
    key, subkey = jax.random.split(key)
    eval_loss = eval_step(model, *get_batch(subkey, eval_data))
    all_eval_losses.append(eval_loss)
    print(f"Step: {i}\t train loss: {loss}\t eval loss: {eval_loss}")
```

```python
plt.title("Convergence of adamw (train loss)")
plt.plot(all_train_losses, label="train", lw=3)
plt.plot(
    jnp.arange(0, len(all_eval_losses) * N_FREQ_EVAL, N_FREQ_EVAL),
    all_eval_losses,
    label="test",
    lw=3,
)
plt.xlabel("steps")
plt.ylabel("loss")
plt.grid()
plt.legend(frameon=False)
plt.show()
```

# Text generation

Finally, after training, we use the generate function to let the NanoGPT model demonstrate its ability to create text that resembles Shakespeare, albeit in a miniature form.

```python
# Let's now generate some text
key, subkey = jax.random.split(key)
text = model.generate(key, 1000)[:, 0, 0].tolist()
print(decode(text))
```
