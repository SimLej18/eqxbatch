# eqxbatch

Batched evaluation of [Equinox](https://github.com/patrick-kidger/equinox) modules: a
`vmap` that lives in the PyTree.

```bash
pip install eqxbatch
```

## Why

Equinox's position is that a `Module` *is* a PyTree, so you batch by transforming
*functions* — `eqx.filter_vmap` at the call site — rather than by wrapping modules. That is
the right answer whenever the module is the outermost object you call, and you should keep
doing it.

It stops being available when the batched module has to remain a **node inside a larger
tree** that is driven through one generic entry point:

```python
Sum(Batched(inner_a), inner_b)     # no call site to hang a filter_vmap on
```

Here the vmap has to live in the tree itself. `Batched` is that node.

## Usage

```python
import equinox as eqx
import jax
from eqxbatch import Batched, broadcast, stack

keys = jax.random.split(jax.random.key(0), 8)
mlps = [eqx.nn.MLP(2, 2, 8, 2, key=k) for k in keys]

ensemble = Batched(stack(*mlps))            # list[Module] -> one batched Module
x = jax.random.normal(jax.random.key(1), (2,))

ensemble(x).shape                           # (8, 2) -- one shared input, eight models
ensemble[3]                                 # mlps[3], as an ordinary module again
len(ensemble)                               # 8
```

One input per batch element instead of a shared one:

```python
per_member = Batched(stack(*mlps), arg_axes=eqx.if_array(0))
xs = jax.random.normal(jax.random.key(1), (8, 2))
per_member(xs).shape                        # (8, 2)
```

Same starting parameters, diverging during training:

```python
ensemble = Batched(broadcast(eqx.nn.MLP(2, 2, 8, 2, key=keys[0]), 8))
```

Or build it the idiomatic Equinox way and wrap the result:

```python
ensemble = Batched(eqx.filter_vmap(lambda k: eqx.nn.MLP(2, 2, 8, 2, key=k))(keys))
```

### Attribute forwarding

Any public attribute of the wrapped module is re-evaluated under the same vmap, so a module
only has to implement a method once, unbatched, for it to work through any number of batch
layers:

```python
ensemble.some_method(y)     # (8, ...) -- called per batch element
ensemble.some_property      # (8, ...) -- evaluated per batch element
```

Arguments are shared across the batch by default; pass `arg_axes` to override:

```python
ensemble.some_method(ys, arg_axes=(eqx.if_array(0),))
```

For anything that spans several attributes at once, drop to the primitive:

```python
ensemble.map(lambda m: (m.weight @ m.bias, m.activation_count))
```

Unlike a raw `jax.vmap`, `map` tolerates non-array leaves in the output: they are
partitioned out and returned unbatched.

### Nesting and composition

`Batched` is an `eqx.Module`, so it nests and is transparent to `jit`, `grad` and outer
`vmap`s:

```python
Batched(Batched(broadcast(broadcast(model, 2), 3)))(x).shape   # (3, 2, ...)
eqx.filter_grad(loss)(ensemble, x)                             # gradients keep the batch axis
```

## API

| | |
|---|---|
| `Batched(inner, *, in_axes, arg_axes, axis_size)` | the batched node |
| `Batched.map(fn, *args, arg_axes=...)` | run `fn(inner, *args)` per batch element |
| `Batched[i]`, `len(Batched)` | unstack one element, batch size |
| `stack(*modules)` | `list[Module] -> Module` with a leading axis |
| `broadcast(module, size)` | one module repeated `size` times |

`in_axes` and `arg_axes` take `eqx.filter_vmap` specs: an int, `None`, a callable such as
`eqx.if_array(0)`, or a pytree prefix for per-leaf control.

## Caveats

- `eqx.filter_vmap` does not accept keyword arguments in the vmapped function. `Batched`
  closes over any `**kwargs` you pass, so they are shared across the batch and cannot be
  given an axis spec.
- Under `jit` the wrapper is free — it compiles to exactly the program a hand-written
  `vmap` would produce. In eager mode it costs a fixed overhead per call plus one copy of
  the output, since `filter_vmap` always applies a `moveaxis` that XLA elides but the
  op-by-op path does not.
- `__getitem__` and `__len__` address the outermost batch axis only.

## License

MIT