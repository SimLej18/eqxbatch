"""
Batched evaluation of Equinox modules.

Equinox's position is that a `Module` *is* a PyTree, so you batch by transforming
*functions* -- `eqx.filter_vmap` at the call site -- rather than by wrapping modules. That
is the right answer whenever the module is the outermost object you call.

It stops being available when the batched module has to remain a *node inside a larger
tree* that is driven through one generic entry point. There is then no call site to hang a
`filter_vmap` on, and the vmap has to live in the tree itself.

`Batched` is that node.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

__version__ = "0.1.0"
__all__ = ["Batched", "broadcast", "stack"]

_T = TypeVar("_T")
_UNSET = object()


def stack(*modules: _T) -> _T:
    """
    Stack like-structured modules into one whose array leaves gain a leading axis.

    Static (non-array) leaves are taken from the first module; they are assumed to agree.
    This is the `list[Module] -> Module` direction, complementary to building an ensemble
    by vmapping a constructor.

    ```python
    ensemble = Batched(stack(*[eqx.nn.MLP(2, 2, 8, 2, key=k) for k in keys]))
    ```
    """
    if not modules:
        raise ValueError("`stack` needs at least one module.")
    arrays, statics = zip(*(eqx.partition(m, eqx.is_array) for m in modules))
    return eqx.combine(jtu.tree_map(lambda *xs: jnp.stack(xs), *arrays), statics[0])


def broadcast(module: _T, size: int) -> _T:
    """
    Repeat every array leaf of `module` `size` times along a new leading axis.

    Use this when every batch element starts from the same parameters and will diverge
    during training.
    """
    arrays, static = eqx.partition(module, eqx.is_array)
    return eqx.combine(
        jtu.tree_map(lambda x: jnp.broadcast_to(x, (size, *x.shape)), arrays), static
    )


def _resolve(spec: Any, tree: Any) -> Any:
    """Turn an axis spec (int, None, callable, or pytree prefix) into one axis per leaf."""
    if spec is None or isinstance(spec, int):
        return jtu.tree_map(lambda _: spec, tree)
    if callable(spec):
        return jtu.tree_map(spec, tree)
    return jtu.tree_map(_resolve, spec, tree, is_leaf=lambda x: x is None)


class Batched(eqx.Module):
    """
    A module evaluated for several parameter sets at once.

    Calling a `Batched` runs the wrapped module once per batch element under
    `eqx.filter_vmap`, and every other public attribute is forwarded through the same vmap.
    A `Batched` is itself an `eqx.Module`, so it nests, composes, and is transparent to
    `jit`/`grad`.

    ```python
    ensemble = Batched(stack(mlp_a, mlp_b, mlp_c))
    ensemble(x)                 # (3, out)  -- one shared input, three models
    ensemble[1]                 # mlp_b, as an ordinary module again
    ```

    **Arguments:**

    - `inner`: the module to batch. Its array leaves are expected to already carry a
      leading batch axis wherever `in_axes` says so; build it with `stack`, `broadcast`, or
      by vmapping a constructor.
    - `in_axes`: axis spec over `inner`, in `eqx.filter_vmap` form -- an int, `None`, a
      callable such as `eqx.if_array(0)` (the default: every array leaf batched, everything
      else shared), or a pytree prefix for per-leaf control.
    - `arg_axes`: default axis spec for call arguments. `None` (the default) shares every
      argument across the batch; `eqx.if_array(0)` batches them.
    - `axis_size`: the batch size. Only strictly required when nothing is batched, but
      supplying it always makes the fully-shared case work by broadcasting.
    """

    inner: Any
    in_axes: Any = eqx.field(static=True)
    arg_axes: Any = eqx.field(static=True)
    axis_size: int | None = eqx.field(static=True)

    def __init__(
        self,
        inner: Any,
        *,
        in_axes: Any = eqx.if_array(0),
        arg_axes: Any = None,
        axis_size: int | None = None,
    ):
        self.inner = inner
        self.in_axes = in_axes
        self.arg_axes = arg_axes
        self.axis_size = axis_size

    def map(self, fn: Callable[..., _T], *args: Any, arg_axes: Any = _UNSET) -> _T:
        """
        Run `fn(inner, *args)` once per batch element.

        This is the only primitive; `__call__` and the attribute forwarding are sugar over
        it. Unlike a raw `jax.vmap`, `fn` may return non-array leaves -- a Module, a Python
        object -- which are partitioned out and returned unbatched.

        **Arguments:**

        - `fn`: callable taking one *unbatched* module, then `*args`.
        - `arg_axes`: axis spec for `args`, overriding this module's default.
        """
        if arg_axes is _UNSET:
            arg_axes = self.arg_axes
        return eqx.filter_vmap(
            lambda m, a: fn(m, *a),
            in_axes=(self.in_axes, arg_axes),
            axis_size=self.axis_size,
        )(self.inner, args)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.map(lambda m, *a: m(*a, **kwargs), *args)

    def __getitem__(self, i: int) -> Any:
        """Recover batch element `i` as an ordinary, unbatched module."""
        axes = _resolve(self.in_axes, self.inner)
        return jtu.tree_map(
            lambda x, ax: x if ax is None else jnp.take(x, i, axis=ax), self.inner, axes
        )

    def __len__(self) -> int:
        if self.axis_size is not None:
            return self.axis_size
        axes = _resolve(self.in_axes, self.inner)
        sizes = {
            x.shape[ax]
            for x, ax in zip(
                jtu.tree_leaves(self.inner),
                jtu.tree_leaves(axes, is_leaf=lambda a: a is None),
            )
            if ax is not None
        }
        if len(sizes) != 1:
            raise ValueError(
                f"Cannot infer the batch size: found {sizes or 'no batched leaf'}. "
                "Pass `axis_size` explicitly."
            )
        return sizes.pop()

    def __getattr__(self, name: str) -> Any:
        """
        Forward any public attribute of the wrapped module through this module's vmap.

        Names defined on `Batched` itself -- `inner`, `in_axes`, `arg_axes`, `axis_size`,
        `map` -- always win, and private names are never forwarded.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            module = object.__getattribute__(self, "inner")
        except AttributeError:  # accessed before __init__ set `inner`
            raise AttributeError(name) from None

        # Peel nested batch layers: they synthesise attributes through `__getattr__` rather
        # than exposing class descriptors, so an outer layer cannot see an inner one's
        # forwarded attributes by looking at `type(inner)` alone.
        while isinstance(module, Batched):
            module = module.inner
        descriptor = getattr(type(module), name, None)

        if callable(descriptor):
            return lambda *args, arg_axes=_UNSET, **kwargs: self.map(
                lambda m, *a: getattr(m, name)(*a, **kwargs), *args, arg_axes=arg_axes
            )

        # A property must be evaluated *under* the vmap, never on the batched leaves: one
        # that reduces (say `jnp.prod(self.scales)`) would otherwise fold the batch axis
        # into its result. A plain field already carries the batch axis, and going through
        # the vmap is a no-op for it, so both take the same path.
        return self.map(lambda m: getattr(m, name))