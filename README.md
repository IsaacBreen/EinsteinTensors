DEPRECATED: use https://github.com/IsaacBreen/einexpr instead.

# Installation

```bash
pip install git+https://github.com/IsaacBreen/EinsteinTensors
```

# Einstein Tensors

Einstein Tensors is a small DSL for writing neural networks more concisely. It compiles into efficient JAX code and takes care of a lot of things for you, including parameter initialization.

Informally, the rules of the DSL are as follows:

- Indices always follow tensor symbols.
    - All dimensions must be indexed. E.g. if `X` is a 2-dimensional tensor, then `X[i, j]` a valid expression, but `X[i]` and `X` are not.
- Assignments are mapped over the indices on the left-hand side automatically.
    - E.g. `for i in range(10): y[i] = x[i]` is replaced by `y[i] = x[i]`.
- When an index appears on the right-hand side of an assignment but not the left-hand side, this implies additive reduction.
    - E.g. if `X` is a 2D tensor (or a matrix), then `y[i] = X.reduce_sum(axis=0)` is equivalent to `y[i] = X[i,j]`.

If you like einsum, you'll love this library.

# Examples

## Linear layer

```python
from einstein_tensors import jax_codegen

einstein_code = """
function Linear(x)
    y[i] = activation(x[j] * w[j, i] + b[i])
    return y[i]
end
"""

jax_code = jax_codegen(einstein_code)

print(jax_code)
```
Output:
```
class Linear:
    def __init__(self, i):
        self.old_init = self.init
        self.init = lambda *args, **kwargs: self.old_init(*args, i=i, **kwargs)

    @staticmethod
    def init(key, j, i, **kwargs):
        keys = jax.random.split(key, 2)
        return {
            "w_j_i": jax.random.normal(keys[0], shape=[j, i]),
            "b_i":   jax.random.normal(keys[1], shape=[i])
        }

    @staticmethod
    @jit
    @lambda apply: vmap(apply, in_axes=(0, None))
    def apply(x_j, params):
        y_i = activation((jnp.einsum('j,ji->i', x_j, params['w_j_i'])) + params['b_i'])
        return y_i
```

## Multihead self-attention (transformer)

```python
from einstein_tensors import jax_codegen

einstein_code = """
function MultiheadSelfAttention(x)
    x[t,i] = x[t,i] * gx[i]
    q[h,t,j] = q[h,j,i] * x[t,i]
    k[h,t,j] = k[h,j,i] * x[t,i]
    v[h,t,j] = v[h,j,i] * x[t,i]
    a[h,t_1,t_2] = jnp.exp(q[h,t_1,j] * k[h,t_2,j])
    u[t,k] = activation(wu[h,j,k] * a[h,t,t_2] * v[h,t_2,j] + bu[k])
    z[t,i] = wz[t,k] * u[t,k] + bz[i]
    return z[t,i]
end
"""

jax_code = jax_codegen(einstein_code)

print(jax_code)
```
Output:
```
class MultiheadSelfAttention:
    def __init__(self, h, k, j):
        self.old_init = self.init
        self.init = lambda *args, **kwargs: self.old_init(*args, h=h, k=k, j=j, **kwargs)

    @staticmethod
    def init(key, i, t, h, k, j, **kwargs):
        keys = jax.random.split(key, 8)
        return {
            "gx_i":     jax.random.normal(keys[0], shape=[i]),
            "q_h_j_i":  jax.random.normal(keys[1], shape=[h, j, i]),
            "k_h_j_i":  jax.random.normal(keys[2], shape=[h, j, i]),
            "v_h_j_i":  jax.random.normal(keys[3], shape=[h, j, i]),
            "wu_h_j_k": jax.random.normal(keys[4], shape=[h, j, k]),
            "bu_k":     jax.random.normal(keys[5], shape=[k]),
            "wz_t_k":   jax.random.normal(keys[6], shape=[t, k]),
            "bz_i":     jax.random.normal(keys[7], shape=[i])
        }

    @staticmethod
    @jit
    @lambda apply: vmap(apply, in_axes=(0, None))
    def apply(x_t_i, params):
        x_t_i = jnp.einsum('ti,i->ti', x_t_i, params['gx_i'])
        q_h_t_j = jnp.einsum('hji,ti->htj', params['q_h_j_i'], x_t_i)
        k_h_t_j = jnp.einsum('hji,ti->htj', params['k_h_j_i'], x_t_i)
        v_h_t_j = jnp.einsum('hji,ti->htj', params['v_h_j_i'], x_t_i)
        a_h_t_t = jnp.exp(jnp.einsum('htj,haj->hta', q_h_t_j, k_h_t_j))
        u_t_k = activation((jnp.einsum('hjk,hat,htj->ak', params['wu_h_j_k'], a_h_t_t, v_h_t_j)) + params['bu_k'][None, :])
        z_t_i = (jnp.einsum('tk,tk->t', params['wz_t_k'], u_t_k))[:, None] + params['bz_i'][None, :]
        return z_t_i
```

```python
import jax
import jax.numpy as jnp

def activation(x):
    return jnp.maximum(x, 0)

exec(jax_code)

attention = MultiheadSelfAttention(k=2, j=2, h=2)
params = attention.init(jax.random.PRNGKey(0), i=2, t=2)
x = jnp.array([[[0,1], [2,3]]])
out = attention.apply(x, params)
print(out)
```
Output:
```
[[[  -400.6252    -398.97708]
  [-22083.512   -22081.863  ]]]
```

# Motivation

At their core, neural networks are remarkably simple. Even modern, state-of-the-art architectures such as the transformer can be [described mathematically in about half a page of A4 paper](https://johnthickstun.com/docs/transformers.pdf). Implementations, however, are often quite complex, with modern tensor libraries necessitating a lot of unnecessary code that obfuscates the rather simple underlying concepts. This library aims to make neural networks easier to write and more readable.
