# Cache

A simple caching library that uses function arguments in the filename that
stores the returned value. The filename extension prescribes whether the
results are pickled (default) or saved using NumPy (as arrays or zipped
archives).

## Install

```
pip install git+https://github.com/mrzv/cache.git
```

## Examples

```python
from cache import Storage

mem = Storage('dir/to/store/cache')

@mem.cache('image-{shape}.npy')
def f(shape):
    return np.random.random(shape)

@mem.cache('images-{shape}-{size}.npz')
def g(shape, size):
    d = {}
    for i in range(size):
        d[str(i)] = np.random.random(shape)
    return d

@mem.cache('plain-{x}')
def h(x):
    return [x]

@mem.cache('hashed-{x}', hash = ['x'])
def k(x):
    return x
```
