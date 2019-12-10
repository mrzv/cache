from cache import Storage

import numpy as np

mem = Storage('tmp')

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

print(f((50,50)))
print(g((50,50), 20))
print(h('hello'))

print(g.cache_filename((50,50), 20))
print(h.cache_filename('hello'))

print(h.load('hello'))

print(k(list(range(100))))
