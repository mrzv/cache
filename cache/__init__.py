from inspect   import getfullargspec
from functools import wraps
from os.path   import join, exists, dirname, splitext
import string, os, pickle

import joblib

from   atomicwrites import atomic_write
import numpy as np

from   decorator import decorate


# memoize that uses joblib.hash for hashing arguments
def memoize(f):
    cache = {}
    def wrapper(f, *args, **kwargs):
        hashed_args = joblib.hash((args, kwargs))
        if hashed_args not in cache:
            cache[hashed_args] = f(*args, **kwargs)
        return cache[hashed_args]
    return decorate(f, wrapper)


class Storage:
    def __init__(self, directory):
        self.directory = directory

    def load(self, fn):
        extension = splitext(fn)[1]
        with open(fn, 'rb') as fo:
            if extension == '.npy':
                return np.load(fo)
            elif extension == '.npz':
                return dict(np.load(fo))       # dict to force read
            else:
                return pickle.load(fo)

    def save(self, fn, result):
        extension = splitext(fn)[1]
        with atomic_write(fn, mode='wb', overwrite=True) as fo:
            if extension == '.npy':
                np.save(fo, result)
            elif extension == '.npz':
                np.savez_compressed(fo, **result)
            else:
                pickle.dump(result, fo)

    def cache(self, fn_format_string, hash = [], transform = {}, verbose = True):
        arg_names = [x[1] for x in string.Formatter().parse(fn_format_string) if x[1] is not None]

        def decorator(f):
            argspec = getfullargspec(f)
            arg_indices = [argspec.args.index(name) for name in arg_names]

            def filename(*args, **kwargs):
                values = {}
                for (index, name) in zip(arg_indices, arg_names):
                    try:
                        value = args[index]
                    except IndexError:
                        value = kwargs[name]

                    if name in hash:
                        value = joblib.hash(value)
                    elif name in transform:
                        value = transform[name](value)

                    values[name] = value

                return join(self.directory, fn_format_string.format(**values))

            def load(*args, **kwargs):
                return self.load(filename(*args, **kwargs))

            def wrapper(f, *args, **kwargs):
                cache_fn = filename(*args, **kwargs)
                os.makedirs(dirname(cache_fn), exist_ok=True)

                if exists(cache_fn):
                    if verbose:
                        print("Loading from cache:", cache_fn)
                    return load(*args, **kwargs)

                result = f(*args, **kwargs)

                if isinstance(result,np.ndarray) or result:
                    self.save(cache_fn, result)

                return result

            decorated_wrapper = decorate(f, wrapper)

            decorated_wrapper.cache_filename = filename
            decorated_wrapper.load           = load

            return decorated_wrapper
        return decorator
