from joblib import load
from os.path import join, exists, dirname

__all__ = [
    #"Pipe"
]

__pipes_folder = join(dirname(__file__), "pipes")
__default_pipes = {}

def load_pipe(default_name_or_custom_path):
    if default_name_or_custom_path in __default_pipes:
        return load(__default_pipes[default_name_or_custom_path])
    elif default_name_or_custom_path is None:
        return None
    elif exists(default_name_or_custom_path):
        return load(default_name_or_custom_path)
    else:
        raise OSError(f"Pipe not found: {default_name_or_custom_path}")

class Pipe():
    def __init__(self):
        raise NotImplementedError

    def __call__(self, X):
        if not isinstance(X, list):
            X = [X]
        results = []
        raise NotImplementedError