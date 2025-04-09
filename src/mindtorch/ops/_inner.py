"""inner ops"""
from mindtorch.executor import execute

def cast(input, dtype):
    return execute('cast', input, dtype)

def assign(input, other):
    return execute('assign', input, other)

def identity(input):
    return execute('identity', input)

__all__ = ['cast', 'assign', 'identity']
