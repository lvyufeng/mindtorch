#!/usr/bin/env python
# -*- coding: utf-8 -*-

def is_tracing():
    return False

def is_scripting():
    return False

def script(obj, optimize=None, _frames_up=0, _rcb=None, example_inputs=None):
    return obj

def ignore(drop=False, **kwargs):

    if callable(drop):
        return drop

    def decorator(fn):
        return fn

    return decorator

def _overload_method(func):
    pass

def interface(obj):
    pass
