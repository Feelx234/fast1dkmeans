import numba
import numpy as np


def is_jitlass(func):
    return str(type(func)) == "<class 'numba.experimental.jitclass.base.JitClassType'>"

def remove_numba(func, seen=None, allowed_packages=tuple()):
    #print(func, type(func))

    if seen is None:
        seen = {}
    if hasattr(func, "py_func"):
        #clean_up["self"] = func
        seen[func]=func.py_func
        func = func.py_func

    if isinstance(func, type(remove_numba)):
        to_iter = func.__globals__
        def set_func(key, value):
            to_iter[key] = value
    elif is_jitlass(func):
        new_methods = {}
        to_iter = func.class_type.jit_methods

        def set_func(key, value):
            new_methods[key] = value
    elif str(type(func)) == "<class 'module'>" and func.__package__ in allowed_packages:
        def set_func(key, value):
            setattr(func, key, value)
        to_iter = {key : getattr(func, key) for key in dir(func)}
    else:
        raise NotImplementedError(type(func))

    clean_up = iter_children(to_iter, seen, set_func, allowed_packages)
    if is_jitlass(func):
        return type(func.class_type.class_name, (), new_methods), clean_up
    return func, clean_up


def iter_children(to_iter, seen, set_func, allowed_packages=tuple()):
    clean_up = {}
    for key, maybe_func in to_iter.items():
        if isinstance(maybe_func, (list, dict, np.ndarray)):
            continue
        if isinstance(maybe_func, (int, float, str)):
            continue
        if str(type(maybe_func)) == "<class 'module'>" and maybe_func.__package__ in allowed_packages:
            #print("module", maybe_func)
            if maybe_func in seen:
                continue
            seen[maybe_func] = None
            #print(seen)
            non_numba_handle, handle_cleanup = remove_numba(maybe_func, seen, allowed_packages)
            clean_up["__module__"+key] = handle_cleanup
            continue


        if not (isinstance(maybe_func, (type(remove_numba), numba.core.registry.CPUDispatcher)) or is_jitlass(maybe_func)):
            continue
        if maybe_func in seen:
            #print("Seen")
            #print(maybe_func)
            clean_up[key] = maybe_func
            clean_up["__children__"+key] = {}
            set_func(key, seen[maybe_func])
            continue

        if hasattr(maybe_func, "py_func") or is_jitlass(maybe_func):
            non_numba_handle, handle_cleanup = remove_numba(maybe_func, seen, allowed_packages)
            clean_up[key] = maybe_func
            clean_up["__children__"+key] = handle_cleanup
            set_func(key, non_numba_handle)
    return clean_up


def restore_numba(func, clean_up, parent=None):
    #print(func)
    if hasattr(func, "py_func"):
        func =  func.py_func
    #print(func)
    if isinstance(func, type(restore_numba)):
        def set_func(key, value):
            func.__globals__[key] = value
        def get_func(key):
            return func.__globals__[key]
    elif str(type(func)) == "<class 'module'>":
        def set_func(key, value):
            setattr(func, key, value)
        def get_func(key):
            return getattr(func, key)
    elif str(type(func)) == "<class 'type'>":
        def set_func(key, value): # pylint: disable=unused-argument
            pass
            #print(key, value)
            #parent.class_type.jit_methods[key]=value
            #setattr(func, key, value)
        def get_func(key):
            return parent.class_type.jit_methods[key]
    else:
        raise NotImplementedError(str(type(func)) + " "+ str(func))
    restore_children(clean_up, get_func, set_func)

def restore_children(clean_up, get_func, set_func):
    for key, value in clean_up.items():
        if key.startswith("__children__"):
            continue
        if key.startswith("__module__"):
            short_key = key[len("__module__"):]
            restore_numba(get_func(short_key), clean_up[key])
            continue
        if get_func(key) is value and not hasattr(value, "py_func"):
            continue
        restore_numba(get_func(key), clean_up["__children__"+key], value)
        set_func(key, value)


def remove_from_class(cls, allowed_packages=tuple()):
    clean_ups = {}
    seen = {}
    for key, value in cls.__dict__.items():
        if hasattr(value, "__module__"):
            _, handle_cleanup = remove_numba(value, seen, allowed_packages)
            clean_ups[key] = (value, handle_cleanup)
            #print(key)
            #print(value.__module__)
    return clean_ups


def restore_to_class(clean_ups):
    for _, (handle, clean_up) in clean_ups.items():
        restore_numba(handle, clean_up)
        #cls.__dict__[key]=handle



class NoNumbaTestCase:
    def setUp(self):
        for cls in self.__class__.__bases__:
            if cls is NoNumbaTestCase:
                continue
            self.cleanup = remove_from_class(cls, allowed_packages=["fast1dkmeans"])
            break

    def tearDown(self) -> None:
        restore_to_class(self.cleanup)
