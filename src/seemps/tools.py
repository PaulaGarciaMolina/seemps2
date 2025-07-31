from __future__ import annotations
from math import cos, sin, sqrt
from typing import Any
from .typing import DenseOperator


class InvalidOperation(TypeError):
    """Exception for operations with invalid or non-matching arguments."""

    def __init__(self, op: str, *args: Any):
        super().__init__(
            f"Invalid operation {op} between arguments of types {(type(x) for x in args)}"
        )


def take_from_list(O: list[Any] | Any, i: int):
    if isinstance(O, list):
        return O[i]
    else:
        return O


class Logger:
    active: bool = False

    def __call__(self, *args: Any, **kwdargs: Any):
        pass

    def __enter__(self) -> Logger:
        return self

    def __exit__(self, exc_type, exc_value, traceback):  # pyright: ignore[reportMissingParameterType]
        pass

    def __bool__(self) -> bool:
        return False

    def close(self):
        pass


DEBUG = 0
PREFIX = ""
NO_LOGGER = Logger()


class VerboseLogger(Logger):
    old_prefix: str
    level: int
    active: bool

    def __init__(self, level: int):
        global PREFIX
        self.old_prefix = PREFIX
        self.level = level
        if level <= DEBUG:
            self.active = True
            PREFIX = PREFIX + " "
        else:
            self.active = False

    def __bool__(self) -> bool:
        return self.active

    def __enter__(self) -> Logger:
        super().__enter__()
        return self

    def __call__(self, *args: Any, **kwdargs: Any):
        if self.active:
            txt = " ".join([str(a) for a in args])
            txt = " ".join([PREFIX + a for a in txt.split("\n")])
            print(txt, **kwdargs)

    def __exit__(self, exc_type, exc_value, traceback):  # pyright: ignore[reportMissingParameterType]
        self.close()
        super().__exit__(exc_type, exc_value, traceback)

    def close(self):
        global PREFIX
        PREFIX = self.old_prefix


def make_logger(level: int = 1) -> Logger:
    """Create an object that logs debug information. This object has a property
    `active` that determines whether logging is working. It also has a `__call__`
    method that allows invoking the object with the information to log, working
    as if it were a `print` statement."""
    if level > DEBUG:
        return NO_LOGGER
    return VerboseLogger(level)


# TODO: Find a faster way to do logs. Currently `log` always gets called
# We should find a way to replace calls to log in the code with an if-statement
# that checks `DEBUG`
def log(*args: Any, debug_level: int = 1) -> None:
    """Optionally log informative messages to the console.

    Logging is only active when :var:`~seemps.tools.DEBUG` is True or an
    integer above or equal to the given `debug_level`.

    Parameters
    ----------
    *args : str
        Strings to be output
    debug_level : int, default = 1
        Level of messages to log
    """
    if DEBUG and (DEBUG is True or DEBUG >= debug_level):
        print(*args)