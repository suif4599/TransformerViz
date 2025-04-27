from collections.abc import Callable
from types import TracebackType
import traceback


class Catch:
    def __init__(self, callback: Callable[[type, object, TracebackType], None] = None):
        self.callback = callback
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type: type, exc_value: object, tb: TracebackType):
        if exc_type is not None:
            if self.callback:
                self.callback(exc_type, exc_value, tb)
            else:
                traceback.print_exception(exc_type, exc_value, tb)
        return True