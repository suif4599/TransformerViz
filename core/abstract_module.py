from abc import ABC, abstractmethod, ABCMeta
from types import TracebackType
from html import escape
import traceback


class TemplateModule:
    def __init__(self, exc_type: type, exc_value: object, tb: TracebackType):
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.tb = tb

    def forward(self, sentence: str) -> None:
        pass

    def get_sentence(self, position_mode: str) -> tuple[list[str], list[str]]:
        return [self.exc_type.__name__], [self.exc_type.__name__]

    def get_name(self) -> str:
        return "Unknown"

    def get_description(self) -> str:
        # format the error message into html
        exc_type_str = escape(f"{self.exc_type.__module__}.{self.exc_type.__name__}")
        exc_value_str = escape(str(self.exc_value)) if self.exc_value else ""
        tb_lines_raw = traceback.format_tb(self.tb)[1:]  # Skip the first line which is the wrapper
        tb_lines = []
        for line in tb_lines_raw:
            tb_lines.extend(line.splitlines())
        formatted_tb = "".join([f"<p>{escape(line)}</p>" for line in tb_lines])
        return f"""
        <div style="
            font-family: 'Courier New', monospace;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 15px;
            margin: 10px;
            color: #333;
        ">
            <div style="
                color: #c7254e;
                padding: 10px;
                border-radius: 3px;
                margin-bottom: 10px;
                font-weight: bold;
            ">
                {exc_type_str}
            </div>

            <div style="
                margin-bottom: 15px;
                padding: 10px;
                border-left: 4px solid #3498db;
            ">
                <span style="color: #2c3e50;">Message: </span>
                <span style="color: blue;">{exc_value_str}</span>
            </div>

            <div style="
                color: black;
                padding: 8px;
                border-radius: 3px;
                margin-bottom: 10px;
            ">
                Stack Trace:
            </div>

            <div style="
                padding: 10px;
                border-radius: 3px;
                line-height: 1.6;
                white-space: pre-wrap;
            ">
                {formatted_tb}
            </div>
        </div>
        """
        
    def get_attention_weights(self, 
                              key: int, 
                              position_mode: str, 
                              layer_mix_mode: str, 
                              head_mix_mode: str,
                              temperature: float) -> list[float | list[float]]:
        return [0.]

    def get_position_mode_list(self) -> list[str]:
        return ["Unknown"]

    def get_layer_mix_mode_list(self) -> list[str]:
        return ["Unknown"]

    def get_head_mix_mode_list(self) -> list[str]:
        return ["Unknown"]

    def get_n_head(self, 
                   position_mode: str, 
                   layer_mix_mode: str, 
                   head_mix_mode: str
                   ) -> int:
        return 1

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def get_other_info(self) -> dict[str, list[str] | None] | None:
        return None
    
    def set_other_info(self, info: dict[str, str]) -> None:
        pass

class NoExceptionMeta(ABCMeta):
    """
    When a class raises an exception in __init__, 
    it will be caught and becomes the description of the class.
    """
    def __new__(cls, name, bases, attrs):
        if name == "AbstractModule":
            return super().__new__(cls, name, bases, attrs)
        if '__init__' in attrs:
            original_init = attrs['__init__']
            def new_init(self, *args, **kwargs):
                try:
                    original_init(self, *args, **kwargs)
                except Exception as e:
                    self.ERROR_DESCRIPTION = TemplateModule(
                        exc_type=type(e),
                        exc_value=e,
                        tb=e.__traceback__
                    )
        else:
            def new_init(self, *args, **kwargs):
                try:
                    try:
                        init = super(cls, self).__init__
                    except:
                        # If the class just doesn't have a __init__ method
                        return
                    init(*args, **kwargs)
                except Exception as e:
                    self.ERROR_DESCRIPTION = TemplateModule(
                        exc_type=type(e),
                        exc_value=e,
                        tb=e.__traceback__
                    )
        attrs['__init__'] = new_init
        for key, value in attrs.items():
            if not key.startswith('__') and key in TemplateModule.__dict__:
                if callable(value):
                    def wrapper(self, *args, __KEY=key, __RAW=value, **kwargs):
                        if hasattr(self, 'ERROR_DESCRIPTION'):
                            return self.ERROR_DESCRIPTION.__getattribute__(__KEY)(*args, **kwargs)
                        return __RAW(self, *args, **kwargs)
                    attrs[key] = wrapper
        return super().__new__(cls, name, bases, attrs)
            

class AbstractModule(ABC, metaclass=NoExceptionMeta):
    @abstractmethod
    def forward(self, sentence: str) -> None:
        "Process the input sentence and store the result"
        pass

    @abstractmethod
    def get_sentence(self, position_mode: str) -> tuple[list[str], list[str]]:
        "Return (target sentence, source sentence)"
        pass

    @abstractmethod
    def get_name(self) -> str:
        "Return the name of the module"
        pass

    @abstractmethod
    def get_description(self) -> str:
        "Return the description of the module"
        pass

    @abstractmethod
    def get_attention_weights(self, 
                              key: int, 
                              position_mode: str, 
                              layer_mix_mode: str, 
                              head_mix_mode: str,
                              temperature: float) -> list[float | list[float]]:
        """Return the attention weights for the given parameters,\n
        if there are multiple heads, return a list of lists,\n
        if there is only one head, return a list of floats"""
        pass

    @abstractmethod
    def get_position_mode_list(self) -> list[str]:
        "Return a list of position modes"
        pass

    @abstractmethod
    def get_layer_mix_mode_list(self) -> list[str]:
        "Return a list of layer mix modes"
        pass

    @abstractmethod
    def get_head_mix_mode_list(self) -> list[str]:
        "Return a list of head mix modes"
        pass

    @abstractmethod
    def get_n_head(self, 
                   position_mode: str, 
                   layer_mix_mode: str, 
                   head_mix_mode: str
                   ) -> int:
        "Return the number of heads for the given parameters"
        pass

    @abstractmethod
    def load(self) -> None:
        "Load the module and prepare it for use"
        pass

    @abstractmethod
    def unload(self) -> None:
        "Unload the module and free up resources"
        pass

    def get_other_info(self) -> dict[str, list[str] | None] | None:
        """Return other information about the module,\n
        such as `target language` for translation models.\n
        let value be None if you need to to make it a input\n"""
        return None
    
    def set_other_info(self, info: dict[str, str]) -> None:
        """Set other information about the module,\n
        such as `target language` for translation models\n"""
        pass