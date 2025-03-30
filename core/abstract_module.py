from abc import ABC, abstractmethod

class AbstractModule(ABC):
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