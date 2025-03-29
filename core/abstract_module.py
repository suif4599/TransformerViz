from abc import ABC, abstractmethod
import torch

class AbstractModule(ABC):
    @abstractmethod
    def forward(self, sentence: str) -> None:
        pass

    @abstractmethod
    def get_input(self) -> list[str]:
        pass

    @abstractmethod
    def get_output(self) -> list[str]:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    def get_attention_weights(self, 
                              key: str, 
                              position_mode: str, 
                              layer_mix_mode: str, 
                              head_mix_mode: str,
                              temperature: float) -> list[float | list[float]]:
        pass

    @abstractmethod
    def get_position_mode_list(self) -> list[str]:
        pass

    @abstractmethod
    def get_layer_mix_mode_list(self) -> list[str]:
        pass

    @abstractmethod
    def get_head_mix_mode_list(self) -> list[str]:
        pass

    @abstractmethod
    def get_n_head(self, 
                   position_mode: str, 
                   layer_mix_mode: str, 
                   head_mix_mode: str
                   ) -> int:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass