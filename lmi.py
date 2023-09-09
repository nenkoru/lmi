from dataclasses import dataclass
from typing import Protocol, Union, Iterable


@dataclass
class GenerationParameters:
    temperature: Union[float, None] = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: Union[float, None] = 1
    max_tokens: Union[int, None] = 16
    return_prompt: bool = True
    do_sample: bool = True
    stop_token: Union[str, None] = None

    def __hash__(self):
        return hash("".join(self.__dict__))


class LMIProtocol(Protocol):
    """Stands for Language Model Interface."""
    
    def generate(
            self, 
            *args,
            inputs: Union[str, Iterable[str]], 
            params: GenerationParameters, 
            **kwargs
            ) -> Union[str, Iterable[str]]:
        raise NotImplementedError("Need to implement this method")

