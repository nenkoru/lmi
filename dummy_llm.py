from typing import Union, Iterable, List



class DummyResponse:
    """Implements LMIProtocol."""

    def generate(
            self, 
            *args,
            inputs: Union[str, Iterable[str]],
            parameters: "lmi.GenerationParameters",
            **kwargs
            ) -> Union[str, List[str]]:

        return ["dummy_response"]




generate = DummyResponse().generate
