import asyncio
import time
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

        time.sleep(10)
        return ["dummy_response"]


class AsyncDummyResponse:
    """Implements LMIProtocol."""

    async def generate(
            self, 
            *args,
            inputs: Union[str, List[str]],
            parameters: "lmi.GenerationParameters",
            **kwargs
            ) -> Union[str, List[str]]:

        if not isinstance(inputs, List):
            inputs = [inputs]

        print(inputs)
        await asyncio.sleep(10)
        return ["dummy_response" for _ in inputs]




generator = DummyResponse().generate
agenerator = AsyncDummyResponse().generate
