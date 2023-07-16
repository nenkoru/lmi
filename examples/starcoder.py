import asyncio

from typing import Union, Iterable, List, Callable
from collections.abc import Iterable as abc_Iterable

import transformers
import ctranslate2


class CT2Generator:
    """Implements LMIProtocol."""

    def __init__(self, 
                 *,
                 generator: "ctranslate2.Generator", 
                 tokenizer: "transformers.AutoTokenizer"
                 ):
        self._generator = generator
        self._tokenizer = tokenizer

    def generate(
            self, 
            *args,
            inputs: str,
            parameters: "lmi.GenerationParameters",
            **kwargs
            ) -> Union[str, List[str]]:

        encoded_prompt = self._tokenizer.encode(inputs)
        tokens = self._tokenizer.convert_ids_to_tokens(encoded_prompt)
        results = self._generator.generate_batch(
                [tokens],
                max_length=parameters.max_tokens,
                include_prompt_in_result=False,
                sampling_topp=parameters.top_p,
                sampling_topk=parameters.top_k,
                sampling_temperature=parameters.temperature,
                end_token=parameters.stop_token
        )
        texts = [self._tokenizer.decode(result.sequences_ids[0]) for result in results]
        return texts

class AsyncCT2Generator:
    """Implements LMIProtocol."""

    def __init__(self, 
                 *,
                 generator: "ctranslate2.Generator", 
                 tokenizer: "transformers.AutoTokenizer"
                 ):
        self._generator = generator
        self._tokenizer = tokenizer

    async def generate(
            self, 
            *args,
            inputs: Union[str, List[str]],
            parameters: "lmi.GenerationParameters",
            callback: Union[Callable, List[Callable]] = None,
            **kwargs
            ) -> Union[str, List[str]]:

        if not isinstance(inputs, List):
            inputs = [inputs]

        if not isinstance(callback, List) and callback is not None:
            callback = [callback]

        if callback is not None:
            assert len(inputs) == len(callback), "Inputs do not equal to callbacks"


        encoded_prompts = [self._tokenizer.encode(input) for input in inputs]
        tokens = [self._tokenizer.convert_ids_to_tokens(encoded_prompt) for encoded_prompt in encoded_prompts]
        async_results = self._generator.generate_batch(
                tokens,
                max_length=parameters.max_tokens,
                include_prompt_in_result=False,
                sampling_topp=parameters.top_p,
                sampling_topk=parameters.top_k,
                sampling_temperature=parameters.temperature,
                end_token=parameters.stop_token,
                asynchronous=True,
        )
        print(async_results)
        if callback is not None:
            async def wrap_to_decode(result):
                result = await result
                return self._tokenizer.decode(result.sequences_ids[0])

            async def poll_wait_for_completion(
                    async_result: ctranslate2.AsyncGenerationResult, 
                    poll_interval=0.01
                    ):
                while not async_result.done():
                    await asyncio.sleep(poll_interval)
                return async_result.result()

            for async_result, _callback in zip(async_results, callback):
                _callback(wrap_to_decode(poll_wait_for_completion(async_result)))



_ct2_generator = ctranslate2.Generator(
        "../starcoderplus_ct2_int8", 
        device="cuda", 
        compute_type="int8"
)
_tokenizer = transformers.AutoTokenizer.from_pretrained(
        "bigcode/starcoderplus"
)
generator = CT2Generator(
        generator=_ct2_generator, 
        tokenizer=_tokenizer,
).generate

agenerator = AsyncCT2Generator(
        generator=_ct2_generator, 
        tokenizer=_tokenizer,
).generate


