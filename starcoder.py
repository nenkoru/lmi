from typing import Union, Iterable, List
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
