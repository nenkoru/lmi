import importlib
import argparse

from typing import Union
from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel

import lmi


app = FastAPI(
    title="FastAPI",
    description="A drop-in replacement for HuggingFace API definition",
    version="0.1.0",
)


class ParametersDTO(BaseModel):
    top_k: int
    top_p: float
    temperature: float
    repetition_penalty: float
    max_new_tokens: int
    max_time: float
    return_full_text: bool
    num_return_sequences: int
    do_sample: bool


class OptionsDTO(BaseModel):
    use_cache: bool
    wait_for_model: bool


class RequestDataclass(BaseModel):
    inputs: str
    parameters: ParametersDTO = ParametersDTO(
            top_k=0, 
            top_p=1.0, 
            temperature=1.0,
            repetition_penalty=1,
            max_new_tokens=16,
            return_full_text=True,
            num_return_sequences=1,
            do_sample=True,
            max_time=10
    )

    options: Union[OptionsDTO, None] = None


def generate_response(inputs: str, parameters: lmi.GenerationParameters):
    return NotImplementedError()

@app.post("/generate")
async def generate(request: RequestDataclass):
    """Generate text using a model defined in LMI.

    The request should follow the following schema:
    {
    "inputs": "string",
    "parameters": {
        "top_k": "int",
        "top_p": "float",
        "temperature": "float",
        "repetition_penalty": "float",
        "max_new_tokens": "int",
        "max_time": "float",
        "return_full_text": "bool",
        "num_return_sequences": "int",
        "do_sample": "bool"
    },
    "options": {
        "use_cache": "bool",
        "wait_for_model": "bool"
    }
   }
   More on that: https://huggingface.co/docs/api-inference/detailed_parameters#text-generation-task
    
    """
    print(request)
    parameters = lmi.GenerationParameters(
            temperature=request.parameters.temperature,
            top_p=request.parameters.top_p,
            top_k=request.parameters.top_k,
            repetition_penalty=request.parameters.repetition_penalty,
            max_tokens=request.parameters.max_new_tokens,
            return_prompt=request.parameters.return_full_text,
            do_sample=request.parameters.do_sample
    )
    return generate_response(inputs=request.inputs, parameters=parameters)

   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--lmi", type=str)
    args = parser.parse_args()
    lmi_module = importlib.import_module(args.lmi)
    generate_response = lmi_module.generate

    uvicorn.run(app, host=args.host, port=args.port)
