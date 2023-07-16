import importlib
import argparse
import asyncio

from typing import Union, List
from dataclasses import dataclass
from collections import Counter

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import lmi


REQUESTS_BATCH_PROCESS_WINDOW_SECONDS = 0.5
REQUESTS_BATCH_SIZE = 50
GENERATE_QUEUE = asyncio.Queue()

app = FastAPI(
    title="FastAPI",
    description="A drop-in replacement for HuggingFace API definition",
    version="0.1.0",
)


@dataclass
class GenerationRequestDataclass:
    http_request: Request
    future: asyncio.Future
    parameters: lmi.GenerationParameters
    inputs: str


class ParametersDTO(BaseModel):
    top_k: int = 1
    top_p: float = 0.95
    temperature: float = 1
    repetition_penalty: float = None
    max_new_tokens: int = None
    max_time: float = None
    return_full_text: bool = True
    num_return_sequences: int = 1
    do_sample: bool = True
    stop_token: Union[str, None] = None

    def __hash__(self):
        return hash("".join(self.__dict__))


class OptionsDTO(BaseModel):
    use_cache: bool
    wait_for_model: bool

    def __hash__(self):
        return hash("".join(self.__dict__))



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


async def generate_response(inputs: Union[str, List[str]], parameters: lmi.GenerationParameters):
    return NotImplementedError()

@app.post("/generate")
async def generate(request: RequestDataclass, http_request: Request):
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
    parameters = lmi.GenerationParameters(
            temperature=request.parameters.temperature,
            top_p=request.parameters.top_p,
            top_k=request.parameters.top_k,
            repetition_penalty=request.parameters.repetition_penalty,
            max_tokens=request.parameters.max_new_tokens,
            return_prompt=request.parameters.return_full_text,
            do_sample=request.parameters.do_sample,
            stop_token=request.parameters.stop_token
    )
    future = asyncio.get_running_loop().create_future()
    generation_request = GenerationRequestDataclass(
            future=future, 
            http_request=http_request, 
            inputs=request.inputs, 
            parameters=parameters
    )
    await GENERATE_QUEUE.put(generation_request)
    output = await future
    output = await output
    output = request.inputs + output if request.parameters.return_full_text else output
    response = {"generated_text": output}
    return JSONResponse([response])





async def batch_generating_loop():

    loop = asyncio.get_running_loop()
    while loop.is_running():
        await asyncio.sleep(REQUESTS_BATCH_PROCESS_WINDOW_SECONDS)
        requests = {}
        for _ in range(REQUESTS_BATCH_SIZE):
            try:
                request = GENERATE_QUEUE.get_nowait()
            except asyncio.QueueEmpty:
                pass
            else:
                if not await request.http_request.is_disconnected():
                    requests.setdefault(request.parameters, []).append(request)

        if not requests:
            continue

        parameters_usage_counter = Counter(requests.keys())

        most_used_parameters = max(
                parameters_usage_counter, 
                key=parameters_usage_counter.get
        )
        param_requests = requests.pop(most_used_parameters)

        for rescheduled_request in requests.values():
            await GENERATE_QUEUE.put(rescheduled_request)

        await generate_response(
                inputs=[request.inputs for request in param_requests],
                parameters=most_used_parameters,
                callback=[request.future.set_result for request in param_requests],
        )


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batch_generating_loop())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--lmi", type=str)
    args = parser.parse_args()
    lmi_module = importlib.import_module(args.lmi)
    generate_response = lmi_module.agenerator

    uvicorn.run(app, host=args.host, port=args.port)
