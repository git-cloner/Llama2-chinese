import os
import sys
import time
import json
import datetime

import argparse
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
from utils.prompter import Prompter
from fastapi import FastAPI, Request
import uvicorn

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

prompter = Prompter("")
model = None
tokenizer = None
stream_buffer = {}
app = FastAPI()


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def removeTimeoutBuffer():
    global stream_buffer
    for key in stream_buffer.copy():
        diff = datetime.datetime.now() - stream_buffer[key]["time"]
        seconds = diff.total_seconds()
        print(key + ": 已存在" + str(seconds) + "秒")
        if seconds > 120:
            if stream_buffer[key]["stop"]:
                del stream_buffer[key]
                print(key + "：已被从缓存中移除")
            else:
                stream_buffer[key]["stop"] = True
                print(key + "：已被标识为结束")


def stream_item(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=1,
    max_new_tokens=256,
    **kwargs,
):
    global model
    global tokenizer
    global stream_buffer

    _prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        repetition_penalty=2.0,
        **kwargs,
    )

    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
    )
    t1 = Thread(target=model.generate, kwargs=generate_kwargs)
    t1.start()

    response = ""
    for new_text in streamer:
        response += new_text
        time.sleep(0.05)
        now = datetime.datetime.now()
        print(response)
        stream_buffer[instruction] = {
            "response": response, "stop": False, "history": [], "time": now}
    stream_buffer[instruction]["stop"] = True
    torch_gc()
    return "OK"


def LoadModel(
    load_4bit: bool = False,
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = ""
):
    global model
    global tokenizer
    if load_4bit:
        load_8bit = False
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            load_in_4bit=load_4bit,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)


@app.post("/stream")
async def create_item(request: Request):
    global stream_buffer
    # remove timeout buffer
    removeTimeoutBuffer()
    # get request
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    # new thread
    now = datetime.datetime.now()
    if stream_buffer.get(prompt) is None:
        stream_buffer[prompt] = {"response": "",
                                 "stop": False, "history": [], "time": now}
        sub_thread = Thread(target=stream_item, args=(prompt,))
        sub_thread.start()
    # return soon
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    response = stream_buffer[prompt]["response"]
    history = stream_buffer[prompt]["history"]
    # stop flag
    if stream_buffer[prompt]["stop"]:
        response = response + '[stop]'
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + \
        prompt + '", response:"' + repr(response) + '"'
    print(log)

    return answer

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--base_model', default="./models/NousResearch/Llama-2-7b-hf", type=str)
    parser.add_argument('--lora_weights', default="./output/model", type=str,
                        help="If None, perform inference on the base model")
    parser.add_argument('--load_8bit', action='store_true',
                        help='only use CPU for inference')
    parser.add_argument('--load_4bit', action='store_true',
                        help='only use CPU for inference')
    args = parser.parse_args()
    LoadModel(args.load_4bit, args.load_8bit,
              args.base_model, args.lora_weights)
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
