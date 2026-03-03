from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from time import perf_counter, sleep
from utils import log_info, log_warn, log_error
import pandas as pd
import subprocess
from abc import ABC, abstractmethod


class ModelInterface(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0
    ) -> str:
        pass


class OpenAIModel(ModelInterface):
    seconds_per_query = (15) + 0.01

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI()
        self.previous_call = perf_counter() - self.seconds_per_query

    def generate(
        self, prompt: str, max_new_tokens: int = 50, temperature: float = None
    ) -> str:
        time_to_wait = self.seconds_per_query - (perf_counter() - self.previous_call)
        if time_to_wait > 0:
            sleep(time_to_wait)
        self.previous_call = perf_counter()
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            max_output_tokens=max_new_tokens,
            temperature=temperature,
        )
        text = response.output_text
        if "[STOP]" in text:
            text = text.split("[STOP]")[0]
        return text.strip()


class HuggingFaceModel(ModelInterface):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = None,
        repetition_penalty: float = 1.1,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            tokenizer=self.tokenizer,
            stop_strings=["[STOP]"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature is not None and temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )
        output_only_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(output_only_ids[0], skip_special_tokens=True)
        text = text.replace("[STOP]", "").strip()
        return text


def call_infer(
    run_name,
    dataset_name,
    split,
    input_file,
    output_file,
    input_column,
    output_column,
    max_new_tokens,
    parameters,
    model=None,
    ignore_checkpoint=False,
):
    if model is None:
        log_error(
            "Model must be specified for inference command", parameters=parameters
        )
    open_ai_batch_name = ""
    if "gpt" in model:
        open_ai_batch_name = f"{model}-{run_name}-{dataset_name}-{split}"
    openaibatch_str = "-n " + open_ai_batch_name if open_ai_batch_name != "" else ""
    command_string = f"bash scripts/infer.sh -i {input_file} -o {output_file} -m {model} -c {input_column} -d {output_column} -t {max_new_tokens} -g {run_name} {openaibatch_str}"
    if ignore_checkpoint:
        command_string += " -r yes"
    log_info(f"Generating validation code with command: {command_string}")
    subprocess.run(command_string, shell=True, check=True)
    try:
        df = pd.read_json(output_file, lines=True)
        if "output_logits" in df.columns:
            df.drop("output_logits", axis=1, inplace=True)
        df.to_json(output_file, orient="records", lines=True)
    except:
        log_warn(
            f"Output file {output_file} not found after inference command. This can happen for OpenAI API models. Run the command again after the batch is complete.",
            parameters=parameters,
        )
        return None
    return df
