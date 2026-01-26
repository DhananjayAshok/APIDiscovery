from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from time import perf_counter, sleep

from abc import ABC, abstractmethod


class ModelInterface(ABC):
    @abstractmethod
    def generate(
        self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0
    ) -> str:
        pass


class OpenAIModel(ModelInterface):
    seconds_per_query = (60 / 20) + 0.01

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
