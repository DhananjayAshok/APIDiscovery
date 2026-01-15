from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

from abc import ABC, abstractmethod


class ModelInterface(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
        pass



class OpenAIModel(ModelInterface):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI()

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            max_output_tokens=max_new_tokens,
            temperature=temperature
        )
        return response.output_text


class HuggingFaceModel(ModelInterface):
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            tokenizer=self.tokenizer,
            stop_strings=["[STOP]"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_only_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(output_only_ids[0], skip_special_tokens=True)
        if "[STOP]" in text:
            text = text.split("[STOP]")[0].strip()
        return text
