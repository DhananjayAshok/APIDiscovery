import os
import sys
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from peft import PeftModel
# model_name is arg 1, final_checkpoint_path is arg 2
model_name = sys.argv[1]
final_checkpoint_path = sys.argv[2]
# error out if neither of these are provided
if not model_name or not final_checkpoint_path:
    raise ValueError("Both model_name and final_checkpoint_path must be provided as arguments.")

# error out if final_checkpoint_path does not exist
if not os.path.exists(final_checkpoint_path):
    raise ValueError(f"final_checkpoint_path {final_checkpoint_path} does not exist.")

global_steps = os.listdir(final_checkpoint_path)
# error out if its empty
if not global_steps:
    raise ValueError(f"final_checkpoint_path {final_checkpoint_path} is empty.")

# get the global step with the highest number, all folders are in format global_step_{number}
global_steps = [int(x.split("_")[-1]) for x in global_steps if x.startswith("global_step_")]
if not global_steps:
    raise ValueError(f"No folders in final_checkpoint_path {final_checkpoint_path} start with 'global_step_'.")
global_step = max(global_steps)
adapter_path = os.path.join(final_checkpoint_path, f"global_step_{global_step}/policy/")
# error out if adapter_path does not exist
if not os.path.exists(adapter_path):
    raise ValueError(f"adapter_path {adapter_path} does not exist.")

print(f"Adapter path: {adapter_path}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()
model.save_pretrained(final_checkpoint_path)
tokenizer.save_pretrained(final_checkpoint_path)
print(f"Model and tokenizer saved to {final_checkpoint_path}")