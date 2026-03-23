from load_data import get_dataset
import click
from utils import log_info, log_error, log_warn, load_parameters
import os
from tqdm import tqdm


def get_test_func_only(func_code):
    func_code_lines = func_code.split("\n")
    func_def_line_index = None
    for i, line in enumerate(func_code_lines):
        if line.strip().startswith("def test_func"):
            func_def_line_index = i
            break
    if func_def_line_index is not None:
        return "\n".join(func_code_lines[func_def_line_index:])
    else:
        raise ValueError("No function definition found in the provided code.")

def get_imports(func_code):
    func_code_lines = func_code.split("\n")
    imports = []
    for line in func_code_lines:
        if line.strip().startswith("import") or line.strip().startswith("from"):
            imports.append(line)
    return "\n".join(imports)

def get_test_line(test_examples, func_code):
    test_input_list_line = f"test_examples = {test_examples}"
    singleton = "arg1" not in func_code
    if singleton:
        test_line = f"for example in test_examples:\n    input_args, expected_output = example\n    result = test_func(eval(input_args))"
    else:
        test_line = f"for example in test_examples:\n    input_args, expected_output = example\n    result = test_func(*eval(input_args))"
    return test_input_list_line + "\n" + test_line


@click.command()
def write_coverage_files():
    dataset = get_dataset("test", load_examples=True)
    dummy_validation_func = "def validate_input_args(*args, **kwargs) -> None:\n    pass"
    save_dir = load_parameters()["tmp_dir"] + "/coverage_files/"
    os.makedirs(save_dir, exist_ok=True)
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
        func_code_to_save = get_imports(row["test_func_validated"]) + "\n\n" + dummy_validation_func + "\n\n" + get_test_func_only(row["test_func_validated"])
        test_examples = row["test_examples"]
        func_code_to_save += "\n\n" + get_test_line(test_examples, row["test_func_validated"])
        with open(save_dir + f"/test_func_{idx}.py", "w") as f:
            f.write(func_code_to_save)
    log_info(f"Saved coverage files to {save_dir}")

if __name__ == "__main__":
    write_coverage_files()