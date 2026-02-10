"""
1. Load various code gen datasets, reshpape them into a common format:
    - test_func_validated: The function code with anonymized function name and added validate_input_args call
    - description: A brief description of what the function does
    - examples: A list of example inputs to the function
    - more_examples: A more exhaustive list of example inputs to the function
2. Load the jsonl files and then filter the examples based on which ones do not trigger an error when provided into the test_func_validated. For each example, add the outputs too. Push the filtered datasets to Hugging Face Hub. Will have final columns:
    - test_func_validated: The function code with anonymized function name and added validate_input_args call
    - description: A brief description of what the function does
    - train_inputs: A list of example inputs to the function. Can be used to guide the api discovery process. Is small (no more than 2 points).
    - train_outputs: A list of outputs from the function for the corresponding inputs. Is the same length as train_inputs.
    - test_inputs: A list of example inputs to the function that did not error. No overlap with train_inputs.
    - test_outputs: A list of outputs from the function for the corresponding inputs. Is the same length as test_inputs.
"""

from utils.parameter_handling import load_parameters, compute_secondary_parameters
from utils import log_error, log_info, log_warn
from utils.lm_inference import HuggingFaceModel
import click
import random
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
import pandas as pd
import os
import subprocess

loaded_parameters = load_parameters()


TEST_DATASETS = ["cruxeval", "mbpp", "humaneval"]
TRAIN_DATASETS = ["code_alpaca", "magic_coder"]


class Prompts:
    validation_creator = """
You are given a function definition with several arguments. Your task is to first, identify the types and other fundamental constraints of the input variables that are required for the function to run without errors. Then, create a validate_input_args function that checks these constraints and raises appropriate exceptions if any of them are violated.
Function: 
test_func(arg0: List[float], arg1: float) -> bool:
    threshold = arg1
    numbers = arg0


    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False
Validation Function: 
def validate_input_args(arg0: List[float], arg1: float) -> None:
    if not isinstance(arg0, list):
        raise TypeError("arg0 must be a list")
    for item in arg0:
        if not isinstance(item, float):
            raise TypeError("All elements in arg0 must be floats")
    if not isinstance(arg1, float):
        raise TypeError("arg1 must be a float")
    return     
[STOP]
Function: 
def test_func(arg0, arg1):
    \"\"\"
    Find the similar elements from the given two tuple lists.
    \"\"\"
    res = tuple(set(arg0) & set(arg1))
    return (res)
Validation Function:
def validate_input_args(arg0: tuple, arg1: tuple) -> None:
    if not isinstance(arg0, tuple):
        raise TypeError("arg0 must be a tuple")
    if not isinstance(arg1, tuple):
        raise TypeError("arg1 must be a tuple")
    return     
[STOP]
Now, generate the validate_input_args function for the following function only, from the def validate_input_args portion to the return line. Make sure to include type annotations on the function definition. After that, say [STOP]
Function: 

"""
    example_creator = """
You are given a function definition. Your task is to create as many example inputs as you can to the function that satisfy the constraints, and also trigger the different branches of the function logic. Output as many examples as you can, that test different parts of the function.     
Output each example on a new line, in the format:
Reasoning: An extremely brief reasoning for the kind of behavior these examples will trigger
 - (arg0, arg1, ..., argN)
 - (arg0, arg1, ..., argN)
Reasoning: brief reasoning for kind of behavior
 - (arg0, arg1, ..., argN)
 - (arg0, arg1, ..., argN)
 - (arg0, arg1, ..., argN)
 - (arg0, arg1, ..., argN) 
[STOP]
Function:
def validate_input_args(arg0):
    if not isinstance(arg0, int):
        raise TypeError("arg0 must be an integer")

def test_func(arg0):
    \"\"\"
    Identify non-prime numbers.
    \"\"\"
    validate_input_args(arg0)
    result = False
    for i in range(2,int(math.sqrt(arg0)) + 1):
        if arg0 % i == 0:
            result = True
    return result

Reasoning: Since the function tests prime numbers, and takes arguments a single integer, we first test with small prime numbers
 - (2)
 - (3)
 - (17)
 - (19)
Reasoning: We can also test with small non-prime numbers
 - (4)
 - (6)
 - (21)
 [STOP]

Note, the type checks in validate_input_args are bindings. So you must always ensure that the inputs you are generating satisfy those type checks. For example, if validate_input_args checks that an argument is a float, you MUST give a float in your examples, not an int. 
Now do this for the following function only. After that, say [STOP].
Function: 
"""

    describe = """
Given the function, briefly describe what the function does in a concise manner.
Example:
Function:
def test_func(arg0: List[float], arg1: float) -> bool:
    validate_input_args(arg0, arg1)
    threshold = arg1
    numbers = arg0
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False
Description:
Checks if there are any two distinct elements in the input list 'arg0' whose absolute difference is less than the specified 'arg1' threshold. It returns True if such a pair exists, otherwise it returns False.
[STOP]
Note, avoid any mention the usage of the validate_input_args method even if it exists, focus only test_func functionallity
Now, describe the following function only and then say [STOP]. It is extremely important you say [STOP] after the description. Do not just keep talking. 
Function:

"""


def anonymize_header(func_code: str) -> str:
    header_start = func_code.index("def ")
    # find the next "(" after header_start
    paren_index = func_code.index("(", header_start)
    # the function name is between header_start + 4 and paren_index
    anonymized_name = (
        func_code[: header_start + 4] + "test_func" + func_code[paren_index:]
    )
    new_paren_index = anonymized_name.index("(", header_start)
    paren_close_index = anonymized_name.index(")", new_paren_index)
    args_raw = anonymized_name[new_paren_index + 1 : paren_close_index].split(",")
    def_end = anonymized_name.index(":", paren_close_index)
    preamble = anonymized_name[: header_start + 4]
    header = anonymized_name[header_start + 4 : def_end + 1]
    body = anonymized_name[def_end + 1 :]
    indent = None
    for line in body.split("\n"):
        stripped_line = line.lstrip()
        if stripped_line != "":
            indent = line[: len(line) - len(stripped_line)]
            break
    if indent == "  ":  # then convert to 4 spaces
        body = body.replace("  ", "    ")
    for i, arg_raw in enumerate(args_raw):
        if ":" in arg_raw:
            arg_raw = arg_raw.split(":")[0]
        arg_name = arg_raw.strip()
        # assumes the indent is 4 spaces
        header = header.replace(f" {arg_name} ", f" arg{i} ")
        header = header.replace(f"({arg_name})", f"(arg{i})")
        header = header.replace(f" {arg_name}:", f" arg{i}:")
        header = header.replace(f",{arg_name}:", f",arg{i}:")
        header = header.replace(f",{arg_name},", f",arg{i},")
        header = header.replace(f",{arg_name} ", f",arg{i} ")
        header = header.replace(f" {arg_name},", f" arg{i},")
        header = header.replace(f"({arg_name} ", f"(arg{i} ")
        header = header.replace(f"({arg_name},", f"(arg{i},")
        header = header.replace(f"({arg_name}:", f"(arg{i}:")
        header = header.replace(f" {arg_name}):", f" arg{i}):")
        header = header.replace(f",{arg_name}):", f",arg{i}):")
        body = f"    {arg_name} = arg{i}\n" + body
    # adding a validate function at the start of the body
    # should be able to exec anonymized code without the validate_input_args call now, do it and if it fails I need to debug:
    validate_call = "    validate_input_args("
    for i, arg_raw in enumerate(args_raw):
        if i > 0:
            validate_call += ", "
        validate_call += f"arg{i}"
    validate_call += ")\n"
    body = validate_call + body
    anonymized_code = preamble + header + "\n" + body
    return anonymized_code


def move_imports_top(func_code: str) -> str:
    lines = func_code.split("\n")
    import_lines = []
    other_lines = []
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("import ") or stripped_line.startswith("from "):
            import_lines.append(stripped_line)
        else:
            other_lines.append(line)
    new_code = "\n".join(import_lines + other_lines)
    return new_code


class RawLoaders:
    """
    Returns the raw datasets in the form:

    {
        split: CSV
    }

    The CSV has guaranteed columns:
    - "test_func": A body of code that imports required dependencies and defines a function called test_func fully. Can run exec on this code
    - "inputs": List of strings representing arg inputs to test_func. Each string can be eval'd to get the actual input.
    - "outputs": List of strings representing outputs from test_func.

    May have other columns.
    - "raw_text": A rough prompt that was used to generate the function. May not be present in all datasets and may need rewriting.
    -

    The following pattern should work:
    ```python
    test_func_str = df["test_func"][0]
    inputs = eval(df["inputs"][0])
    outputs = eval(df["outputs"][0])
    exec(test_func_str)
    for inp, out in zip(inputs, outputs):
        assert test_func(eval(inp)) == eval(out)
    ```
    """

    @staticmethod
    def load_cruxeval(parameters):
        dataset = load_dataset("cruxeval-org/cruxeval", split="test").to_pandas()
        # rename code column to test_func
        dataset = dataset.rename(columns={"code": "test_func"})
        dataset["test_func_anon"] = dataset["test_func"].apply(anonymize_header)

        # dataset['inputs'] = dataset['inputs'].apply(lambda x: [x])
        # dataset['outputs'] = dataset['outputs'].apply(lambda x: [x])
        def get_docstring_func(row, func_column="test_func_anon"):
            func = row[func_column]
            if func is None:
                return None
            first_indented_line = func.index("    validate_input_args")
            # insert a docstring before this
            doc_text = (
                "Example usage: \n"
                + ">>> test_func("
                + row["input"]
                + ")\n"
                + ">>> "
                + row["output"]
            )
            func = (
                func[:first_indented_line]
                + f'    """\n    {doc_text}\n    """\n'
                + func[first_indented_line:]
            )
            return func

        dataset["test_func_anon_w_docstring"] = dataset.apply(
            lambda x: get_docstring_func(x, "test_func_anon"), axis=1
        )
        dataset["validation_prompt"] = dataset["test_func_anon_w_docstring"].apply(
            lambda x: Prompts.validation_creator + x + "\nValidation Function:\n"
        )
        dataset["description_prompt"] = dataset["test_func_anon_w_docstring"].apply(
            lambda x: Prompts.describe + x + "\nDescription: This function takes in "
        )
        return {"test": dataset}

    @staticmethod
    def load_humaneval(parameters):
        dataset = load_dataset("openai/openai_humaneval", split="test").to_pandas()
        # remove the decode_cyclic question at row index 38
        # rewrite the prompt in index 50 to be more clear:
        new_50 = """
def decode_shift(s: str):
    \"\"\"
    returns encoded string by shifting every character by -5 in the alphabet
    \"\"\"
    
        """
        dataset.at[50, "prompt"] = new_50
        dataset = dataset.drop(index=38).reset_index(drop=True)

        def drop_docstrings(prompt):
            while '"""' in prompt:
                first_index = prompt.index('"""')
                second_index = prompt.index('"""', first_index + 3)
                prompt = prompt[:first_index] + prompt[second_index + 3 :]
            return prompt

        def get_setup(prompt):
            all_funcs = prompt.split("def ")
            if len(all_funcs) <= 1:
                return ""
            setup = "def ".join(all_funcs[:-1])
            return setup

        def last_function(prompt):
            funcs = prompt.split("def ")
            return "def " + funcs[-1]

        dataset["header_only"] = dataset["prompt"].apply(drop_docstrings)
        dataset["function_only"] = (
            dataset["header_only"].apply(last_function) + dataset["canonical_solution"]
        )
        dataset["test_func_anon"] = dataset["prompt"].apply(get_setup) + dataset[
            "function_only"
        ].apply(anonymize_header)
        dataset["validation_prompt"] = dataset["test_func_anon"].apply(
            lambda x: Prompts.validation_creator + x + "\nValidation Function:\n"
        )
        dataset["description_prompt"] = dataset["prompt"].apply(
            lambda x: Prompts.describe
            + x
            + "\nDescription: This function takes in takes in "
        )
        return {"test": dataset}

    @staticmethod
    def load_mbpp(parameters):
        # Muennighoff/mbpp
        dataset = load_dataset(
            "Muennighoff/mbpp", "sanitized", split="test"
        ).to_pandas()
        # functionaly the exact same as cruxeval
        dataset = dataset.rename(columns={"code": "test_func"})
        dataset["test_func"] = dataset["test_func"].str.replace(") : \n", "):\n")
        dataset["test_func"] = dataset["test_func"].str.replace(") :  \n", "):\n")
        dataset["test_func"] = dataset["test_func"].str.replace(") :\n", "):\n")

        def last_function(prompt):
            funcs = prompt.split("def ")
            return "def " + funcs[-1]

        def get_setup(prompt):
            funcs = prompt.split("def ")
            if len(funcs) <= 1:
                return ""
            setup = "def ".join(funcs[:-1]) + "\n"
            return setup

        dataset["test_func_anon"] = dataset["test_func"].apply(get_setup) + dataset[
            "test_func"
        ].apply(last_function).apply(anonymize_header)

        def rewrite_test_list(row):
            func = last_function(row["test_func"])
            # find the first ( after "def ")
            paren_index = func.index("(", 4)
            func_name = func[4:paren_index].strip()
            test_list = row["test_list"]
            new_test_list = []
            for item in test_list:
                new_test_list.append(item.replace(func_name, "test_func"))
            test_list = new_test_list
            return test_list
            # get the

        dataset["test_list"] = dataset.apply(rewrite_test_list, axis=1)

        def add_docstring(row, func_column="test_func_anon"):
            if row[func_column] is None:
                return None
            func = last_function(row[func_column])
            text = row["prompt"].split("function to ")[-1].strip()
            test_list = row["test_list"]
            text = text + "\nWill end up satisfying:\n" + "\n".join(test_list)
            # in cruxeval, function declaration is always first and there are no type hints
            header, body = func.split("):", 1)
            docstring = f'    """\n    {text}\n    """'
            new_func = header + "):" + docstring + body
            return new_func

        dataset["validation_prompt"] = dataset.apply(
            lambda x: add_docstring(x, "test_func_anon"), axis=1
        ).apply(lambda x: Prompts.validation_creator + x + "\nValidation Function:\n")
        dataset["description_output"] = dataset["prompt"].apply(
            lambda x: x.split("function to ")[-1].strip()
        )
        return {"test": dataset}

    @staticmethod
    def load_code_alpaca(parameters):
        df = load_dataset("sahil2801/CodeAlpaca-20k", split="train").to_pandas()
        df = df[df["output"].str.startswith("def ")].reset_index(drop=True)
        df = df[
            df["output"].apply(
                lambda x: "end" not in x and "{" not in x and "}" not in x
            )
        ].reset_index(drop=True)
        # manually removing index 129 which has a weird function
        df = df.drop(index=129).reset_index(drop=True)
        df["test_func_anon"] = df["output"].apply(anonymize_header)

        def insert_docstring(row, func_column="test_func_anon"):
            func = row[func_column]
            if func is None:
                return None
            first_indented_line = func.index("    validate_input_args")
            # insert a docstring before this
            doc_text = row["instruction"]
            func = (
                func[:first_indented_line]
                + f'    """\n    {doc_text}\n    """\n'
                + func[first_indented_line:]
            )
            return func

        df["test_function_anon_w_docstring"] = df.apply(insert_docstring, axis=1)
        df["validation_prompt"] = df["test_function_anon_w_docstring"].apply(
            lambda x: Prompts.validation_creator + x + "\nValidation Function:\n"
        )
        df["description_prompt"] = df["test_function_anon_w_docstring"].apply(
            lambda x: Prompts.describe + x + "\nDescription: This function takes in "
        )
        return {"train": df}

    @staticmethod
    def load_magic_coder(parameters):
        df = load_dataset(
            "ise-uiuc/Magicoder-OSS-Instruct-75K", split="train"
        ).to_pandas()
        df = df[df["lang"] == "python"].reset_index(drop=True)

        def get_func(x):
            if x.count("```python") != 1:
                return None
            pstart = x.index("```python") + len("```python")
            x = x[pstart:]
            if x.count("```") != 1:
                return None
            x = x.split("```")[0]
            if x.count("def ") != 1:
                return None
            if "class " in x:
                return None
            return x

        df["func"] = df["solution"].apply(get_func)
        original_length = len(df)
        df = df[~df["func"].isna()].reset_index(drop=True)
        new_length = len(df)
        log_info(
            f"MagicCoder: Removed {original_length - new_length}/{original_length} entries without a valid single python function."
        )
        df["test_func_anon"] = df["func"].apply(anonymize_header)

        def insert_docstring(row, func_column="test_func_anon"):
            func = row[func_column]
            if func is None:
                return None
            first_indented_line = func.index("    validate_input_args")
            # insert a docstring before this
            doc_text = row["problem"]
            func = (
                func[:first_indented_line]
                + f'    """\n    {doc_text}\n    """\n'
                + func[first_indented_line:]
            )
            return func

        df["test_function_anon_w_docstring"] = df.apply(insert_docstring, axis=1)
        df["validation_prompt"] = df["test_function_anon_w_docstring"].apply(
            lambda x: Prompts.validation_creator + x + "\nValidation Function:\n"
        )

        df["description_prompt"] = df["test_function_anon_w_docstring"].apply(
            lambda x: Prompts.describe + x + "\nDescription: This function takes in "
        )
        df = df.sample(frac=0.75, random_state=42).reset_index(drop=True)
        return {"train": df}

    @staticmethod
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
    ):
        if model is None:
            model = parameters["benchmark_creation_model"]
        open_ai_batch_name = ""
        if "gpt" in model:
            open_ai_batch_name = f"{model}-{run_name}-{dataset_name}-{split}"
        openaibatch_str = "-n " + open_ai_batch_name if open_ai_batch_name != "" else ""
        command_string = f"bash scripts/infer.sh -i {input_file} -o {output_file} -m {model} -c {input_column} -d {output_column} -t {max_new_tokens} -g {run_name} {openaibatch_str}"
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

    @staticmethod
    def generate_validation_code(dataset_name, split, parameters):
        input_file = parameters["data_dir"] + f"/raw/{dataset_name}/{split}_proc.jsonl"
        output_file = (
            parameters["data_dir"]
            + f"/raw/{dataset_name}/{split}_proc_validation_output.jsonl"
        )
        input_column = "validation_prompt"
        output_column = "validation_output"
        max_new_tokens = 300
        return RawLoaders.call_infer(
            run_name="validation-code",
            dataset_name=dataset_name,
            split=split,
            input_file=input_file,
            output_file=output_file,
            input_column=input_column,
            output_column=output_column,
            max_new_tokens=max_new_tokens,
            parameters=parameters,
        )

    @staticmethod
    def generate_description(dataset_name, split, parameters):
        input_file = (
            parameters["data_dir"]
            + f"/raw/{dataset_name}/{split}_proc_validation_output.jsonl"
        )
        output_file = (
            parameters["data_dir"]
            + f"/raw/{dataset_name}/{split}_proc_description_output.jsonl"
        )
        input_column = "description_prompt"
        output_column = "description_output"
        max_new_tokens = 600
        return RawLoaders.call_infer(
            run_name="description",
            dataset_name=dataset_name,
            split=split,
            input_file=input_file,
            output_file=output_file,
            input_column=input_column,
            output_column=output_column,
            max_new_tokens=max_new_tokens,
            parameters=parameters,
        )


class MidLoader:
    @staticmethod
    def parse_validation(df):
        from eval import RunTestFunc

        df["test_func_validated"] = None
        df["validation_output"] = df["validation_output"].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )
        for index, row in tqdm(
            df.iterrows(), total=len(df), desc="Parsing validation code"
        ):
            validation_output = row["validation_output"]
            if "def validate_input_args(" in validation_output:
                validation_code = (
                    "def validate_input_args("
                    + validation_output.split("def validate_input_args(")[1]
                )
                if "return" in validation_code:
                    validation_code = validation_code.split("return")[0] + "return"
                else:
                    validation_code = None
            else:
                validation_code = None
            if "import argparse" in validation_output:
                validation_code = None
            if "__name__ == " in validation_output:
                validation_code = None
            if validation_code is None:
                # log_warn(
                #    f"Could not generate validation code for index {index}\n"
                #    + validation_output,
                #    parameters=loaded_parameters,
                # )
                continue
            df.at[index, "validation_code"] = validation_code
            test_func_str = move_imports_top(
                validation_code + "\n" + row["test_func_anon"]
            )  # Should be able to exec this now
            try:
                RunTestFunc(func_code=test_func_str)
                df.at[index, "test_func_validated"] = test_func_str
            except Exception as e:
                # log_warn(
                #    f"Could not exec validated function for index {index}: {str(e)}",
                #    parameters=loaded_parameters,
                # )
                pass
        # drop test_func_validated Nans
        original_length = len(df)
        df = df[~df["test_func_validated"].isna()].reset_index(drop=True)
        new_length = len(df)
        log_info(
            f"Removed {original_length - new_length}/{original_length} invalid validation_code entries. Now with {new_length} entries..."
        )
        return df

    @staticmethod
    def parse_description(df):
        df["description"] = None
        print(df.columns)
        df["description_output"] = df["description_output"].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )
        for index, row in tqdm(
            df.iterrows(), total=len(df), desc="Generating descriptions"
        ):
            description = row["description_output"]
            if description.strip() == "":
                # log_warn(
                #    f"Could not generate description for index {index}",
                #    parameters=loaded_parameters,
                # )
                description = None
            if "\n\n" in description:
                description = description.split("\n\n")[0]
            if "\n" in description:
                description = description.split("\n")[0]
            df.at[index, "description"] = description
        # drop test_func_validated Nans
        original_length = len(df)
        df = df[~df["description"].isna()].reset_index(drop=True)
        new_length = len(df)
        log_info(
            f"Removed {original_length - new_length}/{original_length} invalid description entries. Now with {new_length} entries..."
        )
        return df

    @staticmethod
    def parse_both(df):
        return MidLoader.parse_description(MidLoader.parse_validation(df))

    @staticmethod
    def get_split_dfs(parameters, dataset_name):
        data_dir = parameters["data_dir"] + f"/raw/{dataset_name}/"
        splits = {}
        files = os.listdir(data_dir)
        for file in files:
            if file.endswith("_proc_description_output.jsonl"):
                split_name = file.replace("_proc_description_output.jsonl", "").replace(
                    "_", ""
                )
                df = pd.read_json(data_dir + file, lines=True)
                log_info(f"Parsing File | {dataset_name}: {file}")
                df = MidLoader.parse_both(df)
                splits[split_name] = df
        return splits

    @staticmethod
    def load_cruxeval(parameters):
        dataset = MidLoader.get_split_dfs(parameters, "cruxeval")["test"]

        def get_docstring_func(row, func_column="test_func_anon"):
            func = row[func_column]
            if func is None:
                return None
            first_indented_line = func.index("    validate_input_args")
            # insert a docstring before this
            doc_text = (
                "Example usage: \n"
                + ">>> test_func("
                + row["input"]
                + ")\n"
                + ">>> "
                + row["output"]
            )
            func = (
                func[:first_indented_line]
                + f'    """\n    {doc_text}\n    """\n'
                + func[first_indented_line:]
            )
            return func

        dataset["test_func_validated_w_docstring"] = dataset.apply(
            lambda x: get_docstring_func(x, "test_func_validated"), axis=1
        )
        dataset["example_prompt"] = dataset["test_func_validated_w_docstring"].apply(
            lambda x: Prompts.example_creator + x if x is not None else None
        )
        return {"test": dataset}

    @staticmethod
    def load_humaneval(parameters):
        dataset = MidLoader.get_split_dfs(parameters, "humaneval")["test"]
        dataset["example_prompt"] = dataset["test_func_validated"].apply(
            lambda x: Prompts.example_creator + x if x is not None else None
        )
        return {"test": dataset}

    @staticmethod
    def load_mbpp(parameters):
        # Muennighoff/mbpp
        dataset = MidLoader.get_split_dfs(parameters, "mbpp")["test"]

        def last_function(prompt):
            funcs = prompt.split("def ")
            return "def " + funcs[-1]

        def add_docstring(row, func_column="test_func_anon"):
            if row[func_column] is None:
                return None
            func = last_function(row[func_column])
            text = row["prompt"].split("function to ")[-1].strip()
            test_list = row["test_list"]
            text = text + "\nWill end up satisfying:\n" + "\n".join(test_list)
            # in cruxeval, function declaration is always first and there are no type hints
            header, body = func.split("):", 1)
            docstring = f'    """\n    {text}\n    """'
            new_func = header + "):" + docstring + body
            return new_func

        dataset["example_prompt"] = dataset.apply(
            lambda x: add_docstring(x, "test_func_validated"), axis=1
        ).apply(lambda x: Prompts.example_creator + x if x is not None else None)
        return {"test": dataset}

    @staticmethod
    def load_code_alpaca(parameters):
        df = MidLoader.get_split_dfs(parameters, "code_alpaca")["train"]

        def insert_docstring(row, func_column="test_func_anon"):
            func = row[func_column]
            if func is None:
                return None
            first_indented_line = func.index("    validate_input_args")
            # insert a docstring before this
            doc_text = row["instruction"]
            func = (
                func[:first_indented_line]
                + f'    """\n    {doc_text}\n    """\n'
                + func[first_indented_line:]
            )
            return func

        df["test_func_validated_w_docstring"] = df.apply(
            lambda x: insert_docstring(x, "test_func_validated"), axis=1
        )
        df["example_prompt"] = df["test_func_validated_w_docstring"].apply(
            lambda x: Prompts.example_creator + x if x is not None else None
        )
        return {"train": df}

    @staticmethod
    def load_magic_coder(parameters):
        df = MidLoader.get_split_dfs(parameters, "magic_coder")["train"]

        def insert_docstring(row, func_column="test_func_anon"):
            func = row[func_column]
            if func is None:
                return None
            first_indented_line = func.index("    validate_input_args")
            # insert a docstring before this
            doc_text = row["problem"]
            func = (
                func[:first_indented_line]
                + f'    """\n    {doc_text}\n    """\n'
                + func[first_indented_line:]
            )
            return func

        df["test_func_validated_w_docstring"] = df.apply(
            lambda x: insert_docstring(x, "test_func_validated"), axis=1
        )
        df["example_prompt"] = df["test_func_validated_w_docstring"].apply(
            lambda x: Prompts.example_creator + x if x is not None else None
        )
        return {"train": df}

    @staticmethod
    def generate_examples(dataset_name, split, parameters):
        input_file = (
            parameters["data_dir"] + f"/raw/{dataset_name}/{split}_proc_mid.jsonl"
        )
        output_file = (
            parameters["data_dir"]
            + f"/raw/{dataset_name}/{split}_proc_example_output.jsonl"
        )
        input_column = "example_prompt"
        output_column = "example_output"
        max_new_tokens = 1200
        return RawLoaders.call_infer(
            run_name="examples",
            dataset_name=dataset_name,
            split=split,
            input_file=input_file,
            output_file=output_file,
            input_column=input_column,
            output_column=output_column,
            max_new_tokens=max_new_tokens,
            parameters=parameters,
        )


class FilteredLoader:
    @staticmethod
    def parse_examples(df):
        df["examples"] = None
        df["n_examples"] = None
        df["example_output"] = df["example_output"].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )
        for index, row in tqdm(
            df.iterrows(), total=len(df), desc="Generating example inputs"
        ):
            example_code = row["example_output"]
            df.at[index, "example_output"] = example_code
            examples = []
            for line in example_code.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    line = line[2:].strip()
                if line.startswith("(") and line.endswith(")"):
                    examples.append(line)
            examples = list(set(examples))  # unique only
            df.at[index, "examples"] = examples
        # drop if there are less than 5 examples overall
        original_length = len(df)
        df = df[df["examples"].apply(lambda x: len(x) >= 5)].reset_index(drop=True)
        new_length = len(df)
        log_info(
            f"Removed {original_length - new_length}/{original_length} insufficient example entries. Now with {new_length} entries..."
        )
        return df

    @staticmethod
    def train_test_split(inputs, outputs):
        train_inputs = []
        train_outputs = []
        test_inputs = []
        test_outputs = []
        for i in range(len(inputs)):
            if len(train_inputs) >= 2:
                continue
            if i == 0:
                train_inputs.append(inputs[i])
                train_outputs.append(outputs[i])
            else:
                candidate_input = inputs[i]
                candidate_output = outputs[i]
                # pass it over if the output is already in train outputs
                if candidate_output in train_outputs:
                    test_inputs.append(candidate_input)
                    test_outputs.append(candidate_output)
                    continue
                # pass it over if the existing train input has an identity mapping and this is also an identity mapping.
                existing_identity = False
                candidate_identity = False
                try:  # might not be valid
                    for train_input, train_output in zip(train_inputs, train_outputs):
                        if train_input == train_output:
                            existing_identity = True
                            break

                    if candidate_input == candidate_output:
                        candidate_identity = True
                except Exception as e:
                    pass
                if existing_identity and candidate_identity:
                    test_inputs.append(candidate_input)
                    test_outputs.append(candidate_output)
                else:
                    train_inputs.append(candidate_input)
                    train_outputs.append(candidate_output)
        return train_inputs, train_outputs, test_inputs, test_outputs

    @staticmethod
    def filter_examples(test_func_code: str, examples: list):
        from eval import RunTestFunc

        try:
            runner = RunTestFunc(test_func_code)
        except Exception as e:
            return False, [], [], [], []
        working_inputs = []
        working_outputs = []
        for example in examples:
            try:
                return_value, error = runner.run_test_str(example)
                if error is None:
                    working_inputs.append(example)
                    working_outputs.append(return_value)
                else:
                    pass
            except Exception as e:
                pass
        if len(working_inputs) < 5:
            return True, [], [], [], []  # Not enough to make a decent train set
        train_inputs, train_outputs, test_inputs, test_outputs = (
            FilteredLoader.train_test_split(working_inputs, working_outputs)
        )
        if len(train_inputs) < 2:
            return True, [], [], [], []  # Not enough diverse train examples
        return True, train_inputs, train_outputs, test_inputs, test_outputs

    @staticmethod
    def filter_annotate_dataset(df):
        random.seed(42)
        # dataset has columns: test_func_validated, examples, more_examples, description
        df["train_inputs"] = None
        # df["train_outputs"] = None
        df["test_inputs"] = None
        # df["test_outputs"] = None
        df["execable"] = None
        for index, row in tqdm(
            df.iterrows(), total=len(df), desc="Filtering and annotating dataset"
        ):
            test_func_code = row["test_func_validated"]
            all_examples = list(set(row["examples"]))
            # shuffle all examples
            random.shuffle(all_examples)
            execable, train_inputs, train_outputs, test_inputs, test_outputs = (
                FilteredLoader.filter_examples(test_func_code, all_examples)
            )
            # first 2 filtered inputs/outputs are train, rest are test
            df.at[index, "train_inputs"] = train_inputs
            # df.at[index, "train_outputs"] = train_outputs
            df.at[index, "test_inputs"] = test_inputs
            # df.at[index, "test_outputs"] = test_outputs
            df.at[index, "execable"] = execable
        # take out not execable rows
        original_length = len(df)
        df = df[df["execable"] == True].reset_index(drop=True)
        # take out the rows with empty train_inputs or test_inputs
        df = df[(df["train_inputs"].apply(len) >= 2)].reset_index(drop=True)
        cleaned_length = len(df)
        log_info(
            f"Filtered dataset: removed {original_length - cleaned_length}/{original_length} rows with non-execable test functions or insufficient examples",
            parameters=loaded_parameters,
        )
        return df

    def make_train_dataset(df):
        def get_header(func_code):
            header_start = func_code.index("def test_func(")
            header_end = func_code.index("\n", header_start)
            func_header = func_code[header_start:header_end]
            return func_header

        def make_prompt(row):
            func_header = get_header(row["test_func_validated"])
            from eval import RunTestFunc

            runner = RunTestFunc(row["test_func_validated"])
            prompt = "You are given the following function signature:\n"
            prompt += func_header + "\n"
            prompt += "Based on the following examples of inputs and outputs, provide a description of what this function does.\n"
            for train_input in row["train_inputs"]:
                prompt += f"Input: {train_input}\n"
                prompt += f"Output: {runner.run_test_str(train_input)[0]}\n"
            prompt += "Description: "
            return prompt

        df["direct_prompt"] = df.apply(make_prompt, axis=1)
        return df


@click.command()
@click.option(
    "--dataset_name",
    required=True,
    help="The name of the dataset to load from the Hugging Face Hub.",
    type=click.Choice(TEST_DATASETS + TRAIN_DATASETS),
)
@click.option(
    "--execute_inference",
    default=True,
    help="Whether to execute inference during processing.",
    type=bool,
)
@click.option(
    "--mid_step",
    default=True,
    help="Whether to execute the mid step processing after raw processing.",
    type=bool,
)
@click.pass_obj
def process_raw(parameters, dataset_name, execute_inference, mid_step):
    split_no = 20
    save_dir = parameters["data_dir"] + f"/raw/{dataset_name}/"
    os.makedirs(save_dir, exist_ok=True)
    log_info(f"Processing raw dataset {dataset_name}", parameters=parameters)
    if dataset_name == "humaneval":
        data_splits = RawLoaders.load_humaneval(parameters)
    elif dataset_name == "cruxeval":
        data_splits = RawLoaders.load_cruxeval(parameters)
    elif dataset_name == "mbpp":
        data_splits = RawLoaders.load_mbpp(parameters)
    elif dataset_name == "code_alpaca":
        data_splits = RawLoaders.load_code_alpaca(parameters)
    elif dataset_name == "magic_coder":
        data_splits = RawLoaders.load_magic_coder(parameters)
    else:
        log_error(f"Dataset {dataset_name} not recognized.", parameters=parameters)
    for data_split in data_splits:
        csv = data_splits[data_split]
        csv.to_json(f"{save_dir}/{data_split}_proc.jsonl", orient="records", lines=True)
        if execute_inference:
            if dataset_name in [
                "humaneval"
            ]:  # then they are small enough to do in one batch
                df = RawLoaders.generate_validation_code(
                    dataset_name, data_split, parameters
                )
            else:
                # then we want to split them into 5, run each one separately, and concat
                csv_length = len(csv)
                dfs_compiled = []
                df = None
                done = False
                if os.path.exists(
                    parameters["data_dir"]
                    + f"/raw/{dataset_name}/{data_split}_proc_validation_output.jsonl"
                ):
                    df = pd.read_json(
                        parameters["data_dir"]
                        + f"/raw/{dataset_name}/{data_split}_proc_validation_output.jsonl",
                        lines=True,
                    )
                    done = True
                if not done:
                    for i in range(split_no):
                        save_dir = parameters["data_dir"] + f"/raw/{dataset_name}_{i}/"
                        os.makedirs(save_dir, exist_ok=True)
                        start_index = int(i * csv_length / split_no)
                        end_index = int((i + 1) * csv_length / split_no)
                        csv_split = csv.iloc[start_index:end_index]
                        csv_split.to_json(
                            f"{save_dir}/{data_split}_proc.jsonl",
                            orient="records",
                            lines=True,
                        )
                        df_split = RawLoaders.generate_validation_code(
                            dataset_name + f"_{i}", data_split, parameters
                        )
                        if df_split is None:
                            log_info(
                                f"Split {i} must still be running inference. Skipping the rest."
                            )
                            df = None
                            break
                        dfs_compiled.append(df_split)
                    if len(dfs_compiled) == split_no:
                        df = pd.concat(dfs_compiled).reset_index(drop=True)
                        save_dir = parameters["data_dir"] + f"/raw/{dataset_name}/"
                        df.to_json(
                            f"{save_dir}/{data_split}_proc_validation_output.jsonl",
                            lines=True,
                            orient="records",
                        )
            if dataset_name == "mbpp":
                # save this file to the description_output directly
                df.to_json(
                    f"{save_dir}/{data_split}_proc_description_output.jsonl",
                    orient="records",
                    lines=True,
                )
                log_info(
                    f"Completed raw processing for dataset {dataset_name} split {data_split}",
                    parameters=parameters,
                )
                continue
            else:
                if df is not None:
                    if dataset_name in ["humaneval"]:
                        df = RawLoaders.generate_description(
                            dataset_name, data_split, parameters
                        )
                    else:
                        csv_length = len(df)
                        dfs_compiled = []
                        df_out = None
                        done = False
                        if os.path.exists(
                            parameters["data_dir"]
                            + f"/raw/{dataset_name}/{data_split}_proc_description_output.jsonl"
                        ):
                            df_out = pd.read_json(
                                parameters["data_dir"]
                                + f"/raw/{dataset_name}/{data_split}_proc_description_output.jsonl",
                                lines=True,
                            )
                            done = True
                        if not done:
                            for i in range(split_no):
                                df_split = RawLoaders.generate_description(
                                    dataset_name + f"_{i}", data_split, parameters
                                )
                                if df_split is None:
                                    log_info(
                                        f"Split {i} must still be running inference. Skipping the rest."
                                    )
                                    df_out = None
                                    break
                                dfs_compiled.append(df_split)
                            if len(dfs_compiled) == split_no:
                                df_out = pd.concat(dfs_compiled).reset_index(drop=True)
                                save_dir = (
                                    parameters["data_dir"] + f"/raw/{dataset_name}/"
                                )
                                df_out.to_json(
                                    f"{save_dir}/{data_split}_proc_description_output.jsonl",
                                    lines=True,
                                    orient="records",
                                )
                                df = df_out
                            else:
                                df = None
                if df is not None:
                    log_info(
                        f"Completed raw processing for dataset {dataset_name} split {data_split}",
                        parameters=parameters,
                    )
                else:
                    log_info(
                        f"Skipped inference for dataset {dataset_name} split {data_split}",
                        parameters=parameters,
                    )

    if execute_inference and mid_step:
        if df is not None:
            log_info(
                f"Executing mid step processing for dataset {dataset_name}",
                parameters=parameters,
            )
            if dataset_name == "humaneval":
                data_splits = MidLoader.load_humaneval(parameters)
            elif dataset_name == "cruxeval":
                data_splits = MidLoader.load_cruxeval(parameters)
            elif dataset_name == "mbpp":
                data_splits = MidLoader.load_mbpp(parameters)
            elif dataset_name == "code_alpaca":
                data_splits = MidLoader.load_code_alpaca(parameters)
            elif dataset_name == "magic_coder":
                data_splits = MidLoader.load_magic_coder(parameters)
            else:
                log_error(
                    f"Dataset {dataset_name} not recognized.", parameters=parameters
                )
            for data_split in data_splits:
                csv = data_splits[data_split]
                csv.to_json(
                    f"{save_dir}/{data_split}_proc_mid.jsonl",
                    orient="records",
                    lines=True,
                )
                if dataset_name in ["humaneval"]:
                    df = MidLoader.generate_examples(
                        dataset_name, data_split, parameters
                    )
                else:
                    csv_length = len(csv)
                    dfs_compiled = []
                    df = None
                    for i in range(split_no):
                        save_dir = parameters["data_dir"] + f"/raw/{dataset_name}_{i}/"
                        start_index = int(i * csv_length / split_no)
                        end_index = int((i + 1) * csv_length / split_no)
                        csv_split = csv.iloc[start_index:end_index]
                        csv_split.to_json(
                            f"{save_dir}/{data_split}_proc_mid.jsonl",
                            orient="records",
                            lines=True,
                        )
                        df_split = MidLoader.generate_examples(
                            dataset_name + f"_{i}", data_split, parameters
                        )
                        if df_split is None:
                            log_info(
                                f"Split {i} must still be running inference. Skipping the rest."
                            )
                            df = None
                            break
                        dfs_compiled.append(df_split)
                    if len(dfs_compiled) == split_no:
                        df = pd.concat(dfs_compiled).reset_index(drop=True)
                        save_dir = parameters["data_dir"] + f"/raw/{dataset_name}/"
                        df.to_json(
                            f"{save_dir}/{data_split}_proc_example_output.jsonl",
                            lines=True,
                            orient="records",
                        )
                if df is not None:
                    log_info(
                        f"Completed mid step processing for dataset {dataset_name} split {data_split}",
                        parameters=parameters,
                    )
        else:
            log_info(
                f"Skipped mid step processing for dataset {dataset_name}",
                parameters=parameters,
            )


@click.command()
@click.option(
    "--dataset_name",
    required=True,
    help="The name of the dataset to load from the Hugging Face Hub.",
    type=click.Choice(TEST_DATASETS + TRAIN_DATASETS),
)
@click.pass_obj
def process_final(parameters, dataset_name):
    load_dir = parameters["data_dir"] + f"/raw/{dataset_name}/"
    save_dir = parameters["data_dir"] + f"/final/{dataset_name}/"
    huggingface_hub_username = parameters["huggingface_repo_namespace"]
    huggingface_hub_repo_name = "APIDiscoveryDataset"
    os.makedirs(save_dir, exist_ok=True)
    files = os.listdir(load_dir)
    split_dict = {}
    for file in files:
        if file.endswith("proc_example_output.jsonl"):
            split_name = file.replace("proc_example_output.jsonl", "")
            split_dict[split_name] = FilteredLoader.parse_examples(
                pd.read_json(f"{load_dir}/{file}", lines=True)
            )
    if split_dict == {}:
        log_error(
            f"No cleaned jsonl files found in {load_dir} for dataset {dataset_name}",
            parameters=parameters,
        )
    for split_name in split_dict:
        df = split_dict[split_name]
        df_filtered = FilteredLoader.filter_annotate_dataset(df)
        df_filtered = FilteredLoader.make_train_dataset(df_filtered)
        df_filtered = df_filtered[
            [
                "test_func_validated",
                "description",
                "train_inputs",
                "test_inputs",
                "direct_prompt",
            ]
        ]
        split_name = split_name.replace("_", "")
        df_filtered.to_json(
            f"{save_dir}/{split_name}_filtered.jsonl", orient="records", lines=True
        )
        df_filtered.to_csv(f"{save_dir}/{split_name}_filtered.csv", index=False)
        log_info(
            f"Saved final filtered dataset with {len(df_filtered)} rows for {dataset_name} split {split_name} to {save_dir}/{split_name}_filtered.jsonl",
            parameters=parameters,
        )
        dataset = Dataset.from_pandas(df_filtered)
        dataset.push_to_hub(
            f"{huggingface_hub_username}/{huggingface_hub_repo_name}",
            private=False,
            split=split_name,
            config_name=dataset_name,
        )


@click.command()
@click.pass_obj
def merge_final(parameters):
    for split, options in [("test", TEST_DATASETS), ("train", TRAIN_DATASETS)]:
        all_datasets = []
        for dataset_name in options:
            dataset = load_dataset(
                parameters["huggingface_repo_namespace"] + "/APIDiscoveryDataset",
                dataset_name=dataset_name,
                split=split,
            )
            all_datasets.append(dataset)
        merged_dataset = concatenate_datasets(all_datasets)
        merged_dataset.push_to_hub(
            parameters["huggingface_repo_namespace"] + "/APIDiscoveryDataset",
            private=False,
            split=split,
            config_name="all",
        )


@click.command()
@click.option(
    "--dataset_name",
    required=True,
    help="The name of the dataset to load from the Hugging Face Hub.",
    type=click.Choice(TEST_DATASETS + TRAIN_DATASETS + ["all"]),
)
@click.option(
    "--train_val_split",
    default=0.8,
    help="The proportion of the dataset to use for training when creating parquet files. The rest will be used for validation.",
    type=float,
)
@click.pass_obj
def load_parquets(parameters, dataset_name, train_val_split):
    from baselines import first_reasoning_prompt, get_prev_results_str, get_initial_results
    def get_first_prompt(row):
        test_func_validated = row["test_func_validated"]
        header_start = test_func_validated.index("def test_func(")
        header_end = test_func_validated.index("\n", header_start)
        func_header = test_func_validated[header_start:header_end]
        reasoning_prompt = first_reasoning_prompt.replace("[HEADER]", func_header).replace("[HYPOTHESIS]", "Not yet formed")
        prev_results, runner = get_initial_results(test_func_validated, examples=row["train_inputs"])
        if runner is None:
            return None
        prev_results_str = get_prev_results_str(prev_results, max_previous_results=None)
        reasoning_prompt = reasoning_prompt.replace("[PREV]", prev_results_str)
        return reasoning_prompt

    splits = []
    if dataset_name in TEST_DATASETS + ["all"]:
        splits.append("test")
    if dataset_name in TRAIN_DATASETS + ["all"]:
        splits.append("train")
    for split in splits:
        parquet_path = parameters["data_dir"] + f"/parquets/{dataset_name}/"
        dataset = load_dataset(
            parameters["huggingface_repo_namespace"] + "/APIDiscoveryDataset",
            dataset_name,
            split=split,
        )
        dataset = dataset.map(
            lambda x: {"prompt": [{"role": "user", "content": get_first_prompt(x)}]}
        )
        dataset_length = len(dataset)
        #drop rows where prompt is None
        dataset = dataset.filter(lambda x: x["prompt"] is not None)
        if len(dataset) < dataset_length:
            log_warn(f"Dropped {dataset_length - len(dataset)}/{dataset_length} rows with invalid prompts for {dataset_name} split {split}", parameters=parameters)
        if split == "test":
            dataset.to_parquet(parquet_path + f"test.parquet")
        else:
            dataset.shuffle(seed=parameters["random_seed"])
            train_size = int(len(dataset) * train_val_split)
            train_dataset = dataset.select(range(train_size))
            val_dataset = dataset.select(range(train_size, len(dataset)))
            train_dataset.to_parquet(f"{parquet_path}/train.parquet")
            val_dataset.to_parquet(f"{parquet_path}/val.parquet")
        log_info(
            f"Saved {split} split of dataset {dataset_name} to parquet at {parquet_path}",
            parameters=parameters,
        )


@click.group()
@click.pass_context
def main(ctx):
    compute_secondary_parameters(loaded_parameters)
    ctx.obj = loaded_parameters


main.add_command(process_raw, name="process_raw")
main.add_command(process_final, name="process_final")
main.add_command(merge_final, name="merge_final")
main.add_command(load_parquets, name="load_parquets")

if __name__ == "__main__":
    main()
