from datasets import load_dataset
from utils import model_factory, load_parameters, log_info, log_error, RunTestFunc
from tqdm import tqdm
import click
import os
import pandas as pd

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

def drop_docstrings(func_code):
    """Will mess up if there are triple quotes in the code that are not docstrings, but this is a simple heuristic to drop docstrings from the function code."""
    while '"""' in func_code:
        first_index = func_code.index('"""')
        if '"""' not in func_code[first_index + 3:]:
            break
        second_index = func_code.index('"""', first_index + 3)
        func_code = func_code[:first_index] + func_code[second_index + 3 :]
    return func_code

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

def potentially_unsafe(func_code):
    unsafe_modules = ["os", "sys", "subprocess", "shutil", "socket", "requests"]
    for module in unsafe_modules:
        if f"import {module}" in func_code or f"from {module} " in func_code:
            return True
    return False


def anonymize_header(func_code: str) -> str:
    func_code = move_imports_top(func_code)
    func_code = drop_docstrings(func_code)
    if potentially_unsafe(func_code):
        return None
    if "def " not in func_code:
        return None
    header_start = func_code.index("def ")
    # find the next "(" after header_start
    if "(" not in func_code[header_start:]:
        return None
    paren_index = func_code.index("(", header_start)
    # the function name is between header_start + 4 and paren_index
    anonymized_name = (
        func_code[: header_start + 4] + "test_func" + func_code[paren_index:]
    )
    if "def test_func(" not in anonymized_name:
        return None
    new_paren_index = anonymized_name.index("(", header_start)
    if ")" not in anonymized_name[new_paren_index:]:
        return None
    paren_close_index = anonymized_name.index(")", new_paren_index)
    args_raw = anonymized_name[new_paren_index + 1 : paren_close_index].split(",")
    def_end = anonymized_name.index(":", paren_close_index)
    preamble = anonymized_name[: header_start + 4]
    header = anonymized_name[header_start + 4 : def_end + 1]
    if "-> " in header:
        header = header.split("->")[0].strip() + ":"    
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
        header = header.replace(f" {arg_name})", f" arg{i})")
        header = header.replace(f",{arg_name})", f",arg{i})")                
        body = f"    {arg_name} = arg{i}\n" + body
    # strip out any potential type annotations from the header
    if len(args_raw) == 0:
        pass
    else:
        current_arg_index = 0
        final_arg_index = len(args_raw) - 1
        while current_arg_index <= final_arg_index:
            arg_position = header.find(f"arg{current_arg_index}")
            if arg_position == -1:
                print(f"Could not find argument position for arg{current_arg_index} in header: {header}")
                return None
            if current_arg_index == final_arg_index:
                next_arg_position = header.find("):", arg_position)
                header = header[:arg_position] + f"arg{current_arg_index}):"
            else:
                next_arg_position = header.find(f"arg{current_arg_index + 1}", arg_position)
                if next_arg_position == -1:
                    print(f"Could not find next argument position for arg{current_arg_index + 1} in header: {header}")
                    return None
                header = header[:arg_position] + f"arg{current_arg_index}, " + header[next_arg_position:]                
            current_arg_index += 1
    # adding a validate function at the start of the body
    # should be able to exec anonymized code without the validate_input_args call now, do it and if it fails I need to debug:
    validate_call = "    validate_input_args("
    for i, arg_raw in enumerate(args_raw):
        if i > 0:
            validate_call += ", "
        validate_call += f"arg{i}"
    validate_call += ")\n"
    body = validate_call + body
    saw_return = False
    for i, line in enumerate(body.split("\n")):
        if "return " in line:
            saw_return = True
        if line == "" and saw_return: 
            body = "\n".join(body.split("\n")[:i])
            break
    anonymized_code = preamble + header + "\n" + body
    return anonymized_code





class InitialFormatter:
    """
    Returns a jsonl with the columns: revealed_func, test_func, description
    """
    @staticmethod
    def can_eval_func(func_code):
        validation_function = "def validate_input_args(**kwargs):\n    pass"
        full_code = validation_function + "\n" + func_code
        try:
            runner = RunTestFunc(full_code)
            return True
        except Exception as e:
            return False
        
    @staticmethod
    def drop_non_eval_funcs(df, func_column="test_func"):
        for idx, row in tqdm(df.iterrows(), desc="Dropping non-evalable functions", total=len(df)):
            func_code = row[func_column]
            if not InitialFormatter.can_eval_func(func_code):
                df.at[idx, func_column] = None
        initial_length = len(df)
        df = df[~df[func_column].isna()].reset_index(drop=True)
        final_length = len(df)
        log_info(f"Dropped {initial_length - final_length}/{initial_length} functions that could not be evaled.")
        return df


    @staticmethod
    def load_cruxeval(parameters):
        parameters = load_parameters(parameters)
        dataset = load_dataset("cruxeval-org/cruxeval", split="test").to_pandas()
        # rename code column to test_func
        dataset = dataset.rename(columns={"code": "test_func"})
        dataset["test_func"] = dataset["test_func"].apply(anonymize_header)

        # dataset['inputs'] = dataset['inputs'].apply(lambda x: [x])
        # dataset['outputs'] = dataset['outputs'].apply(lambda x: [x])
        def get_docstring_func(row, func_column="test_func"):
            func = row[func_column]
            if func is None:
                return None
            first_indented_line = func.index("    validate_input_args")
            # insert a docstring before this
            doc_text = (
                "Example usage: \n"
                + "    >>> test_func("
                + row["input"]
                + ")\n"
                + "    >>> "
                + row["output"]
            )
            func = (
                func[:first_indented_line]
                + f'    """\n    {doc_text}\n    """\n'
                + func[first_indented_line:]
            )
            return func

        dataset["revealed_func"] = dataset.apply(
            lambda x: get_docstring_func(x, "test_func"), axis=1
        )
        dataset["description_prompt"] = dataset["revealed_func"].apply(
            lambda x: Prompts.describe + x + "\nDescription: "
        )
        dataset = InitialFormatter.drop_non_eval_funcs(dataset, "test_func")
        model_name = parameters["benchmark_creation_model"]
        model_engine = parameters["benchmark_creation_model_engine"]
        model = model_factory(model_name=model_name, model_kind="lm", model_engine=model_engine)
        dataset["description"] = None
        for idx, row in tqdm(dataset.iterrows(), desc="Generating descriptions", total=len(dataset)):
            prompt = row["description_prompt"]
            description = model.infer(prompt, max_new_tokens=100)
            dataset.at[idx, "description"] = description.strip()
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

        def get_setup(prompt):
            all_funcs = prompt.split("def ")
            if len(all_funcs) <= 1:
                return ""
            setup = "def ".join(all_funcs[:-1])
            return setup

        def last_function(prompt):
            funcs = prompt.split("def ")
            return "def " + funcs[-1]

        dataset["function_only"] = (
            dataset["prompt"].apply(last_function) + dataset["canonical_solution"]
        )
        dataset["test_func"] = dataset["prompt"].apply(get_setup) + dataset[
            "function_only"
        ].apply(anonymize_header)
        dataset["revealed_func"] = dataset["prompt"] + "\n" + dataset["canonical_solution"]
        dataset["description_prompt"] = dataset["revealed_func"].apply(
            lambda x: Prompts.describe
            + x
            + "\nDescription: "
        )
        dataset = InitialFormatter.drop_non_eval_funcs(dataset, "test_func")
        model_name = parameters["benchmark_creation_model"]
        model_engine = parameters["benchmark_creation_model_engine"]
        model = model_factory(model_name=model_name, model_kind="lm", model_engine=model_engine)
        dataset["description"] = None
        for idx, row in tqdm(dataset.iterrows(), desc="Generating descriptions", total=len(dataset)):
            prompt = row["description_prompt"]
            description = model.infer(prompt, max_new_tokens=100)
            dataset.at[idx, "description"] = description.strip()
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

        dataset["revealed_func"] = dataset.apply(
            lambda x: add_docstring(x, "test_func"), axis=1
        )
        dataset["test_func"] = dataset["test_func_anon"]
        dataset["description"] = dataset["prompt"].apply(
            lambda x: x.split("function to ")[-1].strip()
        )
        dataset = InitialFormatter.drop_non_eval_funcs(dataset, "test_func")
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
        df["test_func"] = df["output"].apply(anonymize_header)
        df["revealed_func"] = df["output"]

        def insert_docstring(row, func_column="test_func"):
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

        df["revealed_func"] = df.apply(insert_docstring, axis=1)
        df["description_prompt"] = df["revealed_func"].apply(
            lambda x: Prompts.describe + x + "\nDescription: "
        )
        df = InitialFormatter.drop_non_eval_funcs(df, "test_func")
        model_name = parameters["benchmark_creation_model"]
        model_engine = parameters["benchmark_creation_model_engine"]
        model = model_factory(model_name=model_name, model_kind="lm", model_engine=model_engine)
        df["description"] = None
        for idx, row in tqdm(df.iterrows(), desc="Generating descriptions", total=len(df)):
            prompt = row["description_prompt"]
            description = model.infer(prompt, max_new_tokens=100)
            df.at[idx, "description"] = description.strip()
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
        df["test_func"] = df["func"].apply(anonymize_header)
        df = df[~df["test_func"].isna()].reset_index(drop=True)

        def insert_docstring(row, func_column="test_func"):
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

        df["revealed_func"] = df.apply(insert_docstring, axis=1)
        df["description_prompt"] = df["revealed_func"].apply(
            lambda x: Prompts.describe + x + "\nDescription: "
        )
        df = InitialFormatter.drop_non_eval_funcs(df, "test_func")
        #df = df.sample(frac=0.75, random_state=42).reset_index(drop=True)
        model_name = parameters["benchmark_creation_model"]
        model_engine = parameters["benchmark_creation_model_engine"]
        model = model_factory(model_name=model_name, model_kind="lm", model_engine=model_engine)
        df["description"] = None
        for idx, row in tqdm(df.iterrows(), desc="Generating descriptions", total=len(df)):
            prompt = row["description_prompt"]
            description = model.infer(prompt, max_new_tokens=100)
            df.at[idx, "description"] = description.strip() 
        return {"train": df}
    

def get_validation_output(validation_func):
    validation_func = validation_func.strip()
    if "return" in validation_func:
        return validation_func.split("return")[-1].strip() + "\n"
    else:
        return validation_func + "\n"

def create_validation_function(dataset, parameters):
    dataset["validation_prompt"] = dataset["revealed_func"].apply(lambda x: Prompts.validation_creator + x + "\nValidation Function:\n")
    parameters = load_parameters(parameters)
    model_name = parameters["benchmark_creation_model"]
    model_engine = parameters["benchmark_creation_model_engine"]
    model = model_factory(model_name=model_name, model_kind="lm", model_engine=model_engine)
    model["validation_func"] = None
    model["test_func_validated"] = None
    for idx, row in tqdm(dataset.iterrows(), desc="Generating validation functions", total=len(dataset)):
        prompt = row["validation_prompt"]
        validation_func = model.infer(prompt, max_new_tokens=200)
        dataset.at[idx, "validation_func"] = get_validation_output(validation_func)
        dataset.at[idx, "test_func_validated"] = row["validation_func"] + "\n" + row["test_func"]
        try:
            runner = RunTestFunc(row["test_func_validated"])
        except Exception as e:
            dataset.at[idx, "test_func_validated"] = None
    # drop any rows where test_func_validated is None
    initial_length = len(dataset)
    dataset = dataset[~dataset["test_func_validated"].isna()].reset_index(drop=True)
    final_length = len(dataset)
    log_info(f"Dropped {initial_length - final_length}/{initial_length} functions that could not be validated.")
    return dataset

def create_examples(dataset, parameters):
    # make sure to store the input output examples in separate columns as reprs with a try and except in case they cant be repred. 
    pass

    

@click.group()
def cli():
    pass


@click.command()
@click.option("--dataset", type=click.Choice(TRAIN_DATASETS + TEST_DATASETS), required=True, help="The dataset to load and format.")
def create_initial(dataset):
    parameters = load_parameters()
    df_dict = None
    log_info(f"Creating initial formatted data for {dataset}...")
    if dataset == "cruxeval":
        df_dict = InitialFormatter.load_cruxeval(parameters)
    elif dataset == "humaneval":
        df_dict = InitialFormatter.load_humaneval(parameters)
    elif dataset == "mbpp":
        df_dict = InitialFormatter.load_mbpp(parameters)
    elif dataset == "code_alpaca":
        df_dict = InitialFormatter.load_code_alpaca(parameters)
    elif dataset == "magic_coder":
        df_dict = InitialFormatter.load_magic_coder(parameters)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    save_path = parameters["data_dir"] + f"/benchmark_processing/initial/{dataset}/"  
    os.makedirs(save_path, exist_ok=True)
    for split, df in df_dict.items():
        df.to_json(save_path + f"{split}.jsonl", orient="records", lines=True)
    log_info(f"Saved initial formatted data for {dataset} to {save_path}")


@click.command()
def join_all():
    parameters = load_parameters()
    initial_path = parameters["data_dir"] + f"/benchmark_processing/initial/"
    dfs = []
    get_split = lambda dataset: "train" if dataset in TRAIN_DATASETS else "test"
    for dataset in TRAIN_DATASETS + TEST_DATASETS:
        split = get_split(dataset)
        path = initial_path + f"{dataset}/{split}.jsonl"
        if not os.path.exists(path):
            log_error(f"{split} file for {dataset} not found at {path}.", parameters=parameters)
        df = pd.read_json(path, orient="records", lines=True)
        df["dataset"] = dataset
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_json(parameters["data_dir"] + f"/benchmark_processing/all_{split}.jsonl", orient="records", lines=True)
    log_info(f"Saved all train data to {parameters['data_dir'] + f'/benchmark_processing/all_{split}.jsonl'}")


def do_validation(split, parameters):
    load_path = parameters["data_dir"] + f"/benchmark_processing/all_{split}.jsonl"
    save_path = parameters["data_dir"] + f"/benchmark_processing/validated_{split}.jsonl"
    if not os.path.exists(load_path):
        log_error(f"All {split} file not found at {load_path}.", parameters=parameters)
    if os.path.exists(save_path):
        log_info(f"Validated {split} file already exists at {save_path}, skipping validation.", parameters=parameters)
        return
    df = pd.read_json(load_path, orient="records", lines=True)
    df = create_validation_function(df, parameters)
    df.to_json(save_path, orient="records", lines=True)
    log_info(f"Saved validated {split} data to {save_path}", parameters=parameters)

def do_example_creation(split, parameters):
    load_path = parameters["data_dir"] + f"/benchmark_processing/validated_{split}.jsonl"
    save_path = parameters["data_dir"] + f"/benchmark_processing/validated_{split}_with_examples.jsonl"
    if not os.path.exists(load_path):
        log_error(f"Validated {split} file not found at {load_path}.", parameters=parameters)
    if os.path.exists(save_path):
        log_info(f"Validated with examples {split} file already exists at {save_path}, skipping example creation.", parameters=parameters)
        return
    df = pd.read_json(load_path, orient="records", lines=True)
    df = create_examples(df, parameters)
    df.to_json(save_path, orient="records", lines=True)
    log_info(f"Saved validated with examples {split} data to {save_path}", parameters=parameters)

@click.command()
def validate_all():
    parameters = load_parameters()
    do_validation("train", parameters)
    do_validation("test", parameters)

@click.command()
def create_examples_all():
    parameters = load_parameters()
    do_example_creation("train", parameters)
    do_example_creation("test", parameters)


cli.add_command(create_initial, name="initial")
cli.add_command(join_all, "join")
cli.add_command(validate_all, "validate")
cli.add_command(create_examples_all, "examples")


if __name__ == "__main__":
    cli()