from utils.parameter_handling import load_parameters, compute_secondary_parameters
from utils import log_error, log_info, log_warn
import click
from datasets import load_dataset
import os

loaded_parameters = load_parameters()


TEST_DATASETS = ["cruxeval", "mbpp", "humaneval"]
TRAIN_DATASETS = []


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

Types and Constraints:
- arg0: List of floats. Each element should be a float.
- arg1: Float. Represents the threshold value.

validate_input_args(arg0: List[float], arg1: float) -> None:
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

Types and Constraints:
- arg0: Tuple of elements. Can contain any hashable type.
- arg1: Tuple of elements. Can contain any hashable type.
validate_input_args(arg0: tuple, arg1: tuple) -> None:
    if not isinstance(arg0, tuple):
        raise TypeError("arg0 must be a tuple")
    if not isinstance(arg1, tuple):
        raise TypeError("arg1 must be a tuple")
    return     
[STOP]
Function: 

"""

    
def anonymize_header(func_code: str) -> str:
    header_start = func_code.index("def ")
    # find the next "(" after header_start
    paren_index = func_code.index("(", header_start)
    # the function name is between header_start + 4 and paren_index
    anonymized_name = func_code[:header_start + 4] + "test_func" + func_code[paren_index:]
    new_paren_index = anonymized_name.index("(", header_start)
    paren_close_index = anonymized_name.index(")", new_paren_index)
    args_raw = anonymized_name[new_paren_index + 1:paren_close_index].split(",")
    def_end = anonymized_name.index(":", paren_close_index)
    preamble = anonymized_name[:header_start + 4]
    header = anonymized_name[header_start+4:def_end + 1]
    body = anonymized_name[def_end + 1:]
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
    validate_call = "    validate_input_args("
    for i, arg_raw in enumerate(args_raw):
        if i > 0:
            validate_call += ", "
        validate_call += f"arg{i}"
    validate_call += ")\n"
    body = validate_call + body
    anonymized_code = preamble + header + "\n" + body
    return anonymized_code


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
        dataset["test_func_alone"] = dataset["test_func"].apply(anonymize_header)
        #dataset['inputs'] = dataset['inputs'].apply(lambda x: [x])
        #dataset['outputs'] = dataset['outputs'].apply(lambda x: [x])
        dataset["validation_prompt"] = dataset["test_func_alone"].apply(lambda x: Prompts.validation_creator + x)
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
        dataset.at[50, 'prompt'] = new_50
        dataset = dataset.drop(index=38).reset_index(drop=True)
        def drop_docstrings(prompt):
            while '"""' in prompt:
                first_index = prompt.index('"""')
                second_index = prompt.index('"""', first_index + 3)
                prompt = prompt[:first_index] + prompt[second_index + 3:]
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
        

        dataset['header_only'] = dataset['prompt'].apply(drop_docstrings)
        dataset['function_only'] = dataset['header_only'].apply(last_function) + dataset['canonical_solution']
        dataset["test_func_alone"] = dataset["prompt"].apply(get_setup) + dataset['function_only'].apply(anonymize_header)
        dataset["validation_prompt"] = dataset["test_func_alone"].apply(lambda x: Prompts.validation_creator + x)
        return {"test": dataset}


    def load_mbpp(parameters):
        # Muennighoff/mbpp
        dataset = load_dataset("Muennighoff/mbpp", "sanitized", split="test").to_pandas()
        # functionaly the exact same as cruxeval
        dataset = dataset.rename(columns={"code": "test_func"})
        dataset["test_func_alone"] = dataset["test_func"].apply(anonymize_header)
        def add_docstring(row):
            func = row['test_func']
            text = row['text'].split("function to ")[-1].strip()
            # in cruxeval, function declaration is always first and there are no type hints
            header, body = func.split("):", 1)
            docstring = f'    """\n    {text}\n    """'
            new_func = header + "):" + docstring + body
            return new_func
        dataset["validation_prompt"] = dataset.apply(add_docstring, axis=1).apply(lambda x: Prompts.validation_creator + x)
        return {"test": dataset}




@click.command()
@click.option("--dataset_name", required=True, help="The name of the dataset to load from the Hugging Face Hub.", type=click.Choice(TEST_DATASETS + TRAIN_DATASETS))
@click.pass_obj
def load_raw(parameters, dataset_name):
    if dataset_name == "humaneval":
        data_splits = RawLoaders.load_humaneval(parameters)
    elif dataset_name == "cruxeval":
        data_splits = RawLoaders.load_cruxeval(parameters)
    elif dataset_name == "mbpp":
        data_splits = RawLoaders.load_mbpp(parameters)
    else:
        log_error(f"Dataset {dataset_name} not recognized.", parameters=parameters)
    breakpoint()




@click.group()
@click.pass_context
def main(ctx):
    compute_secondary_parameters(loaded_parameters)
    ctx.obj = loaded_parameters


main.add_command(load_raw, name="load_raw")

if __name__ == "__main__":
    main()
