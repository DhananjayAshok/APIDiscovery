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
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd
import os
from ast import literal_eval

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
Now, generate the validate_input_args function for the following function only. After that, say [STOP]
Function: 

"""
    example_creator = """
You are given a function definition. Your task is to create example inputs to the function that satisfy the constraints, and also trigger the different branches of the function logic. Output as many examples as you can, that test different parts of the function.     
Output each example on a new line, in the format:
 - (arg0, arg1, ..., argN)
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
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
Examples:
 - (2)
 - (4)
 - (15)
 - (17)
 - (9)
 [STOP]

Note, the type checks in validate_input_args are bindings. So you must always ensure that the inputs you are generating satisfy those type checks. For example, if validate_input_args checks that an argument is a float, you MUST give a float in your examples, not an int. 
Now do this for the following function only. After that, say [STOP].
Examples:
"""
    more_examples = """
Given the function:
[FUNC]

We can have the following example inputs:
[EXAMPLES]
Note, the type checks in validate_input_args are bindings. So you must always ensure that the inputs you are generating satisfy those type checks. For example, if validate_input_args checks that an argument is a float, you MUST give a float in your examples, not an int. 
Create an exhaustive list of as many example inputs as you can make, which test various edge cases and functionality of the function. 
Answer in the format: 
 - (arg0, arg1, ..., argN)

 When you have finished the list, say [STOP]

Examples:
"""
    describe = """
Given the function, briefly describe what the function does in a concise manner.
Example:
Function:
def test_func(arg0: List[float], arg1: float) -> bool:
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
Note, there is no need to describe the validate_input_args function if it exists, only test_func
Now, describe the following function only and then say [STOP]
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
    # should be able to exec anonymized code without the validate_input_args call now, do it and if it fails I need to debug:    
    validate_call = "    validate_input_args("
    for i, arg_raw in enumerate(args_raw):
        if i > 0:
            validate_call += ", "
        validate_call += f"arg{i}"
    validate_call += ")\n"
    # if body does not start with an indent, add one
    if not body.startswith("    "):
        body = "    " + body
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
        #dataset['inputs'] = dataset['inputs'].apply(lambda x: [x])
        #dataset['outputs'] = dataset['outputs'].apply(lambda x: [x])
        def get_docstring_func(row):
            func = row["test_func_anon"]
            first_indented_line = func.index("    validate_input_args")
            # insert a docstring before this
            doc_text = "Example usage: \n" + ">>> test_func(" + row['input'] + ")\n" + ">>> " + row['output']
            func = func[:first_indented_line] + f'    """\n    {doc_text}\n    """\n' + func[first_indented_line:]
            return func
        dataset["test_func_anon_w_docstring"] = dataset.apply(get_docstring_func, axis=1)
        dataset["validation_prompt"] = dataset["test_func_anon_w_docstring"].apply(lambda x: Prompts.validation_creator + x + "\nValidation Function:\n")
        dataset["example_prompt"] = dataset["test_func_anon_w_docstring"].apply(lambda x: Prompts.example_creator + x)
        dataset["describe_prompt"] = dataset["test_func_anon_w_docstring"].apply(lambda x: Prompts.describe + x + "\nDescription: This function ")
        dataset = RawLoaders.generate_validation(dataset)
        dataset = RawLoaders.generate_examples(dataset)
        dataset = RawLoaders.generate_more_examples(dataset)
        dataset = RawLoaders.generate_description(dataset)
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
        dataset["test_func_anon"] = dataset["prompt"].apply(get_setup) + dataset['function_only'].apply(anonymize_header)
        dataset["validation_prompt"] = dataset["test_func_anon"].apply(lambda x: Prompts.validation_creator + x + "\nValidation Function:\n")
        dataset["example_prompt"] = dataset["test_func_anon"].apply(lambda x: Prompts.example_creator + x)
        dataset["describe_prompt"] = dataset["test_func_anon"].apply(lambda x: Prompts.describe + x + "\nDescription: This function ")
        dataset = RawLoaders.generate_validation(dataset)
        dataset = RawLoaders.generate_examples(dataset)
        dataset = RawLoaders.generate_more_examples(dataset)    
        dataset = RawLoaders.generate_description(dataset)    
        return {"test": dataset}

    @staticmethod
    def load_mbpp(parameters):
        # Muennighoff/mbpp
        dataset = load_dataset("Muennighoff/mbpp", "sanitized", split="test").to_pandas()
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
        dataset["test_func_anon"] = dataset["test_func"].apply(get_setup) + dataset["test_func"].apply(last_function).apply(anonymize_header)

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
        def add_docstring(row):
            func = last_function(row['test_func_anon'])
            text = row['prompt'].split("function to ")[-1].strip()
            test_list = row['test_list']
            text = text + "\nWill end up satisfying:\n" + "\n".join(test_list)
            # in cruxeval, function declaration is always first and there are no type hints
            header, body = func.split("):", 1)
            docstring = f'    """\n    {text}\n    """'
            new_func = header + "):" + docstring + body
            return new_func
        dataset["validation_prompt"] = dataset.apply(add_docstring, axis=1).apply(lambda x: Prompts.validation_creator + x + "\nValidation Function:\n")        
        dataset["example_prompt"] = dataset["test_func_anon"].apply(lambda x: Prompts.example_creator + x)
        dataset["description"] = dataset["prompt"].apply(lambda x: x.split("function to ")[-1].strip())
        dataset = RawLoaders.generate_validation(dataset)
        dataset = RawLoaders.generate_examples(dataset)
        dataset = RawLoaders.generate_more_examples(dataset)
        return {"test": dataset}
    
    @staticmethod
    def load_code_alpaca(parameters):
        df = load_dataset("sahil2801/CodeAlpaca-20k", split="train").to_pandas()
        df = df[df["output"].str.startswith("def ")].reset_index(drop=True)
        df = df[df['output'].apply(lambda x: "end" not in x and "{" not in x and "}" not in x)].reset_index(drop=True)
        # manually removing index 129 which has a weird function
        df = df.drop(index=129).reset_index(drop=True)    
        df["test_func_anon"] = df["output"].apply(anonymize_header)
        def insert_docstring(row):
            func = row["test_func_anon"]
            first_indented_line = func.index("    validate_input_args")
            # insert a docstring before this
            doc_text = row["instruction"]
            func = func[:first_indented_line] + f'    """\n    {doc_text}\n    """\n' + func[first_indented_line:]
            return func
        df["test_function_anon_w_docstring"] = df.apply(insert_docstring, axis=1)
        df["validation_prompt"] = df["test_function_anon_w_docstring"].apply(lambda x: Prompts.validation_creator + x + "\nValidation Function:\n")
        df["example_prompt"] = df["test_function_anon_w_docstring"].apply(lambda x: Prompts.example_creator + x)
        df["describe_prompt"] = df["test_function_anon_w_docstring"].apply(lambda x: Prompts.describe + x + "\nDescription: This function ")
        df = RawLoaders.generate_validation(df)
        df = RawLoaders.generate_examples(df)
        df = RawLoaders.generate_more_examples(df)
        df = RawLoaders.generate_description(df)
        return {"train": df}
    
    @staticmethod
    def load_magic_coder(parameters):
        df = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train").to_pandas()
        df = df[df['lang'] == "python"].reset_index(drop=True)
        def get_func(x):
            if x.count("```python") != 1:
                return None
            pstart = x.index("```python") + len("```python")
            x = x[pstart:]
            if x.count("```") != 1:
                return None
            x = x.split("```")[0]
            return x
        df['func'] = df['solution'].apply(get_func)
        df["test_func_anon"] = df["func"].apply(anonymize_header)
        def insert_docstring(row):
            func = row["test_func_anon"]
            first_indented_line = func.index("    validate_input_args")
            # insert a docstring before this
            doc_text = row["problem"]
            func = func[:first_indented_line] + f'    """\n    {doc_text}\n    """\n' + func[first_indented_line:]
            return func
        df["test_function_anon_w_docstring"] = df.apply(insert_docstring, axis=1)
        df["validation_prompt"] = df["test_function_anon_w_docstring"].apply(lambda x: Prompts.validation_creator + x + "\nValidation Function:\n")
        df["example_prompt"] = df["test_function_anon_w_docstring"].apply(lambda x: Prompts.example_creator + x)
        df["describe_prompt"] = df["test_function_anon_w_docstring"].apply(lambda x: Prompts.describe + x + "\nDescription: This function ")
        df = RawLoaders.generate_validation(df)
        df = RawLoaders.generate_examples(df)
        df = RawLoaders.generate_more_examples(df)
        df = RawLoaders.generate_description(df)
        return {"train": df}
        
    @staticmethod
    def generate_validation(df):
        model = HuggingFaceModel("meta-llama/Meta-Llama-3-8B-Instruct")
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating validation code"):
            prompt = row["validation_prompt"]
            validation_output = model.generate(prompt, max_new_tokens=500)
            df.at[index, "validation_output"] = validation_output
            if "def validate_input_args(" in validation_output:
                validation_code =  "def validate_input_args(" + validation_output.split("def validate_input_args(")[1]
                if "return" in validation_code:
                    validation_code = validation_code.split("return")[0] + "return"
                else:
                    validation_code = None
            else:
                validation_code = None      
            if validation_code is None:
                new_prompt = prompt + "Do not give an incomplete function or incorrect format response. Example: \n" + validation_output + "\n The above response is WRONG because it does not contain the full validation function implementation from start to finish. Write the validation function now, start with the def and ending with the return. Validation Function:\n"
                validation_output = model.generate(new_prompt, max_new_tokens=800)
                df.at[index, "validation_output"] = validation_output
                if "def validate_input_args(" in validation_output:
                    validation_code =  "def validate_input_args(" + validation_output.split("def validate_input_args(")[1]
                    if "return" in validation_code:
                        validation_code = validation_code.split("return")[0] + "return"
                    else:
                        log_warn(validation_output)
                        log_warn("Became")
                        log_warn(validation_code)                        
                        validation_code = None
                else:
                    log_warn(validation_output)
                    validation_code = None      
            if validation_code is None:
                log_warn(f"Could not generate validation code for index {index}\n" + validation_output, parameters=loaded_parameters)
                continue
            df.at[index, "validation_code"] = validation_code
            df.at[index, "test_func_validated"] = move_imports_top(validation_code + "\n" + row["test_func_anon"])
        return df
    
    @staticmethod
    def generate_examples(df):
        model = HuggingFaceModel("meta-llama/Meta-Llama-3-8B-Instruct")
        df["examples"] = None
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating example inputs"):
            prompt = row["example_prompt"]
            example_code = model.generate(prompt, max_new_tokens=500)
            df.at[index, "example_output"] = example_code
            examples = []
            for line in example_code.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    line = line[2:].strip()
                if line.startswith("(") and line.endswith(")"):                    
                    examples.append("test_func"+line)
            df.at[index, "examples"] = examples
            function_str = prompt.split("Function: ")[-1].strip()
            more_examples_prompt = Prompts.more_examples.replace("[FUNC]", function_str)
            more_examples_prompt.replace("[EXAMPLES]", example_code)
            df.at[index, "more_examples_prompt"] = more_examples_prompt
        return df
    
    @staticmethod
    def generate_more_examples(df):
        model = HuggingFaceModel("meta-llama/Meta-Llama-3-8B-Instruct")
        df["more_examples"] = None
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating more example inputs"):
            prompt = row["more_examples_prompt"]
            example_code = model.generate(prompt, max_new_tokens=500)
            df.at[index, "more_examples_output"] = example_code
            examples = []
            for line in example_code.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    line = line[2:].strip()
                if line.startswith("(") and line.endswith(")"):                    
                    examples.append("test_func"+line)
            df.at[index, "more_examples"] = examples
        return df

    @staticmethod
    def generate_description(df):
        model = HuggingFaceModel("meta-llama/Meta-Llama-3-8B-Instruct")
        df["description"] = None
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Generating descriptions"):
            prompt = row["describe_prompt"]
            description = model.generate(prompt, max_new_tokens=200)
            df.at[index, "description"] = description.replace("Description:", "").strip()
        return df

class RunTestFunc:
    """
    A class to run a test function defined in code.
    """
    def __init__(self, func_code: str):
        """
        Initializes the RunTestFunc with the given function code. Is not safe (i.e. runs exec on func_code, unsure you do not run malicious code through here by mistake).

        :param func_code: The code defining the test function. Should come from the provided dataset. 
        :type func_code: str
        """
        self.func_code = func_code
        exec(func_code)
        self.test_func = locals()["test_func"]
        self.access_counter = 0

    def run_test(self, *args):
        """
        Runs the test function with the given arguments.
        
        :param args: Arguments to pass to the test function.
        :return: A tuple (return_value, error_message). If there is no error, error_message is None.
        :rtype: tuple
        """
        returns = None
        self.access_counter += 1
        try:
            returns = self.test_func(*args)
        except AssertionError as e:
            return None, str(e)
        except Exception as e:
            return None, str(e)
        return returns, None
    
    def run_test_str(self, args_str: str):
        """
        Runs the test function with the given arguments in string form.
        
        :param args_str: Arguments in string form to pass to the test function.
        :type args_str: str
        :return: A tuple (return_value, error_message). If there is no error, error_message is None.
        :rtype: tuple
        """
        try:
            args = literal_eval(args_str) # for safety
        except Exception as e:
            return None, "Invalid input args, is not valid python syntax"
        if not isinstance(args, tuple) and not isinstance(args, list):
            args = (args,) # for single argument functions
        return self.run_test(*args)

class FilteredLoader:
    @staticmethod
    def filter_examples(test_func_code: str, examples: list):
        runner = RunTestFunc(test_func_code)
        filtered_inputs = []
        filtered_outputs = []
        errored_inputs = []
        errored_outputs = []
        for example in examples:
            try:
                args = eval(example)
                return_value, error = runner.run_test(*args)
                if error is None:
                    filtered_inputs.append(example)
                    filtered_outputs.append(repr(return_value))
                else:
                    errored_inputs.append(example)
                    errored_outputs.append(error)
            except Exception as e:
                errored_inputs.append(example)
                errored_outputs.append(str(e))
        return filtered_inputs, filtered_outputs, errored_inputs, errored_outputs


    @staticmethod
    def filter_annotate_dataset(df):
        random.seed(42)
        # dataset has columns: test_func_validated, examples, more_examples, description
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Filtering and annotating dataset"):
            test_func_code = row["test_func_validated"]
            all_examples = list(set(row["examples"] + row["more_examples"]))
            # shuffle all examples
            random.shuffle(all_examples)
            filtered_inputs, filtered_outputs, errored_inputs, errored_outputs = FilteredLoader.filter_examples(test_func_code, all_examples)
            # first 2 filtered inputs/outputs are train, rest are test
            train_inputs = filtered_inputs[:2]
            train_outputs = filtered_outputs[:2]
            test_inputs = filtered_inputs[2:]
            test_outputs = filtered_outputs[2:]
            df.at[index, "train_inputs"] = train_inputs
            df.at[index, "train_outputs"] = train_outputs
            df.at[index, "test_inputs"] = test_inputs
            df.at[index, "test_outputs"] = test_outputs
        return df



@click.command()
@click.option("--dataset_name", required=True, help="The name of the dataset to load from the Hugging Face Hub.", type=click.Choice(TEST_DATASETS + TRAIN_DATASETS))
@click.pass_obj
def process_raw(parameters, dataset_name):
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
        csv_clean = csv[["test_func_validated", "description", "examples", "more_examples"]]
        csv_clean.to_json(f"{save_dir}/{data_split}_clean.jsonl", orient="records", lines=True)
        csv.to_json(f"{save_dir}/{data_split}_proc.jsonl", orient="records", lines=True)
        log_info(f"Saved {dataset_name} split {data_split} to {save_dir}", parameters=parameters)


@click.command()
@click.option("--dataset_name", required=True, help="The name of the dataset to load from the Hugging Face Hub.", type=click.Choice(TEST_DATASETS + TRAIN_DATASETS))
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
        if file.endswith("_clean.jsonl"):
            split_name = file.replace("_clean.jsonl", "")
            split_dict[split_name] = pd.read_json(f"{load_dir}/{file}", lines=True)
    if split_dict == {}:
        log_error(f"No cleaned jsonl files found in {load_dir} for dataset {dataset_name}", parameters=parameters)
    for split_name in split_dict:
        df = split_dict[split_name]
        df_filtered = FilteredLoader.filter_annotate_dataset(df)
        df_filtered_clean = df_filtered[["test_func_validated", "description", "train_inputs", "train_outputs", "test_inputs", "test_outputs"]]
        df_filtered_clean.to_json(f"{save_dir}/{split_name}_final.jsonl", orient="records", lines=True)
        log_info(f"Saved final filtered dataset for {dataset_name} split {split_name} to {save_dir}", parameters=parameters)
        dataset = Dataset.from_pandas(df_filtered_clean)
        dataset.push_to_hub(f"{huggingface_hub_repo_name}", organization=huggingface_hub_username, private=False, split=split_name, config=dataset_name)
        



@click.group()
@click.pass_context
def main(ctx):
    compute_secondary_parameters(loaded_parameters)
    ctx.obj = loaded_parameters


main.add_command(process_raw, name="process_raw")
main.add_command(process_final, name="process_final")

if __name__ == "__main__":
    main()