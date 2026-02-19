from ast import literal_eval
from utils import log_warn, log_info, load_parameters, file_makedir, log_error
from tqdm import tqdm
import pandas as pd
import click
import os
import subprocess
import multiprocessing
import re

class RunTestFunc:
    """
    A class to run a test function defined in code.
    """

    def __init__(self, func_code: str, timeout=0.5):
        """
        Initializes the RunTestFunc with the given function code. Is not safe (i.e. runs exec on func_code, ensure you do not run malicious code through here by mistake).

        :param func_code: The code defining the test function. Should come from the provided dataset.
        :type func_code: str
        :param timeout: The maximum time in seconds to allow the function to run.
        :type timeout: float
        """
        self.func_code = func_code
        self.access_counter = 0
        self.attempted_inputs = []
        self.received_outputs = []
        self.timeout = timeout        
        success = self.try_exec(func_code)
        self._context = {"__builtins__": __builtins__}
        if success:
            exec(func_code, self._context)
            self.test_func = self._context["test_func"]
        else:
            raise RuntimeError("Failed to exec function code, cannot initialize RunTestFunc.")


    @staticmethod
    def exec_worker(func_code, queue):
        """Helper worker to run exec and put the result in a queue."""
        try:
            exec(func_code, {"__builtins__": __builtins__})
            queue.put(True) # runs
        except Exception as e:
            queue.put(False) # fails

    def try_exec(self, func_code):
        """Tries to exec the given code in a separate process with a timeout."""
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=self.exec_worker, args=(func_code, queue))
        p.start()
        p.join(timeout=self.timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            return False
        if not queue.empty():
            return queue.get()
        return False

    def worker(self, func, args, queue):
        """Helper worker to run the function and put the result in a queue."""
        try:
            result = func(*args)
            queue.put((result, None))
        except Exception as e:
            queue.put((None, str(e)))

    def run_test(self, *args):
        self.access_counter += 1
        self.attempted_inputs.append(args)

        # Use a Queue to get the return value back from the child process
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=self.worker, args=(self.test_func, args, queue)
        )

        p.start()

        # Wait
        p.join(timeout=self.timeout)

        if p.is_alive():
            # If the process is still running after 15s, kill it
            p.terminate()
            p.join()
            self.received_outputs.append(
                (None, f"Timeout: Function execution exceeded {self.timeout} seconds")
            )
            return None, f"Timeout: Function execution exceeded {self.timeout} seconds"

        # If it finished, grab the result from the queue
        if not queue.empty():
            result = queue.get()
            self.received_outputs.append(result)
            return result
        self.received_outputs.append((None, "Unknown error during execution"))
        return None, "Unknown error during execution"

    def run_test_str(self, args_str: str):
        """
        Runs the test function with the given arguments in string form.

        :param args_str: Arguments in string form to pass to the test function.
        :type args_str: str
        :return: A tuple (return_value, error_message). If there is no error, error_message is None.
        :rtype: tuple
        """
        try:
            args = literal_eval(args_str)  # for safety
        except Exception as e:
            return None, "Invalid input args, is not valid python syntax"
        if not isinstance(args, tuple) and not isinstance(args, list):
            args = (args,)  # for single argument functions
        return self.run_test(*args)




eval_prompt = f"""
You are given a function description and a hypothesized description of what the function does.
Your task is to rate how accurate the hypothesized description is compared to the true description on a scale from 1 to 5, where 1 means "completely inaccurate" and 5 means "completely accurate".
First, provide an extremely brief explanation (1 sentence) of why you gave that rating. Then, provide your rating in the format "Rating: X" where X is an integer between 1 and 5.
Example:
True Function Description: This function takes a list of integers and returns True if there are any two integers in the list that sum to zero, otherwise it returns False.
Hypothesized Description: This function checks if there are two numbers in the list that add up to zero.
Explanation: The hypothesized description accurately captures the functionality of the true description.
Rating: 5 [STOP]

True Function Description: calulates the nth fibonacci number
Hypothesized Description: This function computes the factorial of a number.
Explanation: The hypothesized description is incorrect as the fibonacci sequence and factorial are different mathematical concepts.
Rating: 1 [STOP]

Now, provide your rating for the following description only. You absolutely must follow the format shown in the examples above and no matter what, you must provide a rating between 1 and 5.
True Function Description: [TRUE]
Hypothesized Description: [HYPOTHESIS]
Explanation (very short):"""

def parse_score(output):
    response = output.strip().lower()
    # look for the regex matching rating: followed by a number from 1 to 5, allowing for any amount of whitespace in between
    match = re.search(r"rating:\s*([1-5])", response)
    if match:
        return int(match.group(1))
    else:
        log_warn(
            "Could not find 'rating: [1-5]' in model response: "
            + response
        )
        return None

def parse_eval(evaluation_path):
    try:
        df = pd.read_json(evaluation_path, lines=True)
    except:
        log_warn(
            f"Output file {evaluation_path} not found after inference command. This can happen for openai inference. Run the script again when the batch is done. "
        )
        return
    if isinstance(df["score_output"][0], list):
        df["score_output"] = df["score_output"].apply(
            lambda x: x[0] if len(x) > 0 else ""
        )
    df["score"] = df["score_output"].apply(parse_score)
    nan_frac = df["score"].isna().mean()
    if nan_frac > 0:
        log_warn(f"Parsed scores from evaluation output, but {round(nan_frac * 100, 2)}% of scores are NaN. This can happen if the judge model did not follow the output format correctly.")
    if "n_queries" not in df.columns:
        df["n_queries"] = 0
    if "concluded" not in df.columns:
        df["concluded"] = False
    df.to_json(evaluation_path, orient="records", lines=True)
    if df is not None:
        avg_n_queries = df["n_queries"].mean()
        avg_score = df["score"].mean()
        perc_concluded = df["concluded"].mean()
        log_info(
            f"n_queries: {avg_n_queries}, concluded: {round(perc_concluded* 100, 2)}, score: {avg_score}"
        )
        log_info(df.groupby("concluded")["score"].mean())
        log_info(df[["n_queries", "score"]].mean())
    log_info(f"Saved scored predictions to {evaluation_path}")


def score_description_predictions(
    *,
    predictions_save_path,
    save_name,
    dataset_name,
    override_eval=False,
):
    parameters = load_parameters()
    model = parameters["evaluation_model_name"]
    model_save_name = model.split("/")[-1].strip()
    save_name = f"{save_name}-{dataset_name}-judge-{model_save_name}"
    evaluation_path = os.path.abspath(
        f"results/{dataset_name}/" + save_name + ".jsonl"
    )
    skip = False
    if not override_eval:
        if os.path.exists(evaluation_path):
            log_info(
                f"Scored predictions already exist at {evaluation_path}, skipping judge generation."
            )
            skip = True
    if not skip:
        df = pd.read_json(predictions_save_path, lines=True)
        def rectify_description(x):
            if isinstance(x, list):
                x = x[0] if len(x) > 0 else ""
            x = x.strip()
            x = x.split("\n")[0] # take only the first line if there are multiple
            return x
        df["predicted_description_clean"] = df["predicted_description"].apply(rectify_description)        
        def get_score_prompt(row):
            description = None
            if "description" in row:
                description = row["description"]
            elif "true_description" in row:
                description = row["true_description"]
            else:
                log_error(f"Row with columns: {row.keys()} does not contain a description column.")
            prompt_filled = eval_prompt.replace("[TRUE]", description).replace(
                "[HYPOTHESIS]", row["predicted_description_clean"]
            )
            return prompt_filled

        df["score_prompt"] = df.apply(get_score_prompt, axis=1)
        df.to_json(predictions_save_path, orient="records", lines=True)
        open_ai_batch_name = ""
        if "gpt" in model:
            open_ai_batch_name = f"{save_name}"
        openaibatch_str = "-n " + open_ai_batch_name if open_ai_batch_name != "" else ""
        command_string = f"bash scripts/infer.sh -i {predictions_save_path} -o {evaluation_path} -m {model} -c score_prompt -d score_output -t 300 -g judge {openaibatch_str} -r yes"
        log_info(f"Generating scores with command: {command_string}")
        subprocess.run(command_string, shell=True, check=True)
    parse_eval(evaluation_path)


def evaluate_code_predictions(true_code, predicted_code, test_inputs):
    can_exec_true_code = False
    can_exec_pred_code = False
    true_outputs = []
    predicted_outputs = []
    exact_matches = []
    ret = [can_exec_true_code, can_exec_pred_code, true_outputs, predicted_outputs, 0.0]
    try:
        runner = RunTestFunc(true_code)
        ret[0] = True
    except Exception as e:
        log_warn(f"Failed to initialize RunTestFunc with true code. Error: {str(e)}. This should not happen.")
        return ret
    if predicted_code is None:
        return ret
    try:
        pred_runner = RunTestFunc(predicted_code)
        ret[1] = True
    except Exception as e:
        return ret
    for test_input in test_inputs:
        true_output, true_error = runner.run_test_str(test_input)
        pred_output, pred_error = pred_runner.run_test_str(test_input)
        true_outputs.append((true_output, true_error))
        predicted_outputs.append((pred_output, pred_error))
        if true_error is None and pred_error is None:
            exact_matches.append(true_output == pred_output)
        else:
            exact_matches.append(False)
    ret[-1] = sum(exact_matches) / len(exact_matches) if len(exact_matches) > 0 else 0.0
    return ret


def evaluate_output_prediction(true_output, predicted_output):
    try:
        eval(predicted_output)
    except: 
        return False
    return true_output == eval(predicted_output)


def evaluate_input_prediction(true_code, target_output, predicted_input):
    try:
        runner = RunTestFunc(true_code)
    except Exception as e:
        log_warn(f"Failed to initialize RunTestFunc with true code. Error: {str(e)}. This should not happen.")
        return False, None
    pred_output, pred_error = runner.run_test_str(predicted_input)
    if pred_output is None and target_output is not None:
        return False, pred_output
    if pred_output == eval(target_output):
        return True, pred_output
    else:
        return False, pred_output


def score_code(predictions_save_path, override_eval=False):
    df = pd.read_json(predictions_save_path, lines=True)
    def rectify_code(x):
        if isinstance(x, list):
            x = x[0] if len(x) > 0 else ""
        if isinstance(x, str):
            x = x.strip()
        return x
    if "predicted_code" not in df.columns:
        log_error(f"predicted_code not in df with columns: {df.columns}")
    if "predicted_code_clean" in df.columns and not override_eval:
        log_info("predicted_code_clean column already exists, skipping code scoring.")
        return
    df["predicted_code_clean"] = df["predicted_code"].apply(rectify_code)        
    df["true_test_outputs"] = None
    df["predicted_test_outputs"] = None
    df["predicted_outputs_exact_match"] = None
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring code predictions"):
        true_code = row["test_func_validated"]
        predicted_code = row["predicted_code_clean"]
        test_inputs = row["test_inputs"]
        can_exec_true_code, can_exec_pred_code, true_outputs, predicted_outputs, exact_match = evaluate_code_predictions(true_code, predicted_code, test_inputs)
        df.at[idx, "can_exec_true_code"] = can_exec_true_code
        df.at[idx, "can_exec_pred_code"] = can_exec_pred_code
        df.at[idx, "true_test_outputs"] = true_outputs
        df.at[idx, "predicted_test_outputs"] = predicted_outputs
        df.at[idx, "predicted_outputs_exact_match"] = exact_match
    df.to_json(predictions_save_path, orient="records", lines=True)
    log_info(f"Saved scored predictions to {predictions_save_path}")
    log_info(f"Average exact match on test outputs: {df['predicted_outputs_exact_match'].mean()} +- {df['predicted_outputs_exact_match'].std()}")


def score_output_prediction(predictions_save_path, override_eval=False):
    df = pd.read_json(predictions_save_path, lines=True)
    if "predicted_output" not in df.columns:
        log_error(f"predicted_output not in df with columns: {df.columns}")
    if "output_prediction_correct_micro" in df.columns and not override_eval:
        log_info("output_prediction_correct_micro column already exists, skipping output prediction scoring.")
        return
    df["output_prediction_correct_micro"] = None
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring output predictions"):
        test_func_code = row["test_func_validated"]
        test_inputs = row["test_inputs"]
        true_output = []
        try:
            runner = RunTestFunc(test_func_code)
        except Exception as e:
            log_warn(f"Failed to initialize RunTestFunc with true code. Error: {str(e)}. This should not happen.")
            continue
        for test_input in test_inputs:
            output, error = runner.run_test_str(test_input)
            true_output.append(output)
        predicted_outputs = row["predicted_output"]
        matches = []
        for idx2, (predicted_output, true_out) in enumerate(zip(predicted_outputs, true_output)):
            match = evaluate_output_prediction(true_output=true_out, predicted_output=predicted_output)
            matches.append(match)
        df.at[idx, "output_prediction_correct_micro"] = sum(matches) / len(matches) if len(matches) > 0 else 0.0
    df.to_json(predictions_save_path, orient="records", lines=True)
    log_info(f"Saved scored predictions to {predictions_save_path}")
    log_info(f"Average micro accuracy on output predictions: {df['output_prediction_correct_micro'].mean()} +- {df['output_prediction_correct_micro'].std()}")

def score_input_prediction(predictions_save_path, override_eval=False):
    df = pd.read_json(predictions_save_path, lines=True)
    if "predicted_input" not in df.columns:
        log_error(f"predicted_input not in df with columns: {df.columns}")
    if "input_prediction_correct_micro" in df.columns and not override_eval:
        log_info("input_prediction_correct_micro column already exists, skipping input prediction scoring.")
        return
    df["input_prediction_correct_micro"] = None
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring input predictions"):
        true_code = row["test_func_validated"]
        target_output = row["target_outputs"]
        predicted_input = row["predicted_input"]
        matches = []
        for pred_inp, target_out in zip(predicted_input, target_output):
            match, pred_out = evaluate_input_prediction(true_code=true_code, target_output=target_out, predicted_input=pred_inp)
            matches.append(match)
        df.at[idx, "input_prediction_exact_match_micro"] = sum(matches) / len(matches) if len(matches) > 0 else 0.0
    df.to_json(predictions_save_path, orient="records", lines=True)
    log_info(f"Saved scored predictions to {predictions_save_path}")
    log_info(f"Average micro accuracy on input predictions: {df['input_prediction_exact_match_micro'].mean()} +- {df['input_prediction_exact_match_micro'].std()}")

@click.command()
@click.option(
    "--dataset_name",
    type=str,
    required=True,
    help="Name of the dataset.",
)
@click.option(
    "--save_name",
    type=str,
    default=None,
    help="Name to use when saving evaluation results. If not provided, will be derived from the model name.",
)
@click.option(
    "--override_eval",
    is_flag=True,
    help="Whether to override existing evaluation results.",
)
def eval_description(
    dataset_name,
    save_name,
    override_eval,
):
    predictions_save_path = os.path.abspath(f"results/{dataset_name}/{save_name}.jsonl")
    if not os.path.exists(predictions_save_path):
        log_error(
            f"Predictions file not found at {predictions_save_path}. Run the generation script first."
        )
    score_description_predictions(
        predictions_save_path=predictions_save_path,
        save_name=save_name,
        dataset_name=dataset_name,
        override_eval=override_eval,
    )

@click.command()
@click.option(
    "--dataset_name",
    type=str,
    required=True,
    help="Name of the dataset.",
)
@click.option(
    "--save_name",
    type=str,
    default=None,
    help="Name to use when saving evaluation results. If not provided, will be derived from the model name.",
)
@click.option(
    "--override_eval",
    is_flag=True,
    help="Whether to override existing evaluation results.",
)
def eval_code(
    dataset_name,
    save_name,
    override_eval,
):
    predictions_save_path = os.path.abspath(f"results/{dataset_name}/{save_name}_code_prediction.jsonl")
    if not os.path.exists(predictions_save_path):
        log_error(
            f"Predictions file not found at {predictions_save_path}. Run the generation script first."
        )
    score_code(
        predictions_save_path=predictions_save_path,
        override_eval=override_eval,
    )    


@click.command()
@click.option(
    "--dataset_name",
    type=str,
    required=True,
    help="Name of the dataset.",
)
@click.option(
    "--save_name",
    type=str,
    default=None,
    help="Name to use when saving evaluation results. If not provided, will be derived from the model name.",
)
@click.option(
    "--override_eval",
    is_flag=True,
    help="Whether to override existing evaluation results.",
)
def eval_output(
    dataset_name,
    save_name,
    override_eval,
):
    predictions_save_path = os.path.abspath(f"results/{dataset_name}/{save_name}_output_prediction.jsonl")
    if not os.path.exists(predictions_save_path):
        log_error(
            f"Predictions file not found at {predictions_save_path}. Run the generation script first."
        )
    score_output_prediction(
        predictions_save_path=predictions_save_path,
        override_eval=override_eval,
    )    


@click.command()
@click.option(
    "--dataset_name",
    type=str,
    required=True,
    help="Name of the dataset.",
)
@click.option(
    "--save_name",
    type=str,
    default=None,
    help="Name to use when saving evaluation results. If not provided, will be derived from the model name.",
)
@click.option(
    "--override_eval",
    is_flag=True,
    help="Whether to override existing evaluation results.",
)
def eval_input(
    dataset_name,
    save_name,
    override_eval,
):
    predictions_save_path = os.path.abspath(f"results/{dataset_name}/{save_name}_input_prediction.jsonl")
    if not os.path.exists(predictions_save_path):
        log_error(
            f"Predictions file not found at {predictions_save_path}. Run the generation script first."
        )
    score_input_prediction(
        predictions_save_path=predictions_save_path,
        override_eval=override_eval,
    )        

@click.group()
def main():
    pass

main.add_command(eval_description, name="description")
main.add_command(eval_code, name="code")
main.add_command(eval_output, name="output")
main.add_command(eval_input, name="input")


if __name__ == "__main__":
    main()
