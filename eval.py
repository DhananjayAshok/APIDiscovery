from utils import (
    log_warn,
    log_info,
    load_parameters,
    file_makedir,
    log_error,
    RunTestFunc,
    get_lm,
)
from load_data import get_dataset, save_dataset_df, load_dataset_df
from tqdm import tqdm
import pandas as pd
import click
import os
import subprocess
import re
from ast import literal_eval


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

os.makedirs("results/evals", exist_ok=True)


def parse_score(output):
    response = output.strip().lower()
    # look for the regex matching rating: followed by a number from 1 to 5, allowing for any amount of whitespace in between
    match = re.search(r"rating:\s*([1-5])", response)
    if match:
        return int(match.group(1))  # convert to 0-4 scale
    else:
        log_warn("Could not find 'rating: [1-5]' in model response: " + response)
        return None


def parse_eval(df):
    df["score"] = df["score_output"].apply(parse_score)
    nan_frac = df["score"].isna().mean()
    if nan_frac > 0:
        log_warn(
            f"Parsed scores from evaluation output, but {round(nan_frac * 100, 2)}% of scores are NaN. This can happen if the judge model did not follow the output format correctly."
        )
    if "n_queries" not in df.columns:
        df["n_queries"] = 0
    if "concluded" not in df.columns:
        df["concluded"] = False
    return df


def score_description_predictions(
    *,
    predictions_save_path,
    save_name,
    override_eval=False,
):
    parameters = load_parameters()
    model = parameters["evaluation_model_name"]
    model_save_name = model.split("/")[-1].strip()
    save_name = (
        f"{save_name}_description_prediction_judge-{model_save_name}"
    )
    evaluation_path = os.path.abspath(f"results/evals/" + save_name + ".jsonl")
    skip = False
    if not override_eval:
        if os.path.exists(evaluation_path):
            log_info(
                f"Scored predictions already exist at {evaluation_path}, skipping judge generation."
            )
            skip = True
            df = load_dataset_df(evaluation_path)
    if not skip:
        df = load_dataset_df(predictions_save_path)

        def get_score_prompt(row):
            description = None
            if "description" in row:
                description = row["description"]
            elif "true_description" in row:
                description = row["true_description"]
            else:
                log_error(
                    f"Row with columns: {row.keys()} does not contain a description column."
                )
            prompt_filled = eval_prompt.replace("[TRUE]", description).replace(
                "[HYPOTHESIS]", row["predicted_description"]
            )
            return prompt_filled

        df["score_prompt"] = df.apply(get_score_prompt, axis=1)
        model_name = parameters["evaluation_model_name"]
        model = get_lm(model_name)
        df["score_output"] = None
        for idx, row in tqdm(
            df.iterrows(), total=len(df), desc="Generating evaluation scores"
        ):
            prompt = row["score_prompt"]
            output = model.infer(prompt, max_new_tokens=300)
            df.at[idx, "score_output"] = output
    df = parse_eval(df)
    save_dataset_df(df, evaluation_path)
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


def evaluate_code_predictions(true_code, predicted_code, test_examples):
    can_exec_pred_code = False
    test_inputs = [ex[0] for ex in test_examples]
    true_outputs = [ex[1] for ex in test_examples]
    predicted_outputs = []
    exact_matches = []
    ret = [
        can_exec_pred_code,
        predicted_outputs,
        0.0,
    ]
    runner = RunTestFunc(true_code)
    if predicted_code is None:
        return ret
    try:
        pred_runner = RunTestFunc(predicted_code)
        ret[1] = True
    except Exception as e:
        return ret
    for i, test_input in enumerate(test_inputs):
        true_output = true_outputs[i]
        pred_output, pred_error = pred_runner.run_test_str(test_input)
        predicted_outputs.append(pred_output)
        exact_matches.append(true_output == pred_output)
    rep_pred_outputs = []
    for item in predicted_outputs:
        try:
            rep_pred_outputs.append(repr(item))
        except:
            rep_pred_outputs.append(repr(None))
    ret[1] = rep_pred_outputs
    ret[-1] = sum(exact_matches) / len(exact_matches) if len(exact_matches) > 0 else 0.0
    return ret


def evaluate_output_prediction(true_output, predicted_output):
    return true_output == RunTestFunc.timed_literal_eval(predicted_output)[0]


def evaluate_input_prediction(true_code, target_output, predicted_input):
    try:
        runner = RunTestFunc(true_code)
    except Exception as e:
        log_warn(
            f"Failed to initialize RunTestFunc with true code. Error: {str(e)}. This should never happen."
        )
        return True, None
    pred_output, pred_error = runner.run_test_str(predicted_input)
    if pred_output is None and target_output is not None:
        return False, pred_output
    if pred_output == target_output:
        return True, pred_output
    else:
        return False, pred_output


def score_code(predictions_save_path, override_eval=False):
    df = load_dataset_df(predictions_save_path)
    if "predicted_code" not in df.columns:
        log_error(f"predicted_code not in df with columns: {df.columns}")
    if "true_test_outputs" in df.columns and not override_eval:
        log_info("true_test_outputs column already exists, skipping code scoring.")
        return
    df["true_test_outputs"] = None
    df["predicted_test_outputs"] = None
    df["predicted_outputs_exact_match"] = None
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring code predictions"):
        true_code = row["test_func_validated"]
        predicted_code = row["predicted_code"]
        test_examples = row["test_examples"]
        (
            can_exec_pred_code,
            predicted_outputs,
            exact_match,
        ) = evaluate_code_predictions(true_code, predicted_code, test_examples)
        df.at[idx, "can_exec_pred_code"] = can_exec_pred_code
        df.at[idx, "predicted_test_outputs"] = predicted_outputs
        df.at[idx, "predicted_outputs_exact_match"] = exact_match
    save_dataset_df(df, predictions_save_path)
    log_info(f"Saved scored predictions to {predictions_save_path}")
    log_info(
        f"Average exact match on test outputs: {df['predicted_outputs_exact_match'].mean()} +- {df['predicted_outputs_exact_match'].std()}"
    )


def score_output_prediction(predictions_save_path, override_eval=False):
    df = load_dataset_df(predictions_save_path)
    if "predicted_output" not in df.columns:
        log_error(f"predicted_output not in df with columns: {df.columns}")
    if "output_prediction_correct_micro" in df.columns and not override_eval:
        log_info(
            "output_prediction_correct_micro column already exists, skipping output prediction scoring."
        )
        return
    df["output_prediction_correct_micro"] = None
    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Scoring output predictions"
    ):
        test_func_code = row["test_func_validated"]
        test_examples = row["test_examples"]
        true_output = [ex[1] for ex in test_examples]
        predicted_outputs = row["predicted_output"]
        matches = []
        for idx2, (predicted_output, true_out) in enumerate(
            zip(predicted_outputs, true_output)
        ):
            match = evaluate_output_prediction(
                true_output=true_out, predicted_output=predicted_output
            )
            if match is None:
                match = False
            matches.append(match)
        df.at[idx, "output_prediction_correct_micro"] = (
            sum(matches) / len(matches) if len(matches) > 0 else 0.0
        )
    save_dataset_df(df, predictions_save_path)
    log_info(f"Saved scored predictions to {predictions_save_path}")
    log_info(
        f"Average micro accuracy on output predictions: {df['output_prediction_correct_micro'].mean()} +- {df['output_prediction_correct_micro'].std()}"
    )


def score_input_prediction(predictions_save_path, override_eval=False):
    df = load_dataset_df(predictions_save_path)
    if "predicted_input" not in df.columns:
        log_error(f"predicted_input not in df with columns: {df.columns}")
    if "input_prediction_correct_micro" in df.columns and not override_eval:
        log_info(
            "input_prediction_correct_micro column already exists, skipping input prediction scoring."
        )
        return
    df["input_prediction_correct_micro"] = None
    for idx, row in tqdm(
        df.iterrows(), total=len(df), desc="Scoring input predictions"
    ):
        true_code = row["test_func_validated"]
        target_output = [ex[1] for ex in row["test_examples"]]
        predicted_input = row["predicted_input"]
        if predicted_input is None:
            df.at[idx, "input_prediction_correct_micro"] = 0.0
            continue
        if target_output is None:
            df.at[idx, "input_prediction_correct_micro"] = 1.0
            continue
        matches = []
        for pred_inp, target_out in zip(predicted_input, target_output):
            match, pred_out = evaluate_input_prediction(
                true_code=true_code, target_output=target_out, predicted_input=pred_inp
            )
            if match is None:
                match = False
            matches.append(match)
        df.at[idx, "input_prediction_exact_match_micro"] = (
            sum(matches) / len(matches) if len(matches) > 0 else 0.0
        )
    save_dataset_df(df, predictions_save_path)
    log_info(f"Saved scored predictions to {predictions_save_path}")
    log_info(
        f"Average micro accuracy on input predictions: {df['input_prediction_exact_match_micro'].mean()} +- {df['input_prediction_exact_match_micro'].std()}"
    )


@click.command()
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
    save_name,
    override_eval,
):
    predictions_save_path = os.path.abspath(f"results/predictions/{save_name}.jsonl")
    if not os.path.exists(predictions_save_path):
        log_error(
            f"Predictions file not found at {predictions_save_path}. Run the generation script first."
        )
    score_description_predictions(
        predictions_save_path=predictions_save_path,
        save_name=save_name,
        override_eval=override_eval,
    )


def get_output_save_name(save_name):
    parameters = load_parameters()
    input_output_model = parameters["input_output_prediction_model_name"]
    code_save_name = input_output_model.split("/")[-1].strip()
    if input_output_model == "self":
        code_save_name = "self"
    save_name = f"{save_name}_output_prediction_judge-{code_save_name}"
    return save_name


def get_input_save_name(save_name):
    parameters = load_parameters()
    input_output_model = parameters["input_output_prediction_model_name"]
    code_save_name = input_output_model.split("/")[-1].strip()
    if input_output_model == "self":
        code_save_name = "self"
    save_name = f"{save_name}_input_prediction_judge-{code_save_name}"
    return save_name


def get_code_save_name(save_name):
    parameters = load_parameters()
    code_generation_model = parameters["code_generation_model_name"]
    code_save_name = code_generation_model.split("/")[-1].strip()
    if code_generation_model == "self":
        code_save_name = "self"
    save_name = f"{save_name}_code_prediction_judge-{code_save_name}"
    return save_name


@click.command()
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
    save_name,
    override_eval,
):
    save_name = get_code_save_name(save_name)
    predictions_save_path = os.path.abspath(f"results/predictions/{save_name}.jsonl")
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
    save_name,
    override_eval,
):
    save_name = get_output_save_name(save_name)
    predictions_save_path = os.path.abspath(f"results/predictions/{save_name}.jsonl")
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
    save_name,
    override_eval,
):
    save_name = get_input_save_name(save_name)
    predictions_save_path = os.path.abspath(f"results/predictions/{save_name}.jsonl")
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
