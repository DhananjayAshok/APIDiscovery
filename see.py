import pandas as pd
import click
import os
import re
from utils import log_error, load_parameters, log_warn, log_info
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np


method_orders = {"in-context": 0, "ft": 1, "zeroshot": 2, "rl": 3, "gold": 4}

model_orders = {
    "Meta-Llama-3-8B-Instruct": 0,
    "Llama-3.1-8B-Instruct": 0,
    "Qwen3-8B": 1,
    "Qwen3-32B": 2,
    "gpt-4o-mini": 3,
    "gpt-4o": 4,
}

model_scales = {
    "Meta-Llama-3-8B-Instruct": 8,
    "Llama-3.1-8B-Instruct": 8,
    "Qwen3-8B": 8,
    "Qwen3-32B": 32,
}

parameters = load_parameters()


def paired_bootstrap(
    sys1,
    sys2,
    num_samples=10000,
    sample_ratio=0.5,
    progress_title=None,
    parameters=None,
):
    """Evaluate with paired boostrap

    This compares two systems, performing a significance tests with
    paired bootstrap resampling to compare the performance of the two systems.

    :param sys1: The eval metrics (instance-wise) of system 1
    :param sys2: The eval metrics (instance-wise) of system 2. Must be of the same length
    :param num_samples: The number of bootstrap samples to take
    :param sample_ratio: The ratio of samples to take every time
    """
    parameters = load_parameters(parameters)

    sys1_scores = []
    sys2_scores = []
    wins = [0, 0, 0]
    n = len(sys1)
    if len(sys2) != n:
        log_warn(
            "System outputs must be of the same length for paired bootstrap evaluation.",
            parameters,
        )
        return
    ids = list(range(n))

    for _ in tqdm(range(num_samples), desc=progress_title):
        # Subsample the gold and system outputs
        reduced_ids = np.random.choice(ids, int(len(ids) * sample_ratio), replace=True)
        reduced_sys1 = [sys1[i] for i in reduced_ids]
        reduced_sys2 = [sys2[i] for i in reduced_ids]
        # Calculate accuracy on the reduced sample and save stats
        sys1_score = np.mean(reduced_sys1)
        sys2_score = np.mean(reduced_sys2)
        if sys1_score > sys2_score:
            wins[0] += 1
        elif sys1_score < sys2_score:
            wins[1] += 1
        else:
            wins[2] += 1
        sys1_scores.append(sys1_score)
        sys2_scores.append(sys2_score)

    # Print win stats
    wins = [x / float(num_samples) for x in wins]
    if progress_title is not None:
        log_info(f"Results for {progress_title}:", parameters)
    print("Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f" % (wins[0], wins[1], wins[2]))
    if wins[0] > wins[1]:
        print("(sys1 is superior with p value p=%.3f)\n" % (1 - wins[0]))
    elif wins[1] > wins[0]:
        print("(sys2 is superior with p value p=%.3f)\n" % (1 - wins[1]))

    # Print system stats
    sys1_scores.sort()
    sys2_scores.sort()
    print(
        "sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]"
        % (
            np.mean(sys1_scores),
            np.median(sys1_scores),
            sys1_scores[int(num_samples * 0.025)],
            sys1_scores[int(num_samples * 0.975)],
        )
    )
    print(
        "sys2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]"
        % (
            np.mean(sys2_scores),
            np.median(sys2_scores),
            sys2_scores[int(num_samples * 0.025)],
            sys2_scores[int(num_samples * 0.975)],
        )
    )


def comparisons(df, metric_col, col="Method"):
    assert col in ["Method", "Model"]
    not_col = "Model" if col == "Method" else "Method"
    all_of = df[col].unique()
    for col_val in all_of:
        subset = df[df[col] == col_val]
        not_col_vals = subset[not_col].unique()
        for not_col_idx in range(len(not_col_vals)):
            for not_col_idx2 in range(not_col_idx + 1, len(not_col_vals)):
                try:
                    yield (col_val, not_col_vals[not_col_idx]), subset[
                        subset[not_col] == not_col_vals[not_col_idx]
                    ].iloc[0][metric_col], (
                        col_val,
                        not_col_vals[not_col_idx2],
                    ), subset[
                        subset[not_col] == not_col_vals[not_col_idx2]
                    ].iloc[
                        0
                    ][
                        metric_col
                    ]
                except:
                    log_warn(
                        f"Skipping comparison for {col_val} with {not_col_vals[not_col_idx]} and {not_col_vals[not_col_idx2]} due to missing data in metric column."
                    )


def do_test(df, metric_col, progress_title):

    for val1, sys1, val2, sys2 in comparisons(df, metric_col):
        paired_bootstrap(
            sys1,
            sys2,
            progress_title=progress_title
            + f" | Comparing: {val1} vs {val2} for {metric_col}",
        )
    for val1, sys1, val2, sys2 in comparisons(df, metric_col, col="Model"):
        paired_bootstrap(
            sys1,
            sys2,
            progress_title=progress_title
            + f" | Comparing: {val1} vs {val2} for {metric_col}",
        )


def l(row):
    if "description" in row:
        description = row["description"]
    else:
        description = row["true_description"]
    # train_inputs = row['train_inputs']
    predicted_description = row["predicted_description"]
    score_output = row["score_output"]
    score = row["score"]
    n_queries = row["n_queries"]
    concluded = row["concluded"]
    # print(f"Train Inputs: {train_inputs}")
    log_info(f"Description: {description}")
    log_info(f"Number of Queries: {n_queries}")
    log_info(f"Concluded: {concluded}")
    log_info(f"Predicted Description: {predicted_description}")
    log_info(f"Score: {score}")
    return


class Stats:
    @staticmethod
    def require_columns(df, columns):
        existing = df.columns
        for col in columns:
            if col not in existing:
                log_warn(
                    f"Column '{col}' is required but not found in the DataFrame. Existing columns: {existing}"
                )
                return False
        return True

    @staticmethod
    def make_df(dataset, eval_dicts, path_dict):
        if len(eval_dicts) == 0 or len(path_dict) == 0:
            return None
        first_eval_key = list(eval_dicts.keys())[0]
        columns = ["Model", "Method"] + list(eval_dicts[first_eval_key].keys())
        data = []
        for path in eval_dicts:
            if path_dict[path]["dataset"] != dataset:
                continue
            details = path_dict[path]
            row = [details["model"], details["method"]]
            for key in eval_dicts[path]:
                row.append(eval_dicts[path][key])
            data.append(row)
        if len(data) == 0:
            return None
        df = pd.DataFrame(data, columns=columns)
        # sort the dataframe by model order and then by method order
        df["Method Order"] = df["Method"].apply(lambda x: method_orders[x])
        df["Model Order"] = df["Model"].apply(lambda x: model_orders[x])
        df = df.sort_values(by=["Method Order", "Model Order"]).reset_index(drop=True)
        return df

    @staticmethod
    def description(df):
        if "true_description" in df.columns:
            df["description"] = df["true_description"]
        columns = ["score", "concluded", "n_queries", "description"]
        if not Stats.require_columns(df, columns):
            return
        # print the following statistics:
        # 1. Average score +- standard deviation
        # 2. Average number of queries +- standard deviation
        # 3. Percentage of concluded cases
        # 4. Average score and number of queries grouped by conclusion status (concluded vs not concluded)
        # 5. correlation between score, concluded, n_queries, and description length
        # 6. Score distribution
        avg_score = df["score"].mean()
        std_score = df["score"].std()
        avg_queries = df["n_queries"].mean()
        std_queries = df["n_queries"].std()
        concluded_percentage = df["concluded"].mean() * 100
        log_info(f"Average Score: {avg_score:.2f} ± {std_score:.2f}")
        log_info(
            f"Percentage of Cases with Score of 1: {(df['score'] == 1).mean() * 100:.2f}%"
        )
        log_info(
            f"Percentage of Cases with Score of 3 or Greater: {(df['score'] >= 3).mean() * 100:.2f}%"
        )
        log_info(f"Average Number of Queries: {avg_queries:.2f} ± {std_queries:.2f}")
        log_info(f"Percentage of Concluded Cases: {concluded_percentage:.2f}%")
        concluded_group = df.groupby("concluded").agg(
            {"score": ["mean", "std"], "n_queries": ["mean", "std"]}
        )
        # log_info(f"Average Score and Number of Queries Grouped by Conclusion Status:\n{concluded_group}")
        df["description_length"] = df["description"].apply(
            lambda x: len(x.split()) if isinstance(x, str) else 0
        )
        correlation = df[
            ["score", "concluded", "n_queries", "description_length"]
        ].corr()
        # log_info(f"Correlation between Score, Concluded, Number of Queries, and Description Length:\n{correlation}")
        # score_distribution = df["score"].value_counts().sort_index()
        # log_info(f"Score Distribution:\n{score_distribution}")
        return {
            "avg_score": avg_score,
            "std_score": std_score,
            "percentage_score_1": (df["score"] == 1).mean() * 100,
            "percentage_score_2": (df["score"] == 2).mean() * 100,
            "percentage_score_3": (df["score"] == 3).mean() * 100,
            "percentage_score_4": (df["score"] == 4).mean() * 100,
            "percentage_score_5": (df["score"] == 5).mean() * 100,
            "percentage_score_3_or_greater": (df["score"] >= 3).mean() * 100,
            "avg_queries": avg_queries,
            "std_queries": std_queries,
            "concluded_percentage": concluded_percentage,
            "all_scores": df["score"].tolist(),
        }

    def plot_description(dataset, eval_dicts, path_dicts):
        """ """
        os.makedirs(f"results/figure_dfs/{dataset}", exist_ok=True)
        df = Stats.make_df(dataset, eval_dicts, path_dicts)
        if df is None:
            log_warn("No data available to plot.")
            return
        df.to_json(
            f"results/figure_dfs/{dataset}/description_stats.jsonl",
            orient="records",
            lines=True,
        )

    def plot_exact_match(dataset, title, eval_dicts, path_dicts, column):
        os.makedirs(f"results/figures/{dataset}", exist_ok=True)
        df = Stats.make_df(dataset, eval_dicts, path_dicts)
        if df is None:
            log_warn("No data available to plot.")
            return
        df.to_json(
            f"results/figure_dfs/{dataset}/{title}_stats.jsonl",
            orient="records",
            lines=True,
        )

    def plot_code(dataset, eval_dicts, path_dicts):
        Stats.plot_exact_match(
            dataset, "code", eval_dicts, path_dicts, "avg_exact_match"
        )

    def plot_output_prediction(dataset, eval_dicts, path_dicts):
        Stats.plot_exact_match(
            dataset, "output", eval_dicts, path_dicts, "avg_exact_match"
        )

    def plot_input_prediction(dataset, eval_dicts, path_dicts):
        Stats.plot_exact_match(
            dataset, "input", eval_dicts, path_dicts, "avg_exact_match"
        )

    @staticmethod
    def exact_match_metric(df, column):
        columns = [column]
        if not Stats.require_columns(df, columns):
            return
        # print the following statistics:
        # 1. Average exact match score ± standard deviation
        # 2. Exact match score distribution
        # 3. Percentage of cases with exact match score of 1
        # 4. Percentage of cases with exact match score of 0.5 or greater
        # 5. Percentage of cases with exact match score of 0
        avg_exact_match = df[column].mean()
        std_exact_match = df[column].std()
        percentage_exact_match_1 = (df[column] == 1).mean() * 100
        percentage_exact_match_05_or_greater = (df[column] >= 0.5).mean() * 100
        percentage_exact_match_0 = (df[column] == 0).mean() * 100
        log_info(
            f"Average Exact Match Score: {avg_exact_match:.2f} ± {std_exact_match:.2f}"
        )
        # log_info(f"Exact Match Score Distribution:\n{exact_match_distribution}")
        log_info(
            f"Percentage of Cases with Exact Match Score of 1: {percentage_exact_match_1:.2f}%"
        )
        log_info(
            f"Percentage of Cases with Exact Match Score of 0.5 or Greater: {percentage_exact_match_05_or_greater:.2f}%"
        )
        log_info(
            f"Percentage of Cases with Exact Match Score of 0: {percentage_exact_match_0:.2f}%"
        )
        # log_info(df[column].describe())
        return {
            "avg_exact_match": avg_exact_match,
            "std_exact_match": std_exact_match,
            "percentage_exact_match_1": percentage_exact_match_1,
            "percentage_exact_match_05_or_greater": percentage_exact_match_05_or_greater,
            "percentage_exact_match_0": percentage_exact_match_0,
            "all_exact_matches": df[column].tolist(),
        }

    def code(df):
        return Stats.exact_match_metric(df, "predicted_outputs_exact_match")

    def output_prediction(df):
        return Stats.exact_match_metric(df, "output_prediction_correct_micro")

    def input_prediction(df):
        return Stats.exact_match_metric(df, "input_prediction_exact_match_micro")


def is_valid_file(path):
    # description pattern is method_model-dataset-judge-judgemodel.jsonl
    description_judge_model = parameters["evaluation_model_name"].split("/")[-1]
    code_judge_model = parameters["code_generation_model_name"].split("/")[-1]
    input_output_judge_model = parameters["input_output_prediction_model_name"].split(
        "/"
    )[-1]
    if f"description_judge-{description_judge_model}.jsonl" in path:
        return "description"
    if f"code_prediction_judge-{code_judge_model}.jsonl" in path:
        return "code"
    if f"output_prediction_judge-{input_output_judge_model}.jsonl" in path:
        return "output_prediction"
    if f"input_prediction_judge-{input_output_judge_model}.jsonl" in path:
        return "input_prediction"
    #print(path)
    #breakpoint()
    return None


def get_file_details(path):
    methods = ["zeroshot", "ft", "in-context", "rl", "gold"]
    task = is_valid_file(path)
    path = os.path.basename(path)
    if task is None:
        return None
    flag = False
    rest_of = None
    for method in methods:
        if path.startswith(method):
            flag = True
            rest_of = path[len(method) + 1 :]
            break
    if not flag:
        return None
    model_name = rest_of.split("_")[0]
    return {"method": method, "model": model_name, "task": task}


@click.command()
@click.option("--n", default=1, help="Number of random samples to display")
@click.option("--method", default="zeroshot", help="Method to filter by")
@click.option("--dataset", default="humaneval", help="Dataset to filter by")
@click.option("--judge", default="Llama-3.1-8B-Instruct", help="Judge to filter by")
@click.option("--model", default="Meta-Llama-3-8B-Instruct", help="Model to filter by")
def d(n, method, dataset, judge, model):
    path = f"results/{dataset}/{method}_{model}-{dataset}-judge-{judge}.jsonl"
    df = pd.read_json(path, lines=True)
    for i in range(n):
        row = df.sample(1).iloc[0]
        l(row)
        print("\n\n XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n\n")


@click.command()
@click.option(
    "--kind",
    default=None,
    help="Kind of statistics to compute (description, code, output_prediction, input_prediction, or None)",
)
@click.option(
    "--method",
    default=None,
    help="Method to filter by. Will filter by prefix in filename",
)
@click.option("--dataset", default=None, help="Dataset to filter by.")
@click.option("--model", default=None, help="Model to filter by.")
def stats_all(kind, method, dataset, model):
    path = f"results/"
    figure_path = f"results/figures/"
    valid_stats = {
        "description": [],
        "code": [],
        "output_prediction": [],
        "input_prediction": [],
    }
    path_mapper = {}
    df_mapper = {}
    if kind == None:
        allowed_kinds = set(valid_stats.keys())
    else:
        allowed_kinds = {kind}
    found_datasets = []
    for use_dataset in os.listdir(path):
        options = os.path.join(path, use_dataset)
        for file in os.listdir(options):
            file_path = os.path.join(options, file)
            if method is not None and method not in file_path:
                continue
            if dataset is not None and dataset not in file_path:
                continue
            if model is not None and model not in file_path:
                continue
            # just ensure the model isn't the judge part
            if "judge" in file_path:
                nonjudge_part = file_path.split("judge")[0]
                if model is not None and model not in nonjudge_part:
                    continue

            stat_type = is_valid_file(file)
            if stat_type and stat_type in allowed_kinds:
                valid_stats[stat_type].append(file_path)
                path_mapper[file_path] = get_file_details(file_path)
                path_mapper[file_path]["dataset"] = use_dataset
                if use_dataset not in found_datasets:
                    found_datasets.append(use_dataset)
            else:
                # log_warn(f"Skipping invalid file: {file_path}")
                pass

    for stat_type, files in valid_stats.items():
        log_info(f"Processing {len(files)} files for stat type: {stat_type}")
        df_mapper = {}
        for file in files:
            log_info(f"Processing file: {file}")
            df = pd.read_json(file, lines=True)
            if stat_type == "description":
                stats = Stats.description(df)
            elif stat_type == "code":
                stats = Stats.code(df)
            elif stat_type == "output_prediction":
                stats = Stats.output_prediction(df)
            elif stat_type == "input_prediction":
                stats = Stats.input_prediction(df)
            else:
                log_warn(f"Unknown stat type: {stat_type} for file: {file}")
                continue
            if stats is not None:
                df_mapper[file] = stats
        for dataset in found_datasets:
            if stat_type == "description":
                Stats.plot_description(dataset, df_mapper, path_mapper)
            if stat_type == "code":
                Stats.plot_code(dataset, df_mapper, path_mapper)
            elif stat_type == "output_prediction":
                Stats.plot_output_prediction(dataset, df_mapper, path_mapper)
            elif stat_type == "input_prediction":
                Stats.plot_input_prediction(dataset, df_mapper, path_mapper)


@click.group()
def cli():
    pass


cli.add_command(d, name="show")
cli.add_command(stats_all, name="stats")

if __name__ == "__main__":
    cli()
