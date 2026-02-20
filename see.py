import pandas as pd
import click
import os
import re
from utils import log_error, load_parameters, log_warn, log_info

parameters = load_parameters()

def l(row):
    if 'description' in row:
        description = row['description']
    else:
        description = row['true_description']
    #train_inputs = row['train_inputs']
    predicted_description = row['predicted_description']
    score_output = row['score_output']
    score = row['score']
    n_queries = row['n_queries']
    concluded = row['concluded']
    #print(f"Train Inputs: {train_inputs}")
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
                raise log_error(f"Column '{col}' is required but not found in the DataFrame. Existing columns: {existing}")
            
    @staticmethod
    def description(df):
        if 'true_description' in df.columns:
            df["description"] = df["true_description"]
        columns = ["score", "concluded", "n_queries", "description"]
        Stats.require_columns(df, columns)
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
        log_info(f"Percentage of Cases with Score of 1: {(df['score'] == 1).mean() * 100:.2f}%")
        log_info(f"Percentage of Cases with Score of 3 or Greater: {(df['score'] >= 3).mean() * 100:.2f}%")
        log_info(f"Average Number of Queries: {avg_queries:.2f} ± {std_queries:.2f}")
        log_info(f"Percentage of Concluded Cases: {concluded_percentage:.2f}%")
        concluded_group = df.groupby("concluded").agg({"score": ["mean", "std"], "n_queries": ["mean", "std"]})
        #log_info(f"Average Score and Number of Queries Grouped by Conclusion Status:\n{concluded_group}")
        df["description_length"] = df["description"].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
        correlation = df[["score", "concluded", "n_queries", "description_length"]].corr()
        #log_info(f"Correlation between Score, Concluded, Number of Queries, and Description Length:\n{correlation}")
        #score_distribution = df["score"].value_counts().sort_index()
        #log_info(f"Score Distribution:\n{score_distribution}")

    @staticmethod
    def exact_match_metric(df, column):
        columns = [column]
        Stats.require_columns(df, columns)
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
        log_info(f"Average Exact Match Score: {avg_exact_match:.2f} ± {std_exact_match:.2f}")
        #log_info(f"Exact Match Score Distribution:\n{exact_match_distribution}")
        log_info(f"Percentage of Cases with Exact Match Score of 1: {percentage_exact_match_1:.2f}%")
        log_info(f"Percentage of Cases with Exact Match Score of 0.5 or Greater: {percentage_exact_match_05_or_greater:.2f}%")
        log_info(f"Percentage of Cases with Exact Match Score of 0: {percentage_exact_match_0:.2f}%")
        #log_info(df[column].describe())

    def code(df):
        return Stats.exact_match_metric(df, "predicted_outputs_exact_match")
    
    def output_prediction(df):
        return Stats.exact_match_metric(df, "output_prediction_correct_micro")
    

    def input_prediction(df):
        return Stats.exact_match_metric(df, "input_prediction_exact_match_micro")

        

def is_valid_file(path):
    # description pattern is method_model-dataset-judge-judgemodel.jsonl
    if "-judge-" in path and path.endswith(".jsonl"):
        return "description"
    if path.endswith("code_prediction.jsonl"):
        return "code"
    if path.endswith("output_prediction.jsonl"):
        return "output_prediction"
    if path.endswith("input_prediction.jsonl"):
        return "input_prediction"
    return None



@click.command()
@click.option('--n', default=1, help='Number of random samples to display')
@click.option('--method', default="zeroshot", help='Method to filter by')
@click.option('--dataset', default="humaneval", help='Dataset to filter by')
@click.option('--judge', default="Llama-3.1-8B-Instruct", help='Judge to filter by')
@click.option('--model', default="Meta-Llama-3-8B-Instruct", help='Model to filter by')
def d(n, method, dataset, judge, model):
    path = f"results/{dataset}/{method}_{model}-{dataset}-judge-{judge}.jsonl"
    df = pd.read_json(path, lines=True)
    for i in range(n):
        row = df.sample(1).iloc[0]
        l(row)
        print("\n\n XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX \n\n")


@click.command()
@click.option("--kind", default=None, help="Kind of statistics to compute (description, code, output_prediction, input_prediction, or None)")
@click.option('--method', default=None, help='Method to filter by. Will filter by prefix in filename')
@click.option('--dataset', default=None, help='Dataset to filter by.')
@click.option('--model', default=None, help='Model to filter by.')
def stats_all(kind, method, dataset, model):
    path = f"results/"
    valid_stats = {"description": [], "code": [], "output_prediction": [], "input_prediction": []}
    if kind == None:
        allowed_kinds = set(valid_stats.keys())
    else:
        allowed_kinds = {kind}
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
            else:
                #log_warn(f"Skipping invalid file: {file_path}")
                pass
    for stat_type, files in valid_stats.items():
        log_info(f"Processing {len(files)} files for stat type: {stat_type}")
        for file in files:
            log_info(f"Processing file: {file}")
            df = pd.read_json(file, lines=True)
            if stat_type == "description":
                Stats.description(df)
            elif stat_type == "code":
                Stats.code(df)
            elif stat_type == "output_prediction":
                Stats.output_prediction(df)
            elif stat_type == "input_prediction":
                Stats.input_prediction(df)
            else:
                log_warn(f"Unknown stat type: {stat_type} for file: {file}")



@click.group()
def cli():
    pass

cli.add_command(d, name="show")
cli.add_command(stats_all, name="stats")

if __name__ == "__main__":
    cli()


