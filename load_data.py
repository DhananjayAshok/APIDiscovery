from utils.parameter_handling import load_parameters
from utils import log_error, log_info, log_warn
import click
from datasets import load_dataset
import json
import os
import pandas as pd
from creation import robust_serialize

loaded_parameters = load_parameters()


def get_dataset(split, parameters=None, load_examples=True):
    parameters = load_parameters()
    username = parameters["huggingface_repo_namespace"]
    reponame = parameters["huggingface_repo_name"]
    use_split = "test" if split == "debug" else split
    dset = load_dataset(
        f"{username}/{reponame}", split=use_split
    ).to_pandas()
    if load_examples:
        dset["train_examples"] = dset["train_examples"].apply(json.loads)
        dset["test_examples"] = dset["test_examples"].apply(json.loads)
        dset["all_examples"] = dset["all_examples"].apply(json.loads)
    if split == "debug":
        dset = dset.sample(n=10, random_state=parameters["random_seed"]).reset_index(drop=True)
    return dset


def save_dataset_df(df, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for column in ["train_examples", "test_examples", "all_examples", "predicted_input", "predicted_output"]:
        if column in df.columns:
            df[column] = df[column].apply(robust_serialize)
    df.to_json(save_path, orient="records", lines=True)
    log_info(f"Saved dataset to {save_path}")

def load_dataset_df(path, deserialize_examples=True):
    df = pd.read_json(path, orient="records", lines=True)
    if deserialize_examples:
        for column in ["train_examples", "test_examples", "all_examples", "predicted_input", "predicted_output"]:
            if column in df.columns:
                df[column] = df[column].apply(json.loads)
    return df


@click.command()
@click.option(
    "--train_val_split",
    default=0.8,
    help="The proportion of the dataset to use for training when creating parquet files. The rest will be used for validation.",
    type=float,
)
def load_train_files(train_val_split):
    parameters = load_parameters()
    splits = ["train", "test"]
    for split in splits:
        parquet_path = parameters["data_dir"] + f"/parquets/"
        csv_path = parameters["data_dir"] + f"/csvs/"
        os.makedirs(parquet_path, exist_ok=True)
        os.makedirs(csv_path, exist_ok=True)
        dataset = get_dataset(split, parameters=parameters, load_examples=False)
        dataset["prompt"] = dataset["interactive_starting_prompt"].apply(lambda x: [{"role": "user", "content": x}] if x else None)
        dataset_length = len(dataset)
        # drop rows where prompt is None
        dataset = dataset[dataset["prompt"].notna()].reset_index(drop=True)
        if len(dataset) < dataset_length:
            log_warn(
                f"Dropped {dataset_length - len(dataset)}/{dataset_length} rows with invalid prompts for split {split}",
                parameters=parameters,
            )
        if split == "test":
            dataset.to_parquet(parquet_path + f"test.parquet")
            dataset.to_csv(csv_path + f"test.csv", index=False)
        else:
            dataset = dataset.sample(frac=1, random_state=parameters["random_seed"]).reset_index(drop=True)
            train_size = int(len(dataset) * train_val_split)
            train_dataset = dataset.loc[:train_size].reset_index(drop=True)
            val_dataset = dataset.loc[train_size:].reset_index(drop=True)
            train_dataset.to_parquet(f"{parquet_path}/train.parquet")
            val_dataset.to_parquet(f"{parquet_path}/val.parquet")
            train_dataset.to_csv(f"{csv_path}/train.csv", index=False)
            val_dataset.to_csv(f"{csv_path}/val.csv", index=False)            
        log_info(
            f"Saved {split} split of dataset to parquet at {parquet_path}",
            parameters=parameters,
        )
        log_info(
            f"Saved {split} split of dataset to csv at {csv_path}",
            parameters=parameters,
        )


if __name__ == "__main__":
    load_train_files()
