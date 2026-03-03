from utils.parameter_handling import load_parameters, compute_secondary_parameters
from utils import log_error, log_info, log_warn
from utils.lm_inference import HuggingFaceModel
import click
from datasets import load_dataset, Dataset
from tqdm import tqdm
import pandas as pd
import os
import subprocess

loaded_parameters = load_parameters()


from creation import TRAIN_DATASETS, TEST_DATASETS


def get_dataset(dataset_name, parameters=None):
    parameters = load_parameters()
    username = parameters["huggingface_repo_namespace"]
    dset = load_dataset(
        f"{username}/APIDiscoveryDataset", dataset_name, split="test"
    ).to_pandas()
    dset["train_inputs"] = dset["train_inputs"].apply(list)
    dset["test_inputs"] = dset["test_inputs"].apply(list)
    return dset


@click.command()
@click.pass_obj
def get_final(parameters):
    for split, options in [("test", TEST_DATASETS), ("train", TRAIN_DATASETS)]:
        for dataset_name in options + ["all"]:
            try:
                dataset = load_dataset(
                    parameters["huggingface_repo_namespace"] + "/APIDiscoveryDataset",
                    dataset_name,
                    split=split,
                )
            except Exception as e:
                if dataset_name == "all":
                    log_warn(
                        f"Tried to get all, maybe you haven't run merge_final yet? Skipping...",
                        parameters=parameters,
                    )
                    continue
                else:
                    log_error(
                        f"Could not load dataset {dataset_name} split {split} from Hugging Face Hub. Make sure you have run the previous steps to process and push the dataset to the hub.\n{e}",
                        parameters=parameters,
                    )
            df = dataset.to_pandas()
            df.to_json(
                f"{parameters['data_dir']}/final/{dataset_name}/{split}_filtered.jsonl",
                orient="records",
                lines=True,
            )
            log_info(
                f"Saved final dataset {dataset_name} split {split} to {parameters['data_dir']}/final/{dataset_name}/{split}_filtered.jsonl",
                parameters=parameters,
            )
            df.to_csv(
                f"{parameters['data_dir']}/final/{dataset_name}/{split}_filtered.csv",
                index=False,
            )
            log_info(
                f"Saved final dataset {dataset_name} split {split} to {parameters['data_dir']}/final/{dataset_name}/{split}_filtered.csv",
                parameters=parameters,
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
            lambda x: {"prompt": [{"role": "user", "content": x["zero_shot_prompt"]}]}
        )
        dataset_length = len(dataset)
        # drop rows where prompt is None
        dataset = dataset.filter(lambda x: x["prompt"][0]["content"] is not None)
        if len(dataset) < dataset_length:
            log_warn(
                f"Dropped {dataset_length - len(dataset)}/{dataset_length} rows with invalid prompts for {dataset_name} split {split}",
                parameters=parameters,
            )
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


main.add_command(get_final, name="get_final")
main.add_command(load_parquets, name="load_parquets")

if __name__ == "__main__":
    main()
