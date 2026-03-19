from utils.parameter_handling import load_parameters
from utils import log_error, log_info, log_warn
import click
from datasets import load_dataset

loaded_parameters = load_parameters()


def get_dataset(split, parameters=None):
    parameters = load_parameters()
    username = parameters["huggingface_repo_namespace"]
    reponame = parameters["huggingface_repo_name"]
    dset = load_dataset(
        f"{username}/{reponame}", split=split
    ).to_pandas()
    dset["train_examples"] = dset["train_examples"].apply(list)
    dset["test_examples"] = dset["test_examples"].apply(list)
    return dset



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

        dataset = dataset.map(
            lambda x: {"prompt": [{"role": "user", "content": x["interactive_starting_prompt"]}]}
        )
        dataset_length = len(dataset)
        # drop rows where prompt is None
        dataset = dataset.filter(lambda x: x["prompt"][0]["content"] is not None)
        if len(dataset) < dataset_length:
            log_warn(
                f"Dropped {dataset_length - len(dataset)}/{dataset_length} rows with invalid prompts for split {split}",
                parameters=parameters,
            )
        if split == "test":
            dataset.to_parquet(parquet_path + f"test.parquet")
            dataset.to_csv(csv_path + f"test.csv", index=False)
        else:
            dataset.shuffle(seed=parameters["random_seed"])
            train_size = int(len(dataset) * train_val_split)
            train_dataset = dataset.select(range(train_size))
            val_dataset = dataset.select(range(train_size, len(dataset)))
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
