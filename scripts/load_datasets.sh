source configs/config.env
# error out if huggingface_repo_namespace not set:
if [ -z "$huggingface_repo_namespace" ]; then
  echo "Error: huggingface_repo_namespace is not set. Please set it in configs/config.env"
  exit 1
fi
test_dataset_names=("humaneval" "cruxeval" "mbpp")

train_dataset_names=("code_alpaca" "magic_coder")

for dataset_name in ${test_dataset_names[@]}; do
  destination=$storage_dir/data/parquets/$dataset_name/test.parquet
  mkdir -p $(dirname $destination)
  python << EOD
from datasets import load_dataset
dataset = load_dataset("$huggingface_repo_namespace/APIDiscoveryDataset", "$dataset_name", split="test")
dataset.to_parquet("$destination")
EOD
  echo "Saved $dataset_name test split to $destination"
done


for dataset_name in ${train_dataset_names[@]}; do
  destination=$storage_dir/data/parquets/$dataset_name/
  mkdir -p $(dirname $destination)
  python << EOD
from datasets import load_dataset
import numpy as np

dataset = load_dataset("$huggingface_repo_namespace/APIDiscoveryDataset", "$dataset_name", split="train")
# Shuffle the dataset
dataset = dataset.shuffle(seed=42)
train_size = int(0.8 * len(dataset))
train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, len(dataset)))
train_dataset.to_parquet("$destination/train.parquet")
val_dataset.to_parquet("$destination/val.parquet")
EOD
  echo "Saved $dataset_name train and val split to $destination"
done