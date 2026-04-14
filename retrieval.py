import os
import torch
import torch.nn.functional as F
import click
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from utils import load_parameters, log_info, log_error
from load_data import get_dataset, load_dataset_df, save_dataset_df
from baselines import get_save_paths
from tqdm import tqdm

BATCH_SIZE = 32


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    return last_hidden_states[:, -1]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


class Qwen3Embedding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left')
        self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B', device_map="auto").eval()

    def embed(self, texts):
        max_length = 1024
        all_embeddings = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
            batch = texts[i:i + BATCH_SIZE]
            batch_dict = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict = batch_dict.to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)


@click.command()
@click.option("--override_gen", is_flag=True, help="Whether to override existing embeddings.")
def embed_train(override_gen):
    parameters = load_parameters()
    save_path = os.path.join(parameters["data_dir"], "embedding", "train.pt")
    if os.path.exists(save_path) and not override_gen:
        log_info(f"Train embeddings already exist at {save_path}, skipping.")
        return
    df = get_dataset("train", parameters=parameters)
    embedding_model = Qwen3Embedding()
    texts = df["description"].tolist()
    log_info(f"Embedding {len(texts)} train descriptions...")
    embeddings = embedding_model.embed(texts)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(embeddings, save_path)
    log_info(f"Saved train embeddings to {save_path}")


@click.command()
@click.option("--load_name", type=str, required=True, help="Name of the predictions file to load.")
@click.option("--k", type=int, default=1, help="Number of top similar training examples to retrieve.")
@click.option("--override_gen", is_flag=True, help="Whether to override existing retrieval.")
def retrieve(load_name, k, override_gen):
    parameters = load_parameters()
    load_path = get_save_paths(load_name, parameters)
    if not os.path.exists(load_path):
        log_error(f"{load_path} not found")
    df = load_dataset_df(load_path)
    if "retrieved_train_indices" in df.columns and not override_gen:
        log_info(f"Retrieved indices already present in {load_path}, skipping. Run with --override_gen to re-run.")
        return
    embedding_model = Qwen3Embedding()
    texts = df["predicted_description"].tolist()
    log_info(f"Embedding {len(texts)} predicted descriptions...")
    pred_embeddings = embedding_model.embed(texts)
    train_embedding_path = os.path.join(parameters["data_dir"], "embedding", "train.pt")
    train_embeddings = torch.load(train_embedding_path)
    scores = pred_embeddings @ train_embeddings.T
    topk = torch.topk(scores, k=k, dim=1)
    df["retrieved_train_indices"] = topk.indices.tolist()
    save_dataset_df(df, load_path)
    log_info(f"Saved retrieved indices to {load_path}")


@click.group()
def cli():
    pass


cli.add_command(embed_train, name="embed-train")
cli.add_command(retrieve, name="retrieve")


if __name__ == "__main__":
    cli()
