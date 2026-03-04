import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import click
from utils import Plotter

figure_df_dir = "results/figure_dfs/"
figure_dir = "results/figures/"


plotter = Plotter()


def plot_description(dataset):
    df_path = f"{figure_df_dir}/{dataset}/description_stats.jsonl"
    if not os.path.exists(df_path):
        print(
            f"No description stats found for dataset {dataset} at path {df_path}, skipping plot."
        )
        return
    df = pd.read_json(df_path, lines=True)
    score_cols = [
        "percentage_score_1",
        "percentage_score_2",
        "percentage_score_3",
        "percentage_score_4",
        "percentage_score_5",
    ]

    # rename the score columns to be Score 1, Score 2, etc for better legend labels
    def rename_score_col(col):
        for i in range(1, 6):
            if f"percentage_score_{i}" == col:
                return f"Score {i}"
        return col

    df.rename(columns=rename_score_col, inplace=True)
    colours = [
        "dodgerblue",
        "orangered",
        "mediumorchid",
        "goldenrod",
        "seagreen",
    ]
    score_cols = [rename_score_col(col) for col in score_cols]
    df.sort_values(by=["Model Order", "Method Order"], inplace=True)
    plot_func = plotter.get_stacked_bar_plot_func(
        df=df,
        x_col="Method",
        stacked_cols=score_cols,
        colours=colours,
        skip_col="Model",
        x_tick_rotation=25,
        skip_text_y_dip=50,
        skip_text_rotation=7,
        y_label="Score Percentage (%)",
    )

    plot_func()
    plt.show()


@click.command()
@click.option("--dataset", type=str, default=None, help="Name of the dataset to plot.")
@click.option(
    "--kind",
    type=str,
    default=None,
    help="metric category to plot (description, code, input, output)",
)
def plot(dataset, kind):
    datasets = [dataset] if dataset else os.listdir(figure_df_dir)
    kinds = [kind] if kind else ["description", "code", "input", "output"]
    for dataset in datasets:
        for kind in kinds:
            if kind == "description":
                plot_description(dataset)
            else:
                print(f"Plotting for kind {kind} not implemented yet, skipping.")


if __name__ == "__main__":
    plot()
