import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import click
from utils import Plotter

figure_df_dir = "results/figure_dfs/"


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
        y_label="Score Spread (%)",
    )

    plot_func()
    plotter.show(save_path=f"{dataset}/description_score_spread.png")


def plot_exact_match(dataset, kind):
    df_path = f"{figure_df_dir}/{dataset}/{kind}_stats.jsonl"
    if not os.path.exists(df_path):
        print(
            f"No {kind} stats found for dataset {dataset} at path {df_path}, skipping plot."
        )
        return
    df = pd.read_json(df_path, lines=True)

    def plot_func():
        # make a bar_plot with avg_exact_match as height and std_exact_match as error bars, with x axis as method and hue as model
        sns.barplot(
            data=df,
            x="Method",
            y="avg_exact_match",
            hue="Model",
        )
        plt.ylabel("Average Exact Match (%)")
        plt.title(f"{kind.capitalize()} Exact Match for {dataset}")

    plot_func()
    plotter.show(save_path=f"{dataset}/{kind}_exact_match.png")

    df["percentage_exact_match_mid"] = (
        100 - df["percentage_exact_match_1"] - df["percentage_exact_match_0"]
    )
    cols = [
        "percentage_exact_match_0_to_20",
        "percentage_exact_match_20_to_40",
        "percentage_exact_match_40_to_60",
        "percentage_exact_match_60_to_80",
        "percentage_exact_match_80_to_100",
    ]

    def rename_exact_match_col(col):
        if col == "percentage_exact_match_0_to_20":
            return "0-19%"
        elif col == "percentage_exact_match_20_to_40":
            return "20-39%"
        elif col == "percentage_exact_match_40_to_60":
            return "40-59%"
        elif col == "percentage_exact_match_60_to_80":
            return "60-79%"
        elif col == "percentage_exact_match_80_to_100":
            return "80-100%"
        else:
            return col

    df.rename(columns=rename_exact_match_col, inplace=True)
    cols = [rename_exact_match_col(col) for col in cols]

    colours = [
        "dodgerblue",
        "orangered",
        "mediumorchid",
        "goldenrod",
        "seagreen",
    ]

    df.sort_values(by=["Model Order", "Method Order"], inplace=True)
    plot_func = plotter.get_stacked_bar_plot_func(
        df=df,
        x_col="Method",
        stacked_cols=cols,
        colours=colours,
        skip_col="Model",
        x_tick_rotation=25,
        skip_text_y_dip=50,
        skip_text_rotation=7,
        y_label="Exact Match Spread (%)",
    )

    plot_func()
    plotter.show(save_path=f"{dataset}/{kind}_exact_match_spread.png")


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
            elif kind in ["code", "input", "output"]:
                plot_exact_match(dataset, kind)
            else:
                print(f"Plotting for kind {kind} not implemented yet, skipping.")


if __name__ == "__main__":
    plot()
