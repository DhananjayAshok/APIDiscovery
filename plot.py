import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import click
from utils import Plotter, load_parameters

parameters = load_parameters()


figure_df_dir = f"{parameters['results_dir']}/figure_dfs/"

plotter = Plotter()


def determine_ranks(all_scores_list):
    # list of scores
    # determine rank of each score in the list, with 1 being the highest rank, and ties getting the same average rank
    sorted_scores = sorted(all_scores_list, reverse=True)
    ranks = []
    for score in all_scores_list:
        rank = sorted_scores.index(score) + 1
        ranks.append(rank)
    return ranks


def rank_correlation(df):
    df = df[df["Method"] == "interactive"].reset_index(drop=True)
    df["unique_id"] = df["Model"]
    df["ranks"] = df["all_scores"].apply(determine_ranks)
    unique_ids = df["unique_id"].unique()
    rank_matrix = []
    for uid_1 in unique_ids:
        row = []
        for uid_2 in unique_ids:
            ranks_1 = df[df["unique_id"] == uid_1]["ranks"].values[0]
            ranks_2 = df[df["unique_id"] == uid_2]["ranks"].values[0]
            correlation = pd.Series(ranks_1).corr(pd.Series(ranks_2), method="spearman")
            row.append(correlation)
        rank_matrix.append(row)
    rank_matrix = pd.DataFrame(rank_matrix, index=unique_ids, columns=unique_ids)
    return rank_matrix


def get_score_df(df, score_col="all_scores"):
    columns = ["Model", "Method"] + [f"score_{i}" for i in range(0, 741)]
    data = []
    for i, row in df.iterrows():
        model = row["Model"]
        method = row["Method"]
        all_scores = row[score_col]
        if len(all_scores) != 741:
            print(
                f"Warning: row {i} has {len(all_scores)} scores instead of 741, skipping."
            )
            continue
        data.append([model, method] + all_scores)
    score_df = pd.DataFrame(data, columns=columns)
    return score_df


def score_statistics(df, score_col="all_scores"):
    score_columns = [f"score_{i}" for i in range(0, 741)]
    score_df = get_score_df(df, score_col)
    mins = score_df[score_columns].min()
    maxs = score_df[score_columns].max()
    means = score_df[score_columns].mean()
    stds = score_df[score_columns].std()
    return mins, maxs, means, stds


def plot_description():
    df_path = f"{figure_df_dir}/description_stats.jsonl"
    if not os.path.exists(df_path):
        print(f"No description stats found at path {df_path}, skipping plot.")
        return
    df = pd.read_json(df_path, lines=True)
    df.sort_values(by=["Model Order", "Method Order"], inplace=True)
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
    plot_func = plotter.get_stacked_bar_plot_func(
        df=df,
        x_col="Method",
        stacked_cols=score_cols,
        colours=colours,
        skip_col="Model",
        x_tick_rotation=25,
        skip_text_y_dip=60,
        skip_text_rotation=15,
        y_label="Score Spread (%)",
    )

    plot_func()
    plotter.show(save_path=f"description_score_spread.png")

    interactive = df[df["Method"] == "interactive"].reset_index(drop=True)
    # sort by the avg_score column:
    interactive.sort_values(by="avg_score", inplace=True)
    plot_func = plotter.get_stacked_bar_plot_func(
        df=interactive,
        x_col="Model",
        stacked_cols=score_cols,
        colours=colours,
        x_tick_rotation=25,
        y_label="Score Spread (%)",
    )

    plot_func()
    plotter.show(save_path=f"description_score_spread_interactive.png")

    def plot_func():
        # venn diagram
        rank_matrix = rank_correlation(df)
        sns.heatmap(rank_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        plt.title("Rank Correlation of Models on Description Scores")
        plt.xlabel("Model")
        plt.ylabel("Model")

    plot_func()
    plotter.show(save_path=f"description_score_rank_correlation.png")

    def plot_func():
        mins, maxs, means, stds = score_statistics(df)
        # bar plot with means as height and stds as error bars:
        sns.barplot(
            x=means.index,
            y=maxs.values,
        )

        plt.xticks(rotation=90)
        plt.ylabel("Average Score (%)")
        plt.title("Average Description Scores with Standard Deviation")

    plot_func()
    plotter.show(save_path=f"description_score_statistics.png")


def plot_exact_match(kind):
    df_path = f"{figure_df_dir}/{kind}_stats.jsonl"
    if not os.path.exists(df_path):
        print(f"No {kind} stats found at path {df_path}, skipping plot.")
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
        plt.title(f"{kind.capitalize()} Exact Match")

    plot_func()
    plotter.show(save_path=f"{kind}_exact_match.png")

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
    plotter.show(save_path=f"{kind}_exact_match_spread.png")


@click.command()
@click.option(
    "--kind",
    type=str,
    default=None,
    help="metric category to plot (description, code_task, code_eval, input, output)",
)
def plot(kind):
    kinds = [kind] if kind else ["description", "code_task", "code_eval"]
    for kind in kinds:
        if kind == "description":
            plot_description()
        elif kind in ["code_task", "code_eval", "input", "output"]:
            plot_exact_match(kind)
        else:
            print(f"Plotting for kind {kind} not implemented yet, skipping.")


if __name__ == "__main__":
    plot()
