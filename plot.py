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


def rank_correlation(df, metric_col="all_scores"):
    df = df[df["Method"] == "interactive"].reset_index(drop=True)
    df["unique_id"] = df["Model"]
    df["ranks"] = df[metric_col].apply(determine_ranks)
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


def construct_max_score_row(df, score_col="all_scores"):
    # add a new row to df with Model = "Max Score", Method = "N/A", and all_scores being the max score for each of the 741 samples across all models and methods
    """
        (Pdb) df.columns
    Index(['Model', 'Method', 'avg_score', 'std_score', 'Score 1', 'Score 2',
           'Score 3', 'Score 4', 'Score 5', 'percentage_score_3_or_greater',
           'avg_queries', 'std_queries', 'concluded_percentage', 'all_scores',
           'Method Order', 'Model Order', 'x_pos'],
          dtype='object')
        all_scores is a list with all scores, we gotta agg on that
    """
    # get first free index:
    methods = df["Method"].unique()
    for method in methods:
        max_index = df.index.max() + 1
        max_all_scores = None
        for i, row in df.iterrows():
            if row["Method"] != method:
                continue
            all_scores = row[score_col]
            if max_all_scores is None:
                max_all_scores = all_scores
            else:
                max_all_scores = [max(a, b) for a, b in zip(max_all_scores, all_scores)]
        df.loc[max_index] = None
        df.at[max_index, "Model"] = "Max"
        df.at[max_index, "Method"] = method
        df.at[max_index, score_col] = max_all_scores
        # Score 1 is percentage of scores that are 1, Score 2 is percentage of scores that are 2, etc. so we can calculate those from the all_scores list:
        if score_col == "all_scores":
            for score in range(1, 6):
                percentage_score = (
                    sum([1 for s in max_all_scores if s == score])
                    / len(max_all_scores)
                    * 100
                )
                df.at[max_index, f"{score}"] = percentage_score
            df.at[max_index, "avg_score"] = sum(max_all_scores) / len(max_all_scores)
        else:
            # count percentage 0-20%, 20-40%, 40-60%, 60-80%, 80-100% for exact match scores:
            percentage_0_to_20 = (
                sum([1 for s in max_all_scores if s < 0.2]) / len(max_all_scores) * 100
            )
            percentage_20_to_40 = (
                sum([1 for s in max_all_scores if 0.2 <= s < 0.4])
                / len(max_all_scores)
                * 100
            )
            percentage_40_to_60 = (
                sum([1 for s in max_all_scores if 0.4 <= s < 0.6])
                / len(max_all_scores)
                * 100
            )
            percentage_60_to_80 = (
                sum([1 for s in max_all_scores if 0.6 <= s < 0.8])
                / len(max_all_scores)
                * 100
            )
            percentage_80_to_100 = (
                sum([1 for s in max_all_scores if s >= 0.8]) / len(max_all_scores) * 100
            )
            df.at[max_index, "0-19%"] = percentage_0_to_20
            df.at[max_index, "20-39%"] = percentage_20_to_40
            df.at[max_index, "40-59%"] = percentage_40_to_60
            df.at[max_index, "60-79%"] = percentage_60_to_80
            df.at[max_index, "80-100%"] = percentage_80_to_100
            df.at[max_index, "avg_exact_match"] = sum(max_all_scores) / len(
                max_all_scores
            )
    return df


def get_top_k_models_df(df, k=5, metric="avg_score"):
    interactive = df[df["Method"] == "interactive"].reset_index(drop=True)
    top_k = (
        interactive.sort_values(by=metric, ascending=False).head(k)["Model"].tolist()
    )
    return df[df["Model"].isin(top_k)].reset_index(drop=True)


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
                return f"{i}"
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
    top_k_df = get_top_k_models_df(df, k=5, metric="avg_score")
    use_df = top_k_df[top_k_df["Method"].isin(["interactive", "incontext"])]
    plot_func = plotter.get_stacked_bar_plot_func(
        df=use_df,
        x_col="Method",
        stacked_cols=score_cols,
        colours=colours,
        skip_col="Model",
        x_tick_rotation=25,
        skip_text_y_dip=70,
        skip_text_rotation=15,
        y_label="Score Spread (%)",
        tight_layout=False,
    )

    plot_func()
    # plotter.show(save_path=f"description_score_spread")
    print("Save plot with name description_score_spread")
    plt.show()

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
    plotter.show(save_path=f"description_score_spread_interactive")

    def plot_func():
        # venn diagram
        rank_matrix = rank_correlation(top_k_df)
        sns.heatmap(rank_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
        # tilt the x labels:
        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.title("Rank Correlation of Models on Description Scores")
        plt.xlabel("")
        plt.ylabel("")

    plot_func()
    plotter.show(save_path=f"description_score_rank_correlation")

    construct_max_score_row(df)
    interactive = df[df["Method"] == "interactive"].reset_index(drop=True)
    plot_func = plotter.get_stacked_bar_plot_func(
        df=interactive,
        x_col="Model",
        stacked_cols=score_cols,
        colours=colours,
        x_tick_rotation=25,
        y_label="Score Spread (%)",
    )

    plot_func()
    plotter.show(save_path=f"description_score_spread_interactive_w_max")
    # plotter.test_sizes(plot_func)


def plot_exact_match(kind):
    plotter.set_size_parameters(legend_font_size=22)
    df_path = f"{figure_df_dir}/{kind}_stats.jsonl"
    if not os.path.exists(df_path):
        print(f"No {kind} stats found at path {df_path}, skipping plot.")
        return
    df = pd.read_json(df_path, lines=True)

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
    top_k_df = get_top_k_models_df(df, k=5, metric="avg_exact_match")
    # keep only interactive and incontext settings:
    use_df = top_k_df[top_k_df["Method"].isin(["interactive", "incontext"])]
    plot_func = plotter.get_stacked_bar_plot_func(
        df=use_df,
        x_col="Method",
        stacked_cols=cols,
        colours=colours,
        skip_col="Model",
        x_tick_rotation=25,
        skip_text_y_dip=70,
        skip_text_rotation=7,
        y_label="CS Spread (%)",
        tight_layout=False,
    )

    plot_func()
    # plotter.show(save_path=f"{kind}_exact_match_spread")
    print(f"Save plot with name {kind}_exact_match_spread")
    plt.show()

    interactive = df[df["Method"] == "interactive"].reset_index(drop=True)
    interactive.sort_values(by="avg_exact_match", inplace=True)
    plot_func = plotter.get_stacked_bar_plot_func(
        df=interactive,
        x_col="Model",
        stacked_cols=cols,
        colours=colours,
        x_tick_rotation=25,
        y_label="CS Spread (%)",
    )

    plot_func()
    plotter.show(save_path=f"{kind}_exact_match_spread_interactive")

    construct_max_score_row(df, score_col="all_exact_matches")
    interactive = df[df["Method"] == "interactive"].reset_index(drop=True)
    plot_func = plotter.get_stacked_bar_plot_func(
        df=interactive,
        x_col="Model",
        stacked_cols=cols,
        colours=colours,
        x_tick_rotation=25,
        y_label="CS Spread (%)",
    )

    plot_func()
    plotter.show(save_path=f"{kind}_exact_match_spread_interactive_w_max")


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
