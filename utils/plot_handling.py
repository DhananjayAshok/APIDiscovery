import seaborn as sns
from utils.parameter_handling import load_parameters
from utils.log_handling import log_error, log_info, log_warn, log_dict
import matplotlib.pyplot as plt
import os


class Plotter:
    COLOURS = ["dodgerblue", "orangered", "mediumorchid"]

    def __init__(self, parameters=None):
        self.parameters = load_parameters(parameters)
        self.size_params = {}
        sns.set_style("whitegrid")
        self.default_plt_params = plt.rcParams.copy()
        self.set_size_parameters()

    def set_size_parameters(
        self,
        scaler=1,
        font_size=16,
        labels_font_size=19,
        xtick_font_size=19,
        ytick_font_size=15,
        legend_font_size=16,
        title_font_size=20,
    ):
        if font_size is None:
            font_size = self.default_plt_params["font.size"]
        plt.rcParams.update({"font.size": font_size * scaler})
        if labels_font_size is None:
            labels_font_size = self.default_plt_params["axes.labelsize"]
        plt.rcParams.update({"axes.labelsize": labels_font_size * scaler})
        if xtick_font_size is None:
            xtick_font_size = self.default_plt_params["xtick.labelsize"]
        plt.rcParams.update({"xtick.labelsize": xtick_font_size * scaler})
        if ytick_font_size is None:
            ytick_font_size = self.default_plt_params["ytick.labelsize"]
        plt.rcParams.update({"ytick.labelsize": ytick_font_size * scaler})
        if title_font_size is None:
            title_font_size = self.default_plt_params["axes.titlesize"]
        plt.rcParams.update({"axes.titlesize": title_font_size * scaler})
        self.size_params["font_size"] = font_size * scaler
        self.size_params["labels_font_size"] = labels_font_size * scaler
        self.size_params["xtick_font_size"] = xtick_font_size * scaler
        self.size_params["ytick_font_size"] = ytick_font_size * scaler
        self.size_params["title_font_size"] = title_font_size * scaler
        self.size_params["legend_font_size"] = legend_font_size * scaler
        return

    def set_size_default(self, scaler=1):
        self.set_size_parameters(
            scaler=scaler,
            font_size=None,
            labels_font_size=None,
            xtick_font_size=None,
            ytick_font_size=None,
            legend_font_size=None,
            title_font_size=None,
        )
        return

    def set_size_parameters_from_dict(self, size_params):
        """
        Set the size parameters from a dictionary. This trusts that the dictionary is correct and does not check for errors.
        """
        self.set_size_parameters(**size_params)
        return

    def get_size_input_number(self, key_name):
        while True:
            got = input(
                f"Enter the size for {key_name} (current value is {self.size_params[key_name]}, hit enter to keep current value ):"
            )
            got = got.strip()
            if got.strip() == "":
                return self.size_params[key_name]
            try:
                got = float(got)
                if got <= 0:
                    log_warn(
                        f"Got {got}, but it must be greater than 0",
                        parameters=self.parameters,
                    )
                    continue
                return got
            except ValueError:
                log_warn(
                    f"Got {got}, but it must be a number", parameters=self.parameters
                )
                continue

    def test_sizes(self, plot_func):
        """
        Test diff sizes for plot parameters.

        Args:
            plot_func: A function that creates a plot. No arguments.
        """
        done = False
        while not done:
            log_info(f"Plot with sizes: ", parameters=self.size_params)
            log_dict(self.size_params, n_indents=1, parameters=self.parameters)
            plot_func()
            plt.show()
            keepgoing = input(
                "Do you want to keep trying different sizes? (only y will keep going):"
            )
            if keepgoing.lower().strip() == "y":
                for key in self.size_params:
                    self.size_params[key] = self.get_size_input_number(key)
                self.set_size_parameters_from_dict(self.size_params)
            else:
                done = True
                break
        return

    def show(self, save_path=None):
        if save_path is not None:
            if os.path.abspath(save_path).startswith(
                os.path.abspath(self.parameters["figure_dir"])
            ):
                figure_path = save_path
            else:
                figure_path = self.parameters["figure_dir"] + f"/{save_path}"
        else:
            figure_path = None
        if figure_path is not None:
            figure_dir = os.path.dirname(figure_path)
            if not os.path.exists(figure_dir):
                os.makedirs(figure_dir)
            plt.savefig(f"{figure_path}.pdf")
            plt.savefig(f"{figure_path}.png")
            log_info(f"Saved figure to {figure_path}.pdf", parameters=self.parameters)
            if not self.parameters["figure_skip_show"]:
                plt.show()
            else:
                plt.clf()
        else:
            plt.show()
        return

    def get_stacked_bar_plot_func(
        self,
        df,
        x_col,
        stacked_cols,
        colours,
        skip_col=None,
        skip_text_y_dip=15,
        skip_text_rotation=20,
        x_tick_rotation=45,
        y_label="Percentage (%)",
    ):
        """
        Returns a function to create a stacked percentage bar plot with optional skips and grouping based on columns in the dataframe.
        df rows must be sorted in [skip_col order, x_col order] for the plot to look correct.
        """
        parameters = self.parameters
        # error out if any of the stacked_cols are not in the dataframe, if the length of colours is not the same as the length of stacked_cols, or if x_col / skip_col is not in the dataframe
        for col in (
            stacked_cols + [x_col] + ([skip_col] if skip_col is not None else [])
        ):
            if col not in df.columns:
                log_error(
                    f"Column {col} is not in the dataframe with columns {list(df.columns)}. Please check your input.",
                    parameters=parameters,
                )
        if len(colours) != len(stacked_cols):
            log_error(
                f"Length of colours {len(colours)} does not match length of stacked_cols {len(stacked_cols)}. Please check your input.",
                parameters=parameters,
            )

        def plot_func():
            fig, ax = plt.subplots(figsize=(12, 6))

            x_positions = []
            current_x = 0
            last_skip = None

            for i, row in df.iterrows():
                # Add a larger gap if the Method changes
                if skip_col is not None:
                    if last_skip is not None and row[skip_col] != last_skip:
                        current_x += 1.0  # The "Small Whitespace" between methods

                x_positions.append(current_x)

                # 3. Plotting the Stack
                bottom = 0
                for col, color in zip(stacked_cols, colours):
                    val = row[col]
                    ax.bar(
                        current_x,
                        val,
                        bottom=bottom,
                        color=color,
                        edgecolor="white",
                        width=0.8,
                    )
                    bottom += val
                if skip_col is not None:
                    last_skip = row[skip_col]
                current_x += 1  # Distance between x_cols within a skip_col

            # 4. Styling the X-Axis
            # ax.xaxis.tick_top()  # Moves ticks to the top
            # ax.xaxis.set_label_position("top")
            ax.set_xticks(x_positions)
            # Combine Model and Method for labels, or just Model
            ax.set_xticklabels(df[x_col], rotation=x_tick_rotation, ha="right")

            # Optional: Add Method labels below the models
            if skip_col is not None:
                df["x_pos"] = x_positions
                for method, group in df.groupby(skip_col, sort=False):
                    mid_point = group["x_pos"].mean()
                    ax.text(
                        mid_point,
                        -skip_text_y_dip,
                        method,
                        ha="center",
                        fontweight="bold",
                        rotation=skip_text_rotation,
                        fontsize=self.size_params["labels_font_size"],
                    )

            ax.set_ylabel(y_label)
            ax.legend(
                stacked_cols,
                loc="lower left",
                bbox_to_anchor=(0, 1, 1, 0.1),  # (x, y, width, height)
                mode="expand",
                ncol=len(stacked_cols),
                borderaxespad=0,
                frameon=False,
                fontsize=self.size_params["legend_font_size"],
            )
            plt.tight_layout(rect=[0, 0, 1, 0.95])

        return plot_func
