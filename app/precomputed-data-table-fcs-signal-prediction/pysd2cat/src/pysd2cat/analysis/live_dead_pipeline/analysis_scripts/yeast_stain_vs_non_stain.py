import os
import sys
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pysd2cat.analysis.live_dead_pipeline.ld_pipeline_classes import LiveDeadPipeline
from pysd2cat.analysis.live_dead_pipeline.names import Names as n

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)

strains = [n.yeast, n.ecoli, n.bacillus]


def overlaid_time_series_plot(concatenated_ratio_df, treatment="ethanol", style_col="was stain used?",
                              style_order=None, font_scale=1.0, title=None, tight=True):
    """
    Takes in a dataframe that has been labeled and generates a time-series plot of
    percent alive vs. time, colored by treatment amount.
    This serves as a qualitative metric that allows us to compare different methods of labeling live/dead.
    """
    matplotlib.use("tkagg")
    sns.set(style="ticks", font_scale=font_scale, rc={"lines.linewidth": 3.0})
    num_colors = len(concatenated_ratio_df[treatment].unique())
    palette = sns.color_palette("bright", num_colors)

    lp = sns.lineplot(x=concatenated_ratio_df[n.time], y=concatenated_ratio_df[n.percent_live],
                      hue=concatenated_ratio_df[treatment], style=concatenated_ratio_df[style_col],
                      style_order=style_order, palette=palette, legend="full")
    plt.legend(bbox_to_anchor=(1.01, 0.7), loc=2, borderaxespad=0.)
    plt.ylim(0, 1)
    lp.set_xticks(range(13))
    lp.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    if title is None:
        plt.title("Predicted Live over Time, with and without Stain Features")
    else:
        plt.title(title)
    if tight:
        plt.tight_layout()
    plt.show()


def main():
    run_models = False

    if run_models:
        # stain model
        ldp_stain = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=1,
                                     y_strain=None, y_treatment=None, y_stain=None)
        ldp_stain.load_data()
        print(ldp_stain.feature_cols, "\n")
        ldp_stain.condition_method(live_conditions=None,
                                   dead_conditions=[{n.ethanol: 1120.0, n.time: n.timepoints[-1]},
                                                    {n.ethanol: 280.0, n.time: n.timepoints[-1]}])
        ldp_stain.evaluate_performance(n.condition_method)

        # non-stain model
        ldp_no_stain = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=0,
                                        y_strain=None, y_treatment=None, y_stain=None)
        ldp_no_stain.load_data()
        print(ldp_no_stain.feature_cols, "\n")
        ldp_no_stain.condition_method(live_conditions=None,
                                      dead_conditions=[{n.ethanol: 1120.0, n.time: n.timepoints[-1]},
                                                       {n.ethanol: 280.0, n.time: n.timepoints[-1]}])
        ldp_no_stain.evaluate_performance(n.condition_method)

    stain_results = pd.read_csv(os.path.join("pipeline_outputs/(yeast_ethanol_1)_(yeast_ethanol_1)/ratio_df.csv"))
    no_stain_results = pd.read_csv(os.path.join("pipeline_outputs/(yeast_ethanol_0)_(yeast_ethanol_0)/ratio_df.csv"))
    stain_results["was stain used?"] = True
    no_stain_results["was stain used?"] = False
    relevant_cols = ["ethanol", "time_point", "predicted %live", "was stain used?"]
    concatenated = pd.concat([stain_results[relevant_cols], no_stain_results[relevant_cols]])
    print(concatenated)
    overlaid_time_series_plot(concatenated_ratio_df=concatenated, treatment="ethanol",
                              style_col="was stain used?", style_order=[True, False],
                              font_scale=1.0)


if __name__ == '__main__':
    main()
