import os
import sys
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pysd2cat.analysis.live_dead_pipeline.ld_pipeline_classes import LiveDeadPipeline
from pysd2cat.analysis.live_dead_pipeline.names import Names as n

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)


def main():
    # ldp = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=1,
    #                        y_strain=None, y_treatment=None, y_stain=1)
    # ldp.load_data()
    # # ldp.plot_distribution(channel=n.sytox_cols[0])
    # # ldp.thresholding_method()
    # ldp.condition_method(live_conditions=None,
    #                      dead_conditions=[
    #                          {n.ethanol: 1120.0, n.time: n.timepoints[-1]},
    #                          {n.ethanol: 280.0, n.time: n.timepoints[-1]}
    #                      ]
    #                      )
    # # ldp.cluster_method(n_clusters=2)
    # # ldp.evaluate_performance(n.thresholding_method)
    # # ldp.evaluate_performance(n.condition_method)
    # # ldp.evaluate_performance(n.cluster_method)
    # ldp.time_series_plot(n.condition_method)

    ldp_2 = LiveDeadPipeline(x_strain=n.yeast, x_treatment=n.ethanol, x_stain=0,
                             y_strain=n.yeast, y_treatment=n.ethanol, y_stain=1)
    ldp_2.load_data()
    ldp_2.plot_distribution(channel=n.sytox_cols[0], plot_x=True, plot_y=True, num_bins=50,
                            drop_zeros=True)


if __name__ == '__main__':
    main()
