import os
import sys
import random
import inspect
import itertools
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pysd2cat.analysis.live_dead_pipeline.names import Names as n
from harness.test_harness_class import TestHarness
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification

matplotlib.use("tkagg")
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', None)

# how is this different from os.path.dirname(os.path.realpath(__file__))?
current_dir_path = os.getcwd()


# TODO: figure out where/when dataframes should be copied or not
# TODO: add in cross-strain and cross-treatment labeling if it makes sense.
class LiveDeadPipeline:
    def __init__(self, x_strain=n.yeast, x_treatment=n.ethanol, x_stain=1,
                 y_strain=None, y_treatment=None, y_stain=None, use_previously_trained_model=False):
        """
        Assuming we will always normalize via Standard Scalar and log10-transformed data, so those are not arguments.
        """
        # TODO: add NotImplementedError checks
        self.x_strain = x_strain
        self.x_treatment = x_treatment
        self.x_stain = x_stain
        self.y_strain = self.x_strain if y_strain is None else y_strain
        self.y_treatment = self.x_treatment if y_treatment is None else y_treatment
        self.y_stain = self.x_stain if y_stain is None else y_stain

        self.x_experiment_id = n.exp_dict[(self.x_strain, self.x_treatment)]
        self.x_data_path = os.path.join(current_dir_path, n.exp_data_dir, self.x_experiment_id)
        self.y_experiment_id = n.exp_dict[(self.y_strain, self.y_treatment)]
        self.y_data_path = os.path.join(current_dir_path, n.exp_data_dir, self.y_experiment_id)

        if (self.x_stain == 0) or (self.y_stain == 0):
            self.feature_cols = n.morph_cols
        else:
            self.feature_cols = n.morph_cols + n.sytox_cols

        self.harness_path = os.path.join(current_dir_path, n.harness_output_dir)
        self.runs_path = os.path.join(self.harness_path, "test_harness_results/runs")
        self.labeled_data_dict = {}

        self.x_info = "{}_{}_{}".format(self.x_strain, self.x_treatment, self.x_stain)
        self.y_info = "{}_{}_{}".format(self.y_strain, self.y_treatment, self.y_stain)
        self.output_dir_name = "({})_({})".format(self.x_info, self.y_info)
        self.output_path = os.path.join(current_dir_path, n.pipeline_output_dir, self.output_dir_name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    # ----- Preprocessing -----

    # TODO: save out intermediate files (e.g. normalized train/test split csvs) and check if they exist already when running pipeline
    # save them with consistent names in the folder specified by self.data_path. Append argument info to file names?
    def load_data(self):
        """
        All data coming in should already be log10-transformed, so no need to transform it.
        :return:
        :rtype:
        """
        x_df = pd.read_csv(os.path.join(self.x_data_path, n.data_file_name))
        if (self.y_strain == self.x_strain) & (self.y_treatment == self.x_treatment) & (self.y_stain == self.x_stain):
            y_df = x_df  # TODO: need to make sure this is ok, might need to do x_df.copy() instead
        else:
            y_df = pd.read_csv(os.path.join(self.y_data_path, n.data_file_name))

        # filter dfs based on if user specified stained data or unstained data
        x_df = x_df.loc[x_df[n.stain] == self.x_stain]
        y_df = y_df.loc[y_df[n.stain] == self.y_stain]

        # we want to have the data normalized before model runs because of data visualizations? Or actually is that what we don't want?
        # if we do want normalized, then should be fine to normalize all the data together (before train/test split),
        # because our models will re-normalize after train/test splitting, which should have the same effect (double check this)
        # x_scaler = StandardScaler()
        # y_scaler = StandardScaler()
        # x_df[self.feature_cols] = x_scaler.fit_transform(x_df[self.feature_cols])
        # y_df[self.feature_cols] = y_scaler.fit_transform(y_df[self.feature_cols])
        # print(x_df.head())
        # print()
        # print(y_df.head())
        # print()
        self.x_df = x_df.copy()
        self.y_df = y_df.copy()

    # ----- Building Blocks -----

    def cluster(self):
        """
        Clusters the data according to the algorithm you choose.
        Hamed look and see if we already have code that does this.
        """
        pass

    def confusion_matrix(self):
        pass

    def find_threshold(self):
        pass

    def invoke_test_harness(self, train_df, test_df, pred_df):
        print("initializing TestHarness object with this output_location: {}\n".format(self.harness_path))
        th = TestHarness(output_location=self.harness_path)
        th.run_custom(function_that_returns_TH_model=random_forest_classification,
                      dict_of_function_parameters={},
                      training_data=train_df,
                      testing_data=test_df,
                      description="method: {}, x_strain: {}, x_treatment: {}, x_stain: {},"
                                  " y_strain: {}, y_treatment: {}, y_stain: {}".format(inspect.stack()[1][3], self.x_strain,
                                                                                       self.x_treatment, self.x_stain,
                                                                                       self.y_strain, self.y_treatment,
                                                                                       self.y_stain),
                      target_cols=n.label,
                      feature_cols_to_use=self.feature_cols,
                      # TODO: figure out how to resolve discrepancies between x_treatment and y_treatment, since col names will be different
                      index_cols=[n.index, self.x_treatment, n.time, n.stain],
                      normalize=True,
                      feature_cols_to_normalize=self.feature_cols,
                      feature_extraction="eli5_permutation",
                      predict_untested_data=pred_df)
        return th.list_of_this_instance_run_ids[-1]

    # ----- Exploratory Methods -----
    def plot_distribution(self, channel=n.sytox_cols[0], plot_x=True, plot_y=False, num_bins=50, drop_zeros=False):
        """
        Creates histogram of the distribution of the passed-in channel.
        Defaults to plotting the distribution of x data.
        Use plot_x and plot_y to configure which distributions to plot.
        TODO: add ability to pass-in list of channels, which creates subplots or a facetgrid
        """
        dp = plt.figure()
        if plot_x:
            x_channel_values = self.x_df[channel]
            if drop_zeros:
                x_channel_values = x_channel_values[x_channel_values != 0]
            sns.distplot(x_channel_values, bins=num_bins, color="tab:red", label=self.x_info,
                         norm_hist=False, kde=False)
        if plot_y:
            y_channel_values = self.y_df[channel]
            if drop_zeros:
                y_channel_values = y_channel_values[y_channel_values != 0]
            sns.distplot(y_channel_values, bins=num_bins, color="tab:cyan", label=self.y_info,
                         norm_hist=False, kde=False)
        if (not plot_x) and (not plot_y):
            raise NotImplementedError("plot_x and plot_y can't both be False, otherwise you aren't plotting anything!")
        plt.legend()
        plt.title("Histogram of {}".format(channel))
        plt.savefig(os.path.join(self.output_path, "histogram_of_{}.png".format(channel)))
        plt.close(dp)

    def scatterplot(self):
        pass
        # palette = itertools.cycle(sns.color_palette())
        # ets = np.array([[0.0, 1, 1],
        #                 [210.0, 1, 1],
        #                 [1120.0, 1, 1]])
        # for i in ets:
        #     ethanol = i[0]
        #     timepoint = i[1]
        #     stain = i[2]
        #     df_sub = self.x_df[(self.x_df['ethanol'] == ethanol) &
        #                     (self.x_df['time_point'] == timepoint) &
        #                     (self.x_df['stain'] == stain)]
        #     if negative_outlier_cutoff is not None:
        #         df_sub = df_sub.loc[df_sub[channel] >= negative_outlier_cutoff]
        #     print(len(df_sub))
        #
        #     sns.distplot(df_sub[channel], bins=50, color=next(palette), norm_hist=False, kde=False,
        #                  label="Eth: {}, Time: {}, Stain: {}".format(ethanol, timepoint, stain))
        #
        # plt.legend()
        # if negative_outlier_cutoff is not None:
        #     plt.title("Distributions of the {} channel. Removed outliers below {}.".format(channel, negative_outlier_cutoff))
        # else:
        #     plt.title("Distributions of the {} channel.".format(channel))
        # plt.show()

    # ----- Filtering Debris Methods -----

    # ----- Labeling methods -----

    def condition_method(self, live_conditions=None, dead_conditions=None):
        """
        Define certain tuples of (treatment, time-point) as live or dead.
            E.g. (0-treatment, final timepoint) = live, (max-treatment, final timepoint) = dead
        Train supervised model on those definitions, and predict live/dead for all points.
        Final product is dataframe with original data and predicted labels, which is set to self.predicted_data


        :param live_conditions:
        :type live_conditions: list of dicts
        :param dead_conditions:
        :type dead_conditions: list of dicts
        """
        labeling_method_name = inspect.stack()[0][3]
        print("Starting {} labeling".format(labeling_method_name))

        if live_conditions is None:
            live_conditions = [{self.x_treatment: n.treatments_dict[self.x_treatment][self.x_strain][0], n.time: n.timepoints[-1]}]
        if dead_conditions is None:
            dead_conditions = [{self.x_treatment: n.treatments_dict[self.x_treatment][self.x_strain][-1], n.time: n.timepoints[-1]}]
        print("Conditions designated as Live: {}".format(live_conditions))
        print("Conditions designated as Dead: {}".format(dead_conditions))
        print()

        # Label points according to live_conditions and dead_conditions
        # first obtain indexes of the rows that correspond to live_conditions and dead_conditions
        live_indexes = []
        dead_indexes = []
        for lc in live_conditions:
            live_indexes += list(self.x_df.loc[(self.x_df[list(lc)] == pd.Series(lc)).all(axis=1), n.index])
        for dc in dead_conditions:
            dead_indexes += list(self.x_df.loc[(self.x_df[list(dc)] == pd.Series(dc)).all(axis=1), n.index])
        labeled_indexes = live_indexes + dead_indexes

        # TODO: check if this should call .copy() or not
        labeled_df = self.x_df[self.x_df[n.index].isin(labeled_indexes)]
        labeled_df.loc[labeled_df[n.index].isin(live_indexes), n.label] = 1
        labeled_df.loc[labeled_df[n.index].isin(dead_indexes), n.label] = 0
        # mainly doing this split so Test Harness can run without balking (it expects testing_data)
        train_df, test_df = train_test_split(labeled_df, train_size=0.95, random_state=5,
                                             stratify=labeled_df[[self.x_treatment, n.time, n.label]])

        # Invoke Test Harness
        run_id = self.invoke_test_harness(train_df=train_df, test_df=test_df, pred_df=self.y_df)
        labeled_df = pd.read_csv(os.path.join(self.runs_path, "run_{}/predicted_data.csv".format(run_id)))
        print(labeled_df.head())

        self.labeled_data_dict[labeling_method_name] = labeled_df

    def thresholding_method(self, channel=n.sytox_cols[0]):
        """
        Currently uses arithmetic mean of channel (default RL1-A).
        Since channels are logged, arithmetic mean is equivalent to geometric mean of original channel values.
        Final product is dataframe with original data and predicted labels, which is set to self.predicted_data
        """
        labeling_method_name = inspect.stack()[0][3]
        print("Starting {} labeling".format(labeling_method_name))

        channel_values = list(self.x_df[channel])
        threshold = np.array(channel_values).mean()
        print("threshold used = {}".format(threshold))

        labeled_df = self.x_df.copy()
        labeled_df.loc[labeled_df[channel] >= threshold, n.label_preds] = 1
        labeled_df.loc[labeled_df[channel] < threshold, n.label_preds] = 0
        print(labeled_df.head())

        self.labeled_data_dict[labeling_method_name] = labeled_df

    def cluster_method(self, n_clusters=4):
        """
        TODO: review this method's code
        """
        labeling_method_name = inspect.stack()[0][3]
        print("Starting {} labeling".format(labeling_method_name))

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        scaled_x_df = self.x_df.copy()
        scaled_y_df = self.y_df.copy()
        scaled_x_df[self.feature_cols] = x_scaler.fit_transform(scaled_x_df[self.feature_cols])
        scaled_y_df[self.feature_cols] = y_scaler.fit_transform(scaled_x_df[self.feature_cols])

        c_model = KMeans(n_clusters=n_clusters, random_state=5)
        c_model.fit(scaled_x_df[self.feature_cols])

        # wrote these 2 lines in a way to make sure that original non-normalized features are kept in labeled_df
        labeled_df = self.y_df.copy()
        labeled_df[n.cluster_preds] = c_model.predict(scaled_y_df[self.feature_cols])

        normalize = "all"
        frequency_table = pd.crosstab(index=labeled_df[self.y_treatment],
                                      columns=labeled_df[n.cluster_preds],
                                      normalize=normalize)
        print(frequency_table)
        print()

        # Define live/dead based on clustering results
        # I define the "live" cluster to be the one with the largest number of points belonging to the 0.0 treatment level
        # All other clusters are defined to be "dead"
        live_clusters = frequency_table.idxmax(axis=1)[0]
        dead_clusters = [x for x in list(frequency_table.columns.values) if x != live_clusters]
        if len(dead_clusters) == 1:
            dead_clusters = dead_clusters[0]
        print("live_cluster: {}".format(live_clusters))
        print("dead_clusters: {}".format(dead_clusters))
        print()

        hm = plt.figure()
        sns.heatmap(frequency_table, cmap="Blues")
        plt.xlabel("Cluster")
        plt.title("{}: {} Concentration vs. {} KMeans Clusters.".format(self.y_strain, self.y_treatment, n_clusters))
        plt.text(0.99, 0.99, "live_clusters: {}\ndead_clusters: {}".format(live_clusters, dead_clusters),
                 ha='right', va='top', fontsize=10, transform=hm.transFigure)
        plt.savefig(os.path.join(self.output_path, "cluster_heatmap_with_{}_clusters.png".format(n_clusters)))
        plt.close(hm)

        # Add labels based on cluster-derived live/dead definitions.
        labeled_df[n.label_preds] = 0  # dead
        labeled_df.loc[labeled_df[n.cluster_preds] == live_clusters, n.label_preds] = 1  # live
        print(labeled_df[n.label_preds].value_counts(dropna=False))
        print()
        print(labeled_df.head())

        self.labeled_data_dict[labeling_method_name] = labeled_df
        # self.invoke_test_harness() ?

    # ----- Performance Evaluation -----
    def evaluate_performance(self, labeling_method):
        """
        Calls qualitative and quantitative methods for performance evaluation
        """
        if labeling_method not in self.labeled_data_dict.keys():
            raise NotImplementedError("The labeling method you are trying to evaluate has not been run yet."
                                      "Please run the labeling method first.")
        ratio_df = self.time_series_plot(labeling_method=labeling_method)
        self.timeseries_scatter(labeling_method=labeling_method)

    def time_series_plot(self, labeling_method):
        """
        Takes in a dataframe that has been labeled and generates a time-series plot of
        percent alive vs. time, colored by treatment amount.
        This serves as a qualitative metric that allows us to compare different methods of labeling live/dead.
        """
        matplotlib.use("tkagg")
        if labeling_method not in self.labeled_data_dict.keys():
            raise NotImplementedError("The labeling method you are trying to create a time_series_plot for has not been run yet."
                                      "Please run the labeling method first.")
        else:
            labeled_df = self.labeled_data_dict[labeling_method]

        ratio_df = pd.DataFrame(columns=[self.y_treatment, n.time, n.num_live, n.num_dead, n.percent_live])
        for tr in list(labeled_df[self.y_treatment].unique()):
            for ti in list(labeled_df[n.time].unique()):
                num_live = len(labeled_df.loc[(labeled_df[self.y_treatment] == tr) & (labeled_df[n.time] == ti) & (
                        labeled_df[n.label_preds] == 1)])
                num_dead = len(labeled_df.loc[(labeled_df[self.y_treatment] == tr) & (labeled_df[n.time] == ti) & (
                        labeled_df[n.label_preds] == 0)])
                ratio_df.loc[len(ratio_df)] = [tr, ti, num_live, num_dead, float(num_live) / (num_live + num_dead)]
        palette = sns.color_palette("bright", 5)

        lp = plt.figure()
        sns.lineplot(x=ratio_df[n.time], y=ratio_df[n.percent_live], hue=ratio_df[self.y_treatment], palette=palette)
        plt.ylim(0, 1)
        plt.title("Predicted Live over Time using {}\n{}".format(labeling_method, self.output_dir_name))  # TODO make title prettier
        # lp.set(ylim=(0, 1))
        # lp.set(title="Predicted Live over Time using {}\n{}".format(labeling_method, self.output_dir_name))  # TODO make title prettier
        # TODO: add make_dir_if_does_not_exist
        plt.savefig(os.path.join(self.output_path, "time_series_{}.png".format(labeling_method)))
        plt.close(lp)
        ratio_df.to_csv(os.path.join(self.output_path, "ratio_df.csv"), index=False)

        return ratio_df

    # TODO: explore other bivariate distribution plots like hexbin plots
    # https://seaborn.pydata.org/tutorial/distributions.html
    def timeseries_scatter(self, labeling_method, xcol="log_SSC-A", ycol="log_RL1-A", sample_fraction=0.1, kdeplot=False):
        matplotlib.use("tkagg")
        if labeling_method not in self.labeled_data_dict.keys():
            raise NotImplementedError("The labeling method you are trying to create a time_series_plot for has not been run yet."
                                      "Please run the labeling method first.")
        else:
            labeled_df = self.labeled_data_dict[labeling_method]

        if xcol not in labeled_df.columns.values:
            labeled_df = pd.merge(labeled_df, self.y_df[[n.index, xcol]], on=n.index)
        if ycol not in labeled_df.columns.values:
            labeled_df = pd.merge(labeled_df, self.y_df[[n.index, ycol]], on=n.index)

        fig, ax = plt.subplots(ncols=len(n.timepoints), nrows=len(n.treatments_dict[self.y_treatment][self.y_strain]),
                               figsize=(4 * len(n.timepoints), 4 * len(n.treatments_dict[self.y_treatment][self.y_strain])), dpi=200)
        # iterate through rows of subplots. Each row corresponds to a treatment concentration
        for i, row in enumerate(ax):
            curr_treatment = n.treatments_dict[self.y_treatment][self.y_strain][i]  # current treatment conc to deal with
            # iterate through columns of subplots. Each column corresponds to a time-point
            for j, col in enumerate(row):
                curr_time = n.timepoints[j]  # current time-point to deal with
                subplot_df = labeled_df.loc[(labeled_df[self.y_treatment] == curr_treatment) &
                                            (labeled_df[n.time] == curr_time)]
                live_df = subplot_df.loc[subplot_df[n.label_preds] == 1]
                dead_df = subplot_df.loc[subplot_df[n.label_preds] == 0]
                live_df = live_df.sample(frac=sample_fraction)
                dead_df = dead_df.sample(frac=sample_fraction)
                try:
                    if kdeplot:
                        sns.kdeplot(live_df[xcol], live_df[ycol], ax=col, alpha=0.5, cmap="Blues", shade=True, label="Live",
                                    shade_lowest=False,
                                    dropna=True)
                    else:
                        col.scatter(live_df[xcol], live_df[ycol], c="Blue", label="pred_{}".format("live"),
                                    s=100, alpha=0.4, marker='o', edgecolor='black', linewidth='0')
                except Exception as e:
                    pass
                try:
                    if kdeplot:
                        sns.kdeplot(dead_df[xcol], dead_df[ycol], ax=col, alpha=0.5, cmap="Reds", shade=True, label="Dead",
                                    shade_lowest=False,
                                    dropna=True)
                    else:
                        col.scatter(dead_df[xcol], dead_df[ycol], c="Red", label="pred_{}".format("dead"),
                                    s=100, alpha=0.4, marker='o', edgecolor='black', linewidth='0')
                    col.legend()
                except Exception as e:
                    pass

                col.set_xlabel("{}".format(xcol))
                col.set_ylabel("{}".format(ycol))
                col.set_xlim(0, 7)
                col.set_ylim(0, 7)

                col.set_title("Ethanol (uL): " + str(curr_treatment) + " Time (h): " + str(curr_time))
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.title(self.y_strain)
        plt.savefig(os.path.join(self.output_path, "scatter_{}_{}_{}.png".format(labeling_method, xcol, ycol)))
        plt.close(fig)

    def quantitative_metrics(self, labeling_method):
        """
        Takes in a dataframe that has been labeled and runs a consistent supervised model on a train/test split of the now-labeled data.
        The model’s performance will be a quantitative way to see if our labels actually make sense.
         For example, if I give random labels to the data, then no supervised model will perform well on that data.
         But if the labels line up with some “ground truth”, then a supervised model should be able to perform better.
         Todo: look into how people evaluate semi-supervised models.
        """


class ComparePipelines:
    pass
