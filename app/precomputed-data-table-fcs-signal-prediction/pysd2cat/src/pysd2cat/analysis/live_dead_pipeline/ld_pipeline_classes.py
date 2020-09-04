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
import plotly
# TODO: figure out orca package issues (will allow us to save plotly figures as static pngs)
# plotly.io.orca.config.executable = 'orca.app'
# /Users/he/anaconda3/bin/orca.app
import plotly.express as px
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
        #     time_point = i[1]
        #     stain = i[2]
        #     df_sub = self.x_df[(self.x_df['ethanol'] == ethanol) &
        #                     (self.x_df['time_point'] == time_point) &
        #                     (self.x_df['stain'] == stain)]
        #     if negative_outlier_cutoff is not None:
        #         df_sub = df_sub.loc[df_sub[channel] >= negative_outlier_cutoff]
        #     print(len(df_sub))
        #
        #     sns.distplot(df_sub[channel], bins=50, color=next(palette), norm_hist=False, kde=False,
        #                  label="Eth: {}, Time: {}, Stain: {}".format(ethanol, time_point, stain))
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
        Define certain tuples of (treatment, time_point) as live or dead.
            E.g. (0-treatment, final time_point) = live, (max-treatment, final time_point) = dead
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
            live_conditions = [{self.x_treatment: n.treatments_dict[self.x_treatment][self.x_strain][0], n.time: n.time_points[-1]}]
        if dead_conditions is None:
            dead_conditions = [{self.x_treatment: n.treatments_dict[self.x_treatment][self.x_strain][-1], n.time: n.time_points[-1]}]
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
        # print(labeled_df.head())

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
        # print(labeled_df.head())

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
        # print(labeled_df.head())

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
        ratio_df = self.plot_percent_live_over_conditions(labeling_method=labeling_method)
        self.plot_features_over_conditions(labeling_method=labeling_method)

    def plot_percent_live_over_conditions(self, labeling_method):
        """
        Plots percent live (predicted) over all of the y_experiment conditions.
        Takes in a dataframe that has been labeled and generates a
        plot of percent alive vs. time, colored by treatment amount.
        This serves as a qualitative metric that allows us to compare different methods of labeling live/dead.
        """
        matplotlib.use("tkagg")
        if labeling_method not in self.labeled_data_dict.keys():
            raise NotImplementedError("The labeling method you are trying to create a plot for has not been run yet."
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

    def plot_features_over_conditions(self, labeling_method, axis_1="log_SSC-A", axis_2="log_RL1-A",
                                      sample_fraction=0.1, kdeplot=False):
        """
        Will use self.y_df not self.x_df
        """
        # matplotlib.use("tkagg")
        if labeling_method not in self.labeled_data_dict.keys():
            raise NotImplementedError("The labeling method you are trying to create a plot for has not been run yet."
                                      "Please run the labeling method first.")
        else:
            labeled_df = self.labeled_data_dict[labeling_method]

        if axis_1 not in labeled_df.columns.values:
            labeled_df = pd.merge(labeled_df, self.y_df[[n.index, axis_1]], on=n.index)
        if axis_2 not in labeled_df.columns.values:
            labeled_df = pd.merge(labeled_df, self.y_df[[n.index, axis_2]], on=n.index)

        if set(labeled_df[n.time].unique()) != set(n.time_points):
            raise Warning("The unique values in the {} column of labeled_df "
                          "do not match those in n.time_points: {}".format(n.time, n.time_points))

        plotly_fig = grid_of_distributions_over_conditions(df1=labeled_df, axis_1=axis_1, axis_2=axis_2,
                                                           grid_rows_source=self.y_treatment,
                                                           grid_cols_source=n.time,
                                                           color_source=n.label_preds,
                                                           sample_fraction=sample_fraction, df2=None, kdeplot=kdeplot)

        html_output = os.path.join(self.output_path, "scatter_{}_{}_{}.html".format(labeling_method, axis_1, axis_2))
        png_output = os.path.join(self.output_path, "scatter_{}_{}_{}.png".format(labeling_method, axis_1, axis_2))

        plotly_fig.write_html(html_output)
        # saving as static image (png) doesn't currently work, need to figure out issues with orca package
        # plotly_fig.write_image(png_output)

    def quantitative_metrics(self, labeling_method):
        """
        Takes in a dataframe that has been labeled and runs a consistent supervised model on a train/test split of the now-labeled data.
        The model’s performance will be a quantitative way to see if our labels actually make sense.
         For example, if I give random labels to the data, then no supervised model will perform well on that data.
         But if the labels line up with some “ground truth”, then a supervised model should be able to perform better.
         Todo: look into how people evaluate semi-supervised models.
        """


class ComparePipelines:
    """
    Allows for the comparison of the labeling results of two pipelines
    """

    def __init__(self, ld_pipeline_1, ld_pipeline_2):
        self.ld_pipeline_1 = ld_pipeline_1
        self.ld_pipeline_2 = ld_pipeline_2

        # TODO update output_path to make things simpler/clearer
        self.output_dir_name = "[{}]_vs_[{}]".format(self.ld_pipeline_1.output_dir_name, self.ld_pipeline_2.output_dir_name)
        self.output_path = os.path.join(current_dir_path, n.pipeline_output_dir, "compare_pipelines", self.output_dir_name)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def compare_plots_of_features_over_conditions(self, labeling_method_1, labeling_method_2,
                                                  axis_1="log_SSC-A", axis_2="log_RL1-A",
                                                  sample_fraction=0.1, kdeplot=False):
        """
        """
        if labeling_method_1 not in self.ld_pipeline_1.labeled_data_dict.keys():
            raise NotImplementedError("The labeling method you are trying to create a plot for has not been run yet."
                                      "Please run the labeling method first.")
        else:
            labeled_df_1 = self.ld_pipeline_1.labeled_data_dict[labeling_method_1]

        if labeling_method_2 not in self.ld_pipeline_2.labeled_data_dict.keys():
            raise NotImplementedError("The labeling method you are trying to create a plot for has not been run yet."
                                      "Please run the labeling method first.")
        else:
            labeled_df_2 = self.ld_pipeline_2.labeled_data_dict[labeling_method_2]

        if axis_1 not in labeled_df_1.columns.values:
            labeled_df = pd.merge(labeled_df_1, self.ld_pipeline_1.y_df[[n.index, axis_1]], on=n.index)
        if (axis_2 is not None) & (axis_2 not in labeled_df_1.columns.values):
            labeled_df = pd.merge(labeled_df_1, self.ld_pipeline_1.y_df[[n.index, axis_2]], on=n.index)

        if axis_1 not in labeled_df_2.columns.values:
            labeled_df = pd.merge(labeled_df_2, self.ld_pipeline_2.y_df[[n.index, axis_1]], on=n.index)
        if (axis_2 is not None) & (axis_2 not in labeled_df_2.columns.values):
            labeled_df = pd.merge(labeled_df_2, self.ld_pipeline_2.y_df[[n.index, axis_2]], on=n.index)

        plotly_fig = grid_of_distributions_over_conditions(df1=labeled_df_1, axis_1=axis_1, axis_2=axis_2,
                                                           grid_rows_source=self.ld_pipeline_1.y_treatment,
                                                           grid_cols_source=n.time,
                                                           color_source=n.label_preds,
                                                           sample_fraction=sample_fraction,
                                                           df2=labeled_df_2, kdeplot=kdeplot)

        # TODO update this file name to include labeling_method_2 or somehow make things clearer
        html_output = os.path.join(self.output_path, "scatter_{}_{}_{}.html".format(labeling_method_1, axis_1, axis_2))
        png_output = os.path.join(self.output_path, "scatter_{}_{}_{}.png".format(labeling_method_1, axis_1, axis_2))

        plotly_fig.write_html(html_output)
        # saving as static image (png) doesn't currently work, need to figure out issues with orca package
        # plotly_fig.write_image(png_output)


def grid_of_distributions_over_conditions(df1, axis_1, axis_2,
                                          grid_rows_source, grid_cols_source, color_source,
                                          sample_fraction=0.1, df2=None, kdeplot=False):
    """
    I put this into it's own function because it can be used by both LiveDeadPipeline and ComparePipelines.
    Returns a seaborn FacetGrid.
    If axis_2 = None, will create histograms of the distribution of the axis_1 feature.
    If both df1 and df2 are provided, then both datasets will be plotted and the legend will differentiate.
        - if using both df1 and df2, then their treatment type and concentrations should be the same
        - same goes for time_points
    """
    matplotlib.use("tkagg")

    assert isinstance(df1, pd.DataFrame), "df1 must be a Pandas DataFrame."
    assert isinstance(axis_1, str), "axis_1 must be a string."
    assert (isinstance(axis_2, str)) or (axis_2 is None), "axis_2 must be a string or None."

    if axis_2 is None:
        relevant_columns = [axis_1, grid_rows_source, grid_cols_source, color_source]
    else:
        relevant_columns = [axis_1, axis_2, grid_rows_source, grid_cols_source, color_source]
    df1 = df1[relevant_columns].copy()

    if df2 is not None:
        df2 = df2[relevant_columns].copy()

        df1_grid_cols = set(df1[grid_cols_source].unique())
        df1_grid_rows = set(df1[grid_rows_source].unique())
        df2_grid_cols = set(df2[grid_cols_source].unique())
        df2_grid_rows = set(df2[grid_rows_source].unique())
        if df1_grid_cols != df2_grid_cols:
            raise Warning("The unique values in the {} column "
                          "do not match between df1 and df2".format(grid_cols_source))
        if df1_grid_rows != df2_grid_rows:
            raise Warning("The unique values in the {} column "
                          "do not match between df1 and df2".format(grid_rows_source))

    if sample_fraction < 1.0:
        # using train_test_split to get a stratified sample of the data
        df1, _ = train_test_split(df1, train_size=sample_fraction, random_state=5,
                                  stratify=df1[[grid_rows_source, grid_cols_source, color_source]])
        if df2 is not None:
            df2, _ = train_test_split(df2, train_size=sample_fraction, random_state=5,
                                      stratify=df2[[grid_rows_source, grid_cols_source, color_source]])

    df1[color_source] = df1[color_source].astype(int)
    if df2 is None:
        df1[color_source] = df1[color_source].astype(str)
        df_plot = df1.copy()
    else:
        df2[color_source] = df2[color_source].astype(int)
        df1[color_source] = 'df1_' + df1[color_source].astype(str)
        df2[color_source] = 'df2_' + df2[color_source].astype(str)
        df_plot = pd.concat([df1, df2])

    grid_col_order = list(df_plot[grid_cols_source].unique())
    grid_row_order = list(df_plot[grid_rows_source].unique())
    grid_hue_order = list(df_plot[color_source].unique())
    grid_col_order.sort()
    grid_row_order.sort()
    grid_hue_order.sort()

    # Plotting code is below here
    if axis_2 is not None:
        plotly_fig = px.scatter(df_plot, x=axis_1, y=axis_2, color=color_source, size=None,
                                facet_col=grid_cols_source, facet_row=grid_rows_source,
                                category_orders={grid_cols_source: grid_col_order,
                                                 grid_rows_source: grid_row_order,
                                                 color_source: grid_hue_order},
                                opacity=0.5)
    else:
        raise NotImplementedError("single axis distplots have not been implemented yet")

    return plotly_fig


"""
Archived code, will use or delete later, didn't want to delete for now

Scatterplot grid code:

        # relplot way:
        with sns.plotting_context(rc={"legend.fontsize": 60}):
            grid = sns.relplot(x=axis_1, y=axis_2, alpha=0.5, data=df_plot, col=grid_cols_source, row=grid_rows_source, hue=color_source,
                               col_order=grid_col_order, row_order=grid_row_order, hue_order=grid_hue_order, legend="full")
                                       

        # FacetGrid way:
        grid = sns.FacetGrid(data=df_plot, col=grid_cols_source, row=grid_rows_source, hue=color_source,
                             col_order=grid_col_order, row_order=grid_row_order, hue_order=grid_hue_order)
        grid.map(sns.scatterplot, axis_1, axis_2)
        # plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        grid.fig.subplots_adjust(top=0.8)
        grid.fig.suptitle("BOOGALOO", size=20, horizontalalignment="center")
        grid.add_legend(title="LEGEND", bbox_to_anchor=(0.5, 0.85), ncol=len(grid_col_order), fontsize=14, title_fontsize=100)
        grid.savefig("asdf.png")
        sys.exit(0)
        
        
        # Updating legend
        legend = grid._legend
        legend.set_bbox_to_anchor([0.5, 0.95])  # legend coordinates
        legend._loc = 9  # says which part of the bbox to put at the coordinates above (9 stands for upper center)
        legend._fancybox = True
        legend._title = "LEGEND"
        legend._fontsize = 25
        legend._title_fontsize = 50
        legend._ncol = 1
        

Distplot code:
        grid = sns.FacetGrid(data=df_plot, col=grid_cols_source, row=grid_rows_source, hue=color_source,
                             col_order=grid_col_order, row_order=grid_row_order, hue_order=grid_hue_order)
        grid.map(sns.distplot, axis_1, bins=100, label="testing123", norm_hist=False, kde=False)
"""
