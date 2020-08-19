import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pandas.core.common import SettingWithCopyWarning
from pysd2cat.analysis.Names import Names

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', -1)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def main():
    cond0 = pd.read_csv("data/output/condition0.csv")
    cond0.drop(columns="Unnamed: 0", inplace=True)
    cond1 = pd.read_csv("data/output/condition1.csv")
    cond1.drop(columns="Unnamed: 0", inplace=True)

    # set y-axis value here:
    value = "yfp_norm"
    cond0 = cond0[cond0[Names.WELL]=="A1"]
    time_of_increase, cond_well_df = find_time_of_increase(cond0, value, "moving_average")
    #print(time_of_increase)
    print()

    # sns.scatterplot(cond_well_df["Time"], cond_well_df[value])
    # sns.scatterplot(cond_well_df["Time"], cond_well_df["moving_average"])
    # sns.scatterplot(cond_well_df["Time"], cond_well_df["moving_average_gradient"])
    # plt.show()

    #print(cond_well_df.shape)
    increasing_df = cond_well_df.loc[cond_well_df[Names.TIME] >= time_of_increase]
    #print(increasing_df.shape)

    #sns.scatterplot(increasing_df["Time"], increasing_df[value])
    #plt.show()

    doubling_time(increasing_df, value)


def find_time_of_increase(cond_df, value_col, time_col=Names.TIME, window=8, method="moving_average"):
    '''
    Get the point of inflection to focus on the growth
    :param cond_df: the dataframe of interest
    :param value_col: the column you want to model
    :param method: moving_average, global_min
    :return:
    '''
    assert (method == "global_min") or (method == "moving_average"), "method must be 'global_min' or 'moving_average'"
    cond_well_df = cond_df.copy()

    if method == "global_min":
        time_of_increase = cond_well_df.loc[cond_well_df[value_col] == cond_well_df[value_col].min(), time_col].item()
    elif method == "moving_average":
        cond_well_df["moving_average"] = cond_well_df[value_col].rolling(window).mean()
        cond_well_df["moving_average_gradient"] = cond_well_df["moving_average"].diff() / cond_well_df[time_col].diff()
        positive_gradients = cond_well_df.loc[cond_well_df["moving_average_gradient"] > 0]
        time_of_increase = positive_gradients.iloc[0][time_col]
    elif method == 'remove_transient':
        time_of_increase = 24
    else:
        raise ValueError("method must be 'global_min' or 'moving_average'")

    return time_of_increase, cond_well_df


def exp_func(t, n0, td):
    return n0 * np.exp2(t / td)

def lin_func(t, n0, m):
    return m*t + n0

def doubling_time(df, value_col, time_col=Names.TIME, func=exp_func):
    '''
    Calculate doubling time. 
    :param df: dataframe of interest
    :param value_col: column of interest
    :return:
    '''
    popt, pcov = curve_fit(func, df[time_col], df[value_col])
    #print("Optimal parameters")
    #print(popt)
    #print("Parameter error")
    #print(np.sqrt(np.diag(pcov)))
    time_points = df[time_col]
    y_vals = func(time_points, *popt)

    # TODO: Move the plotting functionality to qd
    '''
    sns.scatterplot(df[Names.TIME], df[value_col],label='actual').set(xlabel='Time', ylabel='Flourescence')
    sns.scatterplot(time_points, y_vals,label='predicted').set(xlabel='Time', ylabel='Flourescence')

    plt.show()
    '''

    return (time_points,y_vals,popt,pcov)



if __name__ == '__main__':
    main()
