#!/usr/bin/env python

from pysd2citten.Names import Names
from pysd2citten import live_dead_classifier as ldc

import logging

l = logging.getLogger(__file__)
l.setLevel(logging.DEBUG)


def compute_predicted_output(df,
                             training_df=None,
                             data_columns=['FSC_A', 'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H',
                                           'SSC_H', 'BL1_H', 'RL1_H', 'FSC_W', 'SSC_W',
                                           'BL1_W', 'RL1_W'],
                             out_dir='.',
                             strain_col=Names.STRAIN,
                             high_control=Names.NOR_00_CONTROL,
                             low_control=Names.WT_LIVE_CONTROL,
                             id_col='id',
                             use_harness=False,
                             description=None):
    ## Build the training/test input
    if training_df is None:
        c_df = get_classifier_dataframe(df, data_columns=data_columns, strain_col=strain_col, high_control=high_control,
                                        low_control=low_control)
    else:
        c_df = training_df

    if use_harness:
        pass
        # c_df.loc[:, 'output'] = df['output']
        # c_df.loc[:, id_col] = df[id_col]
        # c_df.loc[:, 'index'] = c_df.index
        # df.loc[:, 'index'] = df.index
        # pred_df = ldc.build_model_pd(c_df,
        #                              data_df=df,
        #                              index_cols=['index', 'output', id_col],
        #                              input_cols=data_columns,
        #                              output_cols=['class_label'],
        #                              output_location=out_dir,
        #                              description=description)
        # # result_df.loc[:,'predicted_output'] = pred_df['class_label_predictions'].astype(int)
        # result_df = pred_df.rename(columns={'class_label_predictions': 'predicted_output'})
    else:
        ## Build the classifier
        (model, mean_absolute_error, test_X, test_y, scaler) = ldc.build_model(c_df)
        # print("MAE: " + str(mean_absolute_error))
        ## Predict label for unseen data
        pred_df = df[data_columns]
        pred_df = ldc.predict_live_dead(pred_df, model, scaler)
        if 'output' in df.columns:
            result_df = df[['output', id_col]]
        else:
            result_df = df[[id_col]]
        result_df.loc[:, 'predicted_output'] = pred_df['class_label'].astype(int)

    return result_df


def get_classifier_dataframe(df, data_columns = ['FSC_A', 'SSC_A', 'BL1_A', 'RL1_A', 'FSC_H', 'SSC_H', 'BL1_H', 'RL1_H', 'FSC_W', 'SSC_W', 'BL1_W', 'RL1_W'],
                                 strain_col=Names.STRAIN,
                             high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL):
    """
    Get the classifier data corresponding to controls.
    """
    l.info("get_classifier_dataframe(): strain_col = " + strain_col)
    low_df = df.loc[df[strain_col] == low_control]
    high_df = df.loc[df[strain_col] == high_control]
    low_df.loc[:,strain_col] = low_df.apply(lambda x : strain_to_class(x,  high_control=high_control, low_control=low_control, strain_col=strain_col), axis=1)
    high_df.loc[:,strain_col] = high_df.apply(lambda x : strain_to_class(x,  high_control=high_control, low_control=low_control, strain_col=strain_col), axis=1)
    low_high_df = low_df.append(high_df)

    #print(live_dead_df)
    low_high_df = low_high_df.rename(index=str, columns={strain_col: "class_label"})
    low_high_df = low_high_df[data_columns + ['class_label']]
    return low_high_df


def strain_to_class(x,  strain_col=Names.STRAIN, high_control=Names.NOR_00_CONTROL, low_control=Names.WT_LIVE_CONTROL):
    if x[strain_col] == low_control:
        return 0
    elif x[strain_col] == high_control:
        return 1
    else:
        return None
