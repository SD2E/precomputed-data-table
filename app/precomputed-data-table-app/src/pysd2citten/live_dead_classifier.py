#!/usr/bin/env python

import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

l = logging.getLogger(__file__)
l.setLevel(logging.INFO)


def predict_live_dead(df, model, scaler):
    df_norm = scaler.transform(df)
    predictions = model.predict(df_norm)
    preds = pd.DataFrame(predictions, columns = ['class_label'])
    df.loc[:,'class_label'] = predictions
    return df


def build_model(dataframe):
    X = dataframe.drop(columns=['class_label'])
    y = dataframe['class_label'].astype(int)

    # train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    train_X, val_X, train_y, val_y = train_test_split(X, y, stratify=y,
                                                      test_size=0.2, random_state=5)

    scaler = StandardScaler().fit(train_X)
    train_X_norm = scaler.transform(train_X)

    # Define model
    rf_model = RandomForestClassifier(random_state=1, class_weight='balanced',
                                      n_estimators=361, criterion='entropy', min_samples_leaf=13, n_jobs=-1)

    # Fit model
    rf_model.fit(train_X_norm, train_y)

    val_X_norm = scaler.transform(val_X)
    val_p = pd.DataFrame(rf_model.predict(val_X_norm), columns=['class_label'])
    error = mean_absolute_error(val_y, val_p)
    # print(dataframe.columns)
    # val_X_norm = pd.DataFrame(val_X_norm, columns=dataframe.columns[1:])

    return (rf_model, error, val_X_norm, val_p, scaler)


def get_threshold_pr(df, threshold):
    df['live'] = df['RL1-A'].apply(lambda x: x < threshold)
    return df['live'].mean()
