import pandas as pd
import tensorflow as tf
import sklearn.model_selection
import numpy as np

dataset_path = tf.keras.utils.get_file("auto-mpg.data", ("http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"))

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'ModelYear', 'Origin']
df = pd.read_csv(dataset_path, names=column_names, na_values = '?', comment='\t', sep=' ', skipinitialspace=True)

## drop the NA rows
df = df.dropna()
df = df.reset_index(drop=True)

## train/test splits
df_train, df_test = sklearn.model_selection.train_test_split(df, train_size=0.8)
train_stats = df_train.describe().transpose()

numeric_column_names = ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration']

df_train_norm, df_test_norm = df_train.copy(), df_test.copy()

for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, 'mean']
    std = train_stats.loc[col_name, 'std']
    df_train_norm.loc[:, col_name] = (
    df_train_norm.loc[:, col_name] - mean)/std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean) / std

numeric_features = [tf.feature_column.numeric_column(key=col_name) for col_name in numeric_column_names]
feature_year = tf.feature_column.numeric_column(key='ModelYear')
bucketized_features = [tf.feature_column.bucketized_column(source_column=feature_year, boundaries=[73, 76, 79])]
feature_origin = tf.feature_column.categorical_column_with_vocabulary_list(key='Origin', vocabulary_list=[1, 2, 3])
categorical_indicator_features = [tf.feature_column.indicator_column(feature_origin)]

def train_input_fn(df_train, batch_size=8):
    df = df_train.copy()
    train_x, train_y = df, df.pop('MPG')
    dataset = tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
    # shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)

def eval_input_fn(df_test, batch_size=8):
    df = df_test.copy()
    test_x, test_y = df, df.pop('MPG')
    dataset = tf.data.Dataset.from_tensor_slices((dict(test_x), test_y))
    return dataset.batch(batch_size)


all_feature_columns = (numeric_features + bucketized_features + categorical_indicator_features)

EPOCHS = 1000
BATCH_SIZE = 8
total_steps = EPOCHS * int(np.ceil(len(df_train) / BATCH_SIZE))

""" 
ds = train_input_fn(df_train_norm)
batch = next(iter(ds))
regressor = tf.estimator.DNNRegressor(feature_columns=all_feature_columns, hidden_units=[32, 10], model_dir='models')
regressor.train(input_fn=lambda:train_input_fn( df_train_norm, batch_size=BATCH_SIZE), steps=total_steps) 
"""

reloaded_regressor = tf.estimator.DNNRegressor(feature_columns=all_feature_columns, hidden_units=[32, 10], warm_start_from='models', model_dir='models')

eval_results = reloaded_regressor.evaluate(input_fn=lambda:eval_input_fn(df_test_norm, batch_size=8))

pred_res = reloaded_regressor.predict(input_fn=lambda: eval_input_fn(df_test_norm, batch_size=8))

print(pred_res)