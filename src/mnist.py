import pandas as pd
import numpy as np

train_dataframe = pd.read_csv("../mnist/train.csv")
test_dataframe = pd.read_csv("../mnist/test.csv")

train_data = np.array(train_dataframe)
test_data = np.array(test_dataframe)

train_cols, train_rows = train_data.shape
train_data = train_data.T
y = train_data[0]
x = train_data[1:train_rows]
x = x / 255.0

test_cols, test_rows = test_data.shape
test_data = test_data.T
y_test = test_data[0]
x_test = test_data[1:test_rows]
x_test = x_test / 255.0
