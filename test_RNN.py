import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# load model
model = load_model('C:/Users/Sravani Subraveti/OneDrive/Spring 2020/ECE657-Tools of Intelligent Systems/Assignments/Assignment 3/models/20847722_RNN_model.h5')
data_load = read_csv("C:/Users/Sravani Subraveti/OneDrive/Spring 2020/ECE657-Tools of Intelligent Systems/Assignments/Assignment 3/data/train_data_RNN.csv")
data_load_test = read_csv("C:/Users/Sravani Subraveti/OneDrive/Spring 2020/ECE657-Tools of Intelligent Systems/"
                          "Assignments/Assignment 3/data/test_data_RNN.csv")
data_X= data_load.iloc[:, 1:13].values
data_Y= data_load.iloc[:, -1:].values
data_test_X = data_load_test.iloc[:, 1:13].values
data_test_Y = data_load_test.iloc[:, -1:].values
data_test_X = data_test_X.astype('float32')

# normalize the data attributes
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_X = scaler.fit_transform(data_X)
normalized_X_test = scaler.transform(data_test_X)
normalized_X_test = np.reshape(normalized_X_test, (normalized_X_test.shape[0], 3, 4))
testPredict = model.predict(normalized_X_test)

data_test_Y_1 = [data_test_Y]

testScore = math.sqrt(mean_squared_error(data_test_Y_1[0], testPredict[:,0]))

print('Test Score: %.2f RMSE' % (testScore))
print("done")

plt.figure()
plt.plot(testPredict)
plt.plot(data_test_Y)
plt.title('Prediction vs Real Stock Price')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Prediction', 'Original'], loc='upper left')
plt.show()



and loss
