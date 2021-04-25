# import required packages
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import numpy as np
from sklearn.model_selection import train_test_split

series = pd.read_csv (r'C:\Users\Sravani Subraveti\Downloads\q2_dataset.csv')
#print (series)
X= series.iloc[:, 2:6].values
Y= series.iloc[:,:2].values
temps = DataFrame(X)
temp_Y = DataFrame(Y)
print(temps)
print(temp_Y)

#dataframe = concat([temps.shift(2),temps.shift(1), temps], axis=1)
dataframe = concat([temps.shift(-3), temps.shift(-2), temps.shift(-1), temps], axis = 1)
#window = dataframe.rolling(window=4)
#print(window)
#dataframe = concat([shifted, temps], axis = 1)
dataframe.columns = ['Volume_P3', 'Open_P3', 'High_P3', 'Low_P3', 'Volume_P2', 'Open_P2', 'High_P2', 'Low_P2','Volume_P1', 'Open_P1', 'High_P1', 'Low_P1','Volume_F', 'Open_F', 'High_F', 'Low_F']
#print(type(dataframe))
dataframe=dataframe.drop(dataframe.index[[1256, 1257, 1258]])
dataframe=dataframe.drop(dataframe.columns[[12, 14, 15]], axis=1)
print(dataframe)
dataframe_past_X = dataframe.iloc[:, 0:12].values
dataframe_fut_Y =  dataframe.iloc[:, -1:].values
print(dataframe_past_X.shape)
print(dataframe_fut_Y.shape)

dataframe_past_X = dataframe_past_X.reshape(1256,3,4)
print(dataframe_past_X.shape)
# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
X_train, X_test, y_train, y_test = train_test_split(dataframe_past_X, dataframe_fut_Y, test_size=0.3, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#df = pd.DataFrame(X_train, columns = ['Volume_P3', 'Open_P3', 'High_P3', 'Low_P3', 'Volume_P2', 'Open_P2', 'High_P2', 'Low_P2','Volume_P1', 'Open_P1', 'High_P1', 'Low_P1','Volume_F'])
#df
#df.to_csv('train_data_RNN.csv')
#if __name__ == "__main__":
	# 1. load your training data

	# 2. Train your network
	# 		Make sure to print your training loss within training to show progress
	# 		Make sure you print the final training loss

	# 3. Save your model

#testPredict_dataset = np.zeros(shape=(len(testPredict), 12))
#testPredict_dataset[:, 0] = testPredict[:, 0]
#testPredict = scaler.inverse_transform(testPredict_dataset)[:, 0]
#testPredict_inv = testPredict_inv.reshape(377, 1)

match_similar_words= model.most_similar(positive=["good", "amazing"],negative =["worst"])
doesnt_similar_words =model.doesnt_match()

