import pandas as pd
import numpy as np

#read data
read_data = pd.read_csv("churn-bigml-80.csv",sep=',')
print("Shape of the data is:{shape}".format(shape=read_data.shape))

#get column names
column_names = list(read_data.columns.values)
#create duplicate dataset
dup_df = read_data.drop(columns=columns_to_drop,axis=1)

print("The Shape of the dataset after dropping the column is:{s}".format(s=dup_df.shape))

#mapping the dataset
#change categorical data to numerical data
dup_df['International plan'] = dup_df["International plan"].apply({'Yes':1.0,'No':0.0}.get)
dup_df['Voice mail plan'] =  dup_df['Voice mail plan'].apply({'Yes':1.0,'No':0.0}.get)

#output label is bool we need to convert into float
dup_df['Churn'] = dup_df['Churn'] * 1.0

#define the input and output variables from the dataset
input_data = dup_df.iloc[:,:-1]
target_data = dup_df.iloc[:,-1:]

#convert all data into float
input_data['Account length'] = input_data['Account length'].astype(float)
input_data['Number vmail messages'] = input_data['Number vmail messages'].astype(float)
input_data['Total night calls'] = input_data['Total night calls'].astype(float)
input_data['Total intl calls'] = input_data['Total intl calls'].astype(float)

#split the data into train and test
train_X, train_Y = input_data.iloc[:2000], target_data.iloc[:2000]
test_X, test_Y = input_data.iloc[2000:], target_data.iloc[2000:]

#convert into numpy array
train_X,test_X = train_X.values, test_X.values
train_Y,test_Y = train_Y.values, test_Y.values


# Define the normalized function
def min_max_normalized(data):
    col_max = np.max(data, axis=0)
    col_min = np.min(data, axis=0)
    return np.divide(data - col_min, col_max - col_min)

train_X = min_max_normalized(train_X)
test_X = min_max_normalized(test_X)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(train_X, train_Y)

y_pred = logreg.predict(test_X)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(test_X, test_Y)))


actual = test_Y.T
pred = y_pred

zipped_value = zip(list(actual[0]),pred)

for act, pred in zipped_value:
    print("Actual:{}, predicted:{}".format(act, pred))

