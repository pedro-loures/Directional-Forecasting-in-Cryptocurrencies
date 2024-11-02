
import os
def train_val_split(dataframe):
    # Features and target
    X = dataframe.drop(columns=['target'])
    y = dataframe['target']

    # Split the data into training and validation sets
    X_train = X.iloc[:len(X)*4//5]
    y_train = y.iloc[:len(y)*4//5]
    X_val= X.iloc[len(X)*4//5:]
    y_val = y.iloc[len(y)*4//5:]
    
    return X_train, y_train, X_val, y_val

with open('data_path.txt', 'r') as file:
    datapath = file.readline().strip()

train_datapath = os.path.join(datapath, 'train.csv')
test_datapath = os.path.join(datapath, 'test.csv')