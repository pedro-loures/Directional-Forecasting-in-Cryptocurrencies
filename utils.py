
import os
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

with open('data_path.txt', 'r') as file:
    datapath = file.readline().strip()
train_datapath = os.path.join(datapath, 'train.csv')
test_datapath = os.path.join(datapath, 'test.csv')


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


def accuracy_f1_score(y_pred, y_val_seq):
    y_val = y_val_seq.copy()
    filled_y_pred = y_pred.copy()
    if len(y_val_seq) == len(y_pred):       # If the lengths are the same, do nothing
        print('Lengths are the same')
        pass
    elif len(y_val_seq) > len(y_pred):      # If the target is longer than the prediction
        print('Target is longer than prediction')
        difference = len(y_val_seq) - len(y_pred)
        filled_y_pred = np.concatenate([np.zeros(difference), filled_y_pred])
    else:                                   # If the prediction is longer than the target 
        print('Prediction is longer than target')
        y_val = np.concatenate([np.zeros(1), y_val])
        
        
    
    # Calculate accuracy
    accuracy = accuracy_score(y_val, filled_y_pred)
    print(f'Validation Accuracy: {accuracy:.5f}')

    # Calculate F1 macro score
    f1_macro = f1_score(y_val, filled_y_pred, average='macro')
    print(f'Validation F1 Macro Score: {f1_macro:.5f}')

    return accuracy, f1_macro

def evaluate_model_performance(model, X_val_seq, y_val_seq):
    # Predict probabilities
    y_pred_prob = model.predict(X_val_seq)

    # Convert probabilities to binary predictions
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy_f1_score(y_pred, y_val_seq)

    return y_pred, y_pred_prob
