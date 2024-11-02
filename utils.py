
def train_val_split(dataframe):
    # Features and target
    X = train_df.drop(columns=['target'])
    y = train_df['target']

    # Split the data into training and validation sets
    X_train = X.iloc[:len(X)*4//5]
    y_train = y.iloc[:len(y)*4//5]
    X_val= X.iloc[len(X)*4//5:]
    y_val = y.iloc[len(y)*4//5:]
    
    return X_train, y_train, X_val, y_val