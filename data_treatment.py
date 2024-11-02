# Data preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline





# Paths
with open('data_path.txt', 'r') as file:
    data_folder = file.read().replace('\n', '')
TRAIN_PATH = data_folder + '/train.csv'
TEST_PATH = data_folder + '/test.csv'

def compute_rsi(data, window=15):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def compute_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal
def compute_bollinger_band(data, window=15):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return upper_band, lower_band

def feature_engineering(original_df):
    # Calculate price changes

    df = original_df.copy()
    df['price_change'] = df['close'].pct_change()

    # Calculate rolling averages
    df['rolling_avg_5'] = df['close'].rolling(window=5).mean()
    df['rolling_avg_15'] = df['close'].rolling(window=15).mean()

    # Calculate exponential moving averages
    df['exp_roll_avg_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['exp_roll_avg_15'] = df['close'].ewm(span=15, adjust=False).mean()

    # Calculate volatility (standard deviation of price changes)
    df['volatility'] = df['price_change'].rolling(window=15).std()

    # Calculate taker_buy_base_volume/volume ratio
    df['Taker_Buy_Volume_Ratio'] = df['taker_buy_base_volume'] / df['volume']

    # Calculate Relative Strength Index (RSI)
    df['RSI'] = compute_rsi(df['close'])

    # Calculate MACD (Moving Average Convergence Divergence)
    df['MACD'], df['MACD_signal'] = compute_macd(df['close'])

    # Calculate Bollinger Upper and Lower
    df['bollinger_upper'], df['bollinger_lower'] = compute_bollinger_band(df['close'])

    return df



train_df = pd.read_csv(TRAIN_PATH)
target = train_df['target']
test_df = pd.read_csv(TEST_PATH)
row_id = test_df['row_id']

scaler = StandardScaler()
# Apply the feature engineering function to the train_df
treated_train_df = feature_engineering(train_df.drop(columns='target'))
columns = treated_train_df.columns
treated_train_df = scaler.fit_transform(treated_train_df)
treated_train_df = pd.DataFrame(treated_train_df, columns=columns)
treated_train_df['target'] = target 
treated_train_df = treated_train_df.dropna()


# Apply the feature engineering function to the test_df
treated_test_df = feature_engineering(test_df.drop(columns='row_id'))
columns = treated_test_df.columns
treated_test_df = scaler.fit_transform(treated_test_df)
treated_test_df = pd.DataFrame(treated_test_df, columns=columns)
treated_test_df['row_id'] = row_id
treated_test_df = treated_test_df.dropna()

# Check for null values in treated_train_df and treated_test_df
nulls_train = treated_train_df[treated_train_df.isnull().any(axis=1)]
nulls_test = treated_test_df[treated_test_df.isnull().any(axis=1)]


# Get DataFrame with only new features
new_features_train_df = treated_train_df.drop(columns=train_df.drop(columns='target').columns)
new_features_test_df = treated_test_df.drop(columns=test_df.drop(columns='row_id').columns)

# Get targets for test df
targets_for_test_df = test_df['close'] / test_df['close'].shift(1)
targets_for_test_df.index = pd.to_datetime(test_df['timestamp'].shift(-1), unit='s')
targets_for_test_df = targets_for_test_df > 1
targets_for_test_df = targets_for_test_df.astype(int).shift(-1)
targets_for_test_df.dropna(inplace=True)






# Create a pipeline with StandardScaler and TruncatedSVD
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svd', TruncatedSVD(n_components=10, random_state=42))
])

# Fit the pipeline on the train dataset
pipeline.fit(treated_train_df.drop(columns=['target', 'timestamp']))

# Transform both train and test datasets using the fitted pipeline
svd_train_features = pipeline.transform(treated_train_df.drop(columns=['target', 'timestamp']))
svd_test_features = pipeline.transform(treated_test_df.drop(columns=['row_id', 'timestamp']))

# Convert the results back to DataFrames and Use target index in svd_train_df
svd_train_df = pd.DataFrame(svd_train_features, columns=[f'svd_{i}' for i in range(1, 11)])
svd_train_df.index = treated_train_df.index
svd_train_df[['target','timestamp']] = treated_train_df[['target', 'timestamp']]
svd_test_df = pd.DataFrame(svd_test_features, columns=[f'svd_{i}' for i in range(1, 11)])
svd_test_df.index = treated_test_df.index
svd_test_df[['row_id', 'timestamp']] = treated_test_df[['row_id', 'timestamp']]

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