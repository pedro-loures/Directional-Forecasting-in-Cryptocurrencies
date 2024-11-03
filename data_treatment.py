# Data preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from utils import train_val_split
from utils import train_datapath, test_datapath



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
train_timestamp = train_df['timestamp']
test_df = pd.read_csv(TEST_PATH)
row_id = test_df['row_id']
test_timestamp = test_df['timestamp']

scaler = StandardScaler()
# Create a pipeline with StandardScaler
pipeline = Pipeline([
    ('scaler', StandardScaler()),
])

# Fit the pipeline on the train dataset


# Apply the feature engineering function to the train_df
treated_train_df = feature_engineering(train_df.drop(columns=['target', 'timestamp']))
pipeline.fit(train_df.drop(columns=['target', 'timestamp']))
columns = treated_train_df.columns
treated_train_df = pipeline.transform(treated_train_df.drop(columns=['target', 'timestamp']))
treated_test_df = pd.DataFrame(treated_train_df, columns=columns)
treated_train_df['target'] = target 
treated_train_df['timestamp'] = train_timestamp
treated_train_df = treated_train_df.dropna()
treated_train_df.to_csv('data/treated_train.csv', index=False)

# Apply the feature engineering function to the test_df
treated_test_df = feature_engineering(test_df.drop(columns=['row_id', 'timestamp']))
columns = treated_test_df.columns
treated_test_df =  pipeline.transform(treated_test_df)
treated_test_df = pd.DataFrame(treated_test_df, columns=columns)
treated_test_df['row_id'] = row_id
treated_test_df['timestamp'] = test_timestamp
treated_test_df = treated_test_df.dropna()
treated_test_df.to_csv('data/treated_test.csv', index=False)

# Check for null values in treated_train_df and treated_test_df
nulls_train = treated_train_df[treated_train_df.isnull().any(axis=1)]
nulls_test = treated_test_df[treated_test_df.isnull().any(axis=1)]


# Get DataFrame with only new features
new_features_train_df = treated_train_df.drop(columns=train_df.drop(columns='target').columns)
new_features_train_df.to_csv('data/new_features_train.csv', index=False)
new_features_test_df = treated_test_df.drop(columns=test_df.drop(columns='row_id').columns)
new_features_test_df.to_csv('data/new_features_test.csv', index=False)

# Get targets for test df
targets_for_test_df = test_df['close'] / test_df['close'].shift(1)
targets_for_test_df.index = pd.to_datetime(test_df['timestamp'].shift(-1), unit='s')
targets_for_test_df = targets_for_test_df > 1
targets_for_test_df = targets_for_test_df.astype(int).shift(-1)
targets_for_test_df.dropna(inplace=True)
targets_for_test_df.to_csv('data/targets_for_test.csv', index=True)


# ----------- Dimensionality Reduction ------------


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
svd_train_df.to_csv('data/svd_train.csv', index=False)
svd_test_df = pd.DataFrame(svd_test_features, columns=[f'svd_{i}' for i in range(1, 11)])
svd_test_df.index = treated_test_df.index
svd_test_df[['row_id', 'timestamp']] = treated_test_df[['row_id', 'timestamp']]
svd_test_df.to_csv('data/svd_test.csv', index=False)

# ------- Creating H5py model --------------
import h5py

def create_sliding_windows_batch(df, window_size, batch_size):
    for start in range(0, len(df) - window_size + 1, batch_size):
        end = min(start + batch_size, len(df) - window_size + 1)
        batch_windows = [
            df.iloc[i:i + window_size].values for i in range(start, end)
        ]
        yield np.array(batch_windows)
        
def create_h5py_file(df, window_size, batch_size, filename):
    with h5py.File(filename, 'w') as h5f:
        batch_index = 0
        for batch in create_sliding_windows_batch(df, window_size, batch_size=batch_size):
            train_images = batch.reshape(-1, window_size, df.shape[1])
            h5f.create_dataset(f'batch_{batch_index}', data=train_images)
            batch_index += 1

train_df = pd.read_csv(train_datapath)
test_df = pd.read_csv(test_datapath)


window_size = 60
batch_size = 1024

# split the train_df into train and val
X_train, y_train, X_val, y_val = train_val_split(train_df)
X_train['target'] = y_train
train_slice_df = X_train.copy()

X_val['target'] = y_val
val_slice_df = X_val.copy()

train_file = 'data/train_images.h5'
if not os.path.exists(train_file):
    create_h5py_file(train_slice_df, window_size, batch_size, train_file)

val_file = 'data/validation_images.h5'
if not os.path.exists(val_file):
    create_h5py_file(val_slice_df, window_size, batch_size, val_file)

test_file = 'data/test_images.h5'
if not os.path.exists(test_file):
    create_h5py_file(test_df, window_size, batch_size, test_file)
    
del train_df, test_df, X_train, y_train, X_val, y_val, train_slice_df, val_slice_df
