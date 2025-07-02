import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta

rcParams['figure.figsize'] = 20, 10

def add_technical_indicators(df):
    # Add technical indicators
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['BB_high'] = ta.volatility.bollinger_hband(df['Close'])
    df['BB_low'] = ta.volatility.bollinger_lband(df['Close'])
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    # Add price changes
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Add rolling statistics
    df['Rolling_Mean'] = df['Close'].rolling(window=20).mean()
    df['Rolling_Std'] = df['Close'].rolling(window=20).std()
    
    return df

def prepare_data(df, sequence_length=60):
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Drop NaN values
    df = df.dropna()
    
    # Select features for prediction
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 
                'BB_high', 'BB_low', 'ATR', 'Price_Change', 'Volume_Change',
                'Rolling_Mean', 'Rolling_Std']
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[features])
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict Close price
    
    return np.array(X), np.array(y), scaler, features

def create_model(input_shape):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'Root Mean Squared Error: {rmse:.2f}')
    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'R2 Score: {r2:.2f}')
    
    return mse, rmse, mae, r2

# Load and prepare data
df = pd.read_csv("data01.csv")
df["Date"] = pd.to_datetime(df["Date"], format='mixed', dayfirst=False)
df["Close/Last"] = df["Close/Last"].str.replace('$', '').astype(float)
df["Open"] = df["Open"].str.replace('$', '').astype(float)
df["High"] = df["High"].str.replace('$', '').astype(float)
df["Low"] = df["Low"].str.replace('$', '').astype(float)
df.set_index("Date", inplace=True)

# Rename columns for consistency
df = df.rename(columns={'Close/Last': 'Close'})

# Prepare data for each company
companies = df["Company"].unique()
results = {}

for company in companies:
    print(f"\nProcessing {company}...")
    
    # Get company data
    company_data = df[df["Company"] == company].copy()
    company_data = company_data.sort_index(ascending=True)
    
    # Prepare data
    X, y, scaler, features = prepare_data(company_data)
    
    if len(X) == 0:
        print(f"Insufficient data for {company}")
        continue
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train model
    model = create_model(input_shape=(X.shape[1], X.shape[2]))
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    predictions = scaler.inverse_transform(np.concatenate([predictions, np.zeros((len(predictions), len(features)-1))], axis=1))[:, 0]
    actual = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features)-1))], axis=1))[:, 0]
    
    # Evaluate predictions
    print(f"\nEvaluation metrics for {company}:")
    mse, rmse, mae, r2 = evaluate_predictions(actual, predictions)
    
    # Store results
    results[company] = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions,
        'actual': actual,
        'dates': company_data.index[train_size+60:]
    }
    
    # Plot results
    plt.figure(figsize=(15, 7))
    plt.plot(company_data.index[train_size+60:], actual, label='Actual')
    plt.plot(company_data.index[train_size+60:], predictions, label='Predicted')
    plt.title(f'Stock Price Prediction for {company}')
    plt.xlabel('Date')
    plt.ylabel('Price')
plt.legend()
plt.show()

# Save the best model for each company
for company, result in results.items():
    if result['r2'] > 0.7:  # Only save models with good performance
        model.save(f"models/{company}_model.h5")
        print(f"Saved model for {company} with R2 score: {result['r2']:.2f}")

# Save predictions to CSV
predictions_data = []
for company, result in results.items():
    for i in range(len(result['predictions'])):
        predictions_data.append({
            'Company': company,
            'Date': result['dates'][i],
            'Actual': result['actual'][i],
            'Predicted': result['predictions'][i]
        })

predictions_df = pd.DataFrame(predictions_data)
predictions_df.to_csv("ai_predictions.csv", index=False)
print("\nPredictions saved to ai_predictions.csv")
