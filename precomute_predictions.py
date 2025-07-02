import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense


import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

def process_company_data(company_data):
    try:
        # Prepare data
        data = company_data[["Close/Last"]].values
        if len(data) < 60:
            return None, None, None, None
            
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Split data
        train_size = int(len(scaled_data) * 0.8)
        if train_size < 60:
            return None, None, None, None
            
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size-60:]
        
        # Create sequences
        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        
        if len(x_train) == 0:
            return None, None, None, None
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        return scaler, train_data, test_data, train_size
        
    except Exception as e:
        print(f"Error in data processing: {str(e)}")
        return None, None, None, None

def create_and_train_model(x_train, y_train):
    try:
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model with reduced epochs for faster processing
        model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        return model
    except Exception as e:
        print(f"Error in model creation/training: {str(e)}")
        return None

def main():
    try:
        print("Loading data...")
        # Load the data
        df = pd.read_csv("data01.csv")
        df["Date"] = pd.to_datetime(df["Date"], format='mixed', dayfirst=False)
        df["Close/Last"] = df["Close/Last"].str.replace('$', '').astype(float)
        df.set_index("Date", inplace=True)
        
        # Get unique companies
        companies = df["Company"].unique()
        print(f"Found {len(companies)} companies")
        
        # Store predictions
        all_predictions = []
        
        for company in companies:
            try:
                print(f"\nProcessing {company}...")
                
                # Get company data
                company_data = df[df["Company"] == company].copy()
                company_data = company_data.sort_index(ascending=True)
                
                # Process data
                scaler, train_data, test_data, train_size = process_company_data(company_data)
                if scaler is None:
                    print(f"Skipping {company} due to insufficient data")
                    continue
                
                # Create sequences for training
                x_train, y_train = [], []
                for i in range(60, len(train_data)):
                    x_train.append(train_data[i-60:i, 0])
                    y_train.append(train_data[i, 0])
                
                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                
                # Create and train model
                model = create_and_train_model(x_train, y_train)
                if model is None:
                    print(f"Skipping {company} due to model training error")
                    continue
                
                # Prepare test data
                x_test = []
                for i in range(60, len(test_data)):
                    x_test.append(test_data[i-60:i, 0])
                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
                
                # Make predictions
                predictions = model.predict(x_test, verbose=0)
                predictions = scaler.inverse_transform(predictions)
                
                # Get actual values
                actual = company_data["Close/Last"].values[train_size:]
                
                # Store predictions
                for i in range(len(predictions)):
                    all_predictions.append({
                        'Company': company,
                        'Date': company_data.index[train_size + i],
                        'Actual': actual[i],
                        'Predicted': predictions[i][0]
                    })
                
                print(f"Successfully processed {company}")
                
            except Exception as e:
                print(f"Error processing {company}: {str(e)}")
                continue
        
        if all_predictions:
            # Create predictions DataFrame
            predictions_df = pd.DataFrame(all_predictions)
            
            # Save predictions
            predictions_df.to_csv("ai_predictions.csv", index=False)
            print("\nPredictions saved to ai_predictions.csv")
            print(f"Generated predictions for {len(predictions_df['Company'].unique())} companies")
        else:
            print("\nNo predictions were generated. Please check the data and try again.")
            
    except Exception as e:
        print(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main() 
