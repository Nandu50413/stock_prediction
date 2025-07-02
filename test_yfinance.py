import yfinance as yf
import traceback # Import traceback module

print("Starting yfinance test script...") # Added print at start

# List of symbols to test
symbols_to_test = ['TCS.NS', 'RELIANCE.NS', 'HDFCBANK.NS']

for symbol in symbols_to_test:
    print(f"Testing symbol: {symbol}")
    try:
        ticker = yf.Ticker(symbol)
        # Try fetching data for a short period, like 1 month
        data = ticker.history(period='1mo')
        if data.empty:
            print(f"No data found for {symbol} using period='1mo'")
        else:
            print(f"Successfully fetched data for {symbol}. Head of data:\n{data.head()}")
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        traceback.print_exc() # Print full traceback
    print("---") 

print("yfinance test script finished.") # Added print at end 
