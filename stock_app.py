import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import yfinance as yf
from datetime import datetime, timedelta
import traceback
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
import os
import json
import requests
from bs4 import BeautifulSoup
import time
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor
from flask_compress import Compress

# Initialize the Dash app
app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                prevent_initial_callbacks=True,
                meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}])

server = app.server

# Configure server for better performance
server.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
server.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000  # 1 year cache for static files

# Add compression middleware
Compress(app.server)

# Add caching headers middleware
@app.server.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'public, max-age=300'  # Cache for 5 minutes
    return response


CACHE_DURATION = 300  # 5 minutes cache
MAX_CACHE_SIZE = 100  # Maximum number of cached items
stock_data_cache = {}
cache_lock = threading.Lock()

# Add cache cleanup function
def cleanup_cache():
    """Remove expired cache entries"""
    current_time = time.time()
    with cache_lock:
        expired_keys = [k for k, (timestamp, _) in stock_data_cache.items() 
                       if current_time - timestamp > CACHE_DURATION]
        for k in expired_keys:
            del stock_data_cache[k]

# Define stock symbols for Indian companies
STOCK_SYMBOLS = {
    'RELIANCE': 'RELIANCE.NS',
    'TCS': 'TCS.NS',
    'HDFCBANK': 'HDFCBANK.NS',
    'INFY': 'INFY.NS',
    'ICICIBANK': 'ICICIBANK.NS',
    'HINDUNILVR': 'HINDUNILVR.NS',
    'SBIN': 'SBIN.NS',
    'BHARTIARTL': 'BHARTIARTL.NS',
    'KOTAKBANK': 'KOTAKBANK.NS',
    'BAJFINANCE': 'BAJFINANCE.NS',
    'ASIANPAINT': 'ASIANPAINT.NS',
    'AXISBANK': 'AXISBANK.NS',
    'MARUTI': 'MARUTI.NS',
    'ULTRACEMCO': 'ULTRACEMCO.NS',
    'TITAN': 'TITAN.NS',
    'NESTLEIND': 'NESTLEIND.NS',
    'ONGC': 'ONGC.NS',
    'POWERGRID': 'POWERGRID.NS',
    'NTPC': 'NTPC.NS',
    'HCLTECH': 'HCLTECH.NS'
}

@lru_cache(maxsize=MAX_CACHE_SIZE)
def get_cached_stock_data_period(symbol, period):
    """Get stock data for a specific period with optimized caching"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval='1d')
        
        if df.empty:
            return pd.DataFrame()
        
        # Calculate only essential indicators
        if len(df) >= 50:
            df['SMA50'] = df['Close'].rolling(window=50).mean()
        else:
            df['SMA50'] = np.nan
        
        if len(df) >= 20:
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            df['RSI'] = 100 - (100 / (1 + rs))
        else:
            df['SMA20'] = np.nan
            df['RSI'] = np.nan
        
        # Trend (based on SMA20)
        if len(df) >= 20:
            df['Trend'] = 'Neutral'
            df.loc[df['Close'] > df['SMA20'], 'Trend'] = 'Up'
            df.loc[df['Close'] < df['SMA20'], 'Trend'] = 'Down'
        else:
            df['Trend'] = 'N/A'
        
        df = df.fillna(np.nan) # Fill remaining NaNs from calculations with NaN
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol} period {period}: {str(e)}")
        return pd.DataFrame()

def get_stock_data(symbol, period='2y'):
    """Get stock data with caching, respecting the period"""
    cache_key = f"{symbol}_{period}"
    current_time = time.time()
    
    with cache_lock:
        # Check both the in-memory cache and the lru_cache wrapper
        if cache_key in stock_data_cache and current_time - stock_data_cache[cache_key][0] < CACHE_DURATION:
            return stock_data_cache[cache_key][1]
    
    # If not in custom cache or expired, get data (this will use lru_cache internally)
    df = get_cached_stock_data_period(symbol, period)
    
    with cache_lock:
        stock_data_cache[cache_key] = (current_time, df)
    
    return df

def get_futures_data(symbol, period='2y'):
    """Get futures data for a symbol, respecting the period"""
    try:
        # Add .NS suffix if not present
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        # Get futures data using the specified period
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval='1d') # Use period here
        
        if df.empty:
            print(f"No futures data found for {symbol} period {period}")
            return pd.DataFrame()
        
        # Calculate basic futures metrics (ensure data exists)
        if len(df) > 1 and 'Close' in df.columns and 'Volume' in df.columns:
            df['Open Interest'] = df['Volume'] * 0.8  # Simulated open interest
            # Avoid division by zero
            df['Futures Premium'] = (df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1).replace(0, np.nan) * 100
        else:
            df['Open Interest'] = 0
            df['Futures Premium'] = np.nan
        
        df = df.fillna(np.nan)
        return df
    except Exception as e:
        print(f"Error fetching futures data for {symbol} period {period}: {str(e)}")
        return pd.DataFrame()

def get_options_data(symbol, period='2y'):
    """Get options data for a symbol, respecting the period"""
    try:
        if not symbol.endswith('.NS'):
            symbol = f"{symbol}.NS"
        
        stock = yf.Ticker(symbol)
        df = stock.history(period=period, interval='1d') # Use period here
        
        if df.empty:
            print(f"No options data found for {symbol} period {period}")
            return pd.DataFrame()
        
        # Calculate basic options metrics (ensure data exists)
        if len(df) > 20 and 'Close' in df.columns and 'Volume' in df.columns:
             # Ensure enough data points for rolling calculations
            df['Implied Volatility'] = df['Close'].pct_change().rolling(window=20).std() * 100
            # Avoid division by zero
            volume_mean = df['Volume'].rolling(window=5).mean().replace(0, np.nan)
            df['Put-Call Ratio'] = df['Volume'] / volume_mean
        else:
            df['Implied Volatility'] = np.nan
            df['Put-Call Ratio'] = np.nan
        
        df = df.fillna(np.nan)
        return df
    except Exception as e:
        print(f"Error fetching options data for {symbol} period {period}: {str(e)}")
        return pd.DataFrame()

def update_charts_async(selected_stocks, time_range):
    """Update charts asynchronously with optimized rendering"""
    price_fig = go.Figure()
    volume_fig = go.Figure()
    prediction_cards = []
    
    def process_stock(symbol):
        df = get_stock_data(symbol, time_range)
        if df.empty:
            return None
        df = calculate_technical_indicators(df)  # Add technical indicators
            
        # Optimize price trace
        price_trace = go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=f"{STOCK_SYMBOLS.get(symbol, symbol)}",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            showlegend=True
        )
        
        # Optimize moving averages
        ma20_trace = None
        if 'SMA20' in df.columns and not df['SMA20'].isna().all():
            ma20_trace = go.Scatter(
                x=df.index,
                y=df['SMA20'],
                name='20-day MA',
                line=dict(color='orange', width=1),
                showlegend=True
            )
        
        ma50_trace = None
        if 'SMA50' in df.columns and not df['SMA50'].isna().all():
            ma50_trace = go.Scatter(
                x=df.index,
                y=df['SMA50'],
                name='50-day MA',
                line=dict(color='blue', width=1),
                showlegend=True
            )
        
        # Optimize volume trace
        colors = ['#26a69a' if row['Close'] >= row['Open'] else '#ef5350' 
                 for _, row in df.iterrows()]
        
        volume_trace = go.Bar(
            x=df.index,
            y=df['Volume'],
            name=f"{STOCK_SYMBOLS.get(symbol, symbol)}",
            marker_color=colors,
            opacity=0.7,
            showlegend=True
        )
        
        # Optimize prediction card data
        current_price = df['Close'].iloc[-1] if not df.empty else None
        price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2] if len(df) > 1 else None
        price_change_pct = (price_change / df['Close'].iloc[-2]) * 100 if price_change is not None and df['Close'].iloc[-2] != 0 else 0
        current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not df['RSI'].isna().all() else None
        trend = df['Trend'].iloc[-1] if 'Trend' in df.columns and not df['Trend'].empty else 'N/A'
        current_macd = df['MACD'].iloc[-1] if 'MACD' in df.columns and not df['MACD'].isna().all() else None
        current_atr = df['ATR'].iloc[-1] if 'ATR' in df.columns and not df['ATR'].isna().all() else None
        current_mfi = df['MFI'].iloc[-1] if 'MFI' in df.columns and not df['MFI'].isna().all() else None
        
        prediction_card = html.Div(className='prediction-card', children=[
            html.H3(f"{STOCK_SYMBOLS.get(symbol, symbol)}"),
            html.Div(className='prediction-content', children=[
                html.Div(className='prediction-item', children=[
                    html.Span("Price: "),
                    html.Span(f"₹{current_price:.2f}" if current_price is not None else 'N/A')
                ]),
                html.Div(className='prediction-item', children=[
                    html.Span("Change: "),
                    html.Span(f"{price_change_pct:+.2f}%", 
                            style={'color': 'green' if price_change_pct >= 0 else 'red'}) if price_change_pct is not None else 'N/A'
                ]),
                html.Div(className='prediction-item', children=[
                    html.Span("RSI: "),
                    html.Span(f"{current_rsi:.1f}" if current_rsi is not None else 'N/A', 
                            style={'color': 'red' if current_rsi is not None and current_rsi > 70 else 'green' if current_rsi is not None and current_rsi < 30 else 'black'})
                ]),
                html.Div(className='prediction-item', children=[
                    html.Span("MACD: "),
                    html.Span(f"{current_macd:.2f}" if current_macd is not None else 'N/A')
                ]),
                html.Div(className='prediction-item', children=[
                    html.Span("ATR: "),
                    html.Span(f"{current_atr:.2f}" if current_atr is not None else 'N/A')
                ]),
                html.Div(className='prediction-item', children=[
                    html.Span("MFI: "),
                    html.Span(f"{current_mfi:.2f}" if current_mfi is not None else 'N/A')
                ]),
                html.Div(className='prediction-item', children=[
                    html.Span("Trend: "),
                    html.Span([
                        html.I(className="fas fa-arrow-up trend-icon", style={"color": "#2ecc71"}) if trend == 'Up' else
                        html.I(className="fas fa-arrow-down trend-icon", style={"color": "#e74c3c"}) if trend == 'Down' else
                        html.I(className="fas fa-minus trend-icon", style={"color": "#f1c40f"}),
                        f" {trend}"
                    ])
                ])
            ])
        ])
        
        return {
            'price_trace': price_trace,
            'ma20_trace': ma20_trace,
            'ma50_trace': ma50_trace,
            'volume_trace': volume_trace,
            'prediction_card': prediction_card
        }
    
    # Process stocks in parallel with optimized thread pool
    with ThreadPoolExecutor(max_workers=min(len(selected_stocks), 3)) as executor:
        results = list(executor.map(process_stock, selected_stocks))
    
    # Combine results efficiently
    for result in results:
        if result:
            price_fig.add_trace(result['price_trace'])
            if result['ma20_trace']:
                price_fig.add_trace(result['ma20_trace'])
            if result['ma50_trace']:
                price_fig.add_trace(result['ma50_trace'])
            volume_fig.add_trace(result['volume_trace'])
            prediction_cards.append(result['prediction_card'])
    
    # Optimize chart layouts
    price_fig.update_layout(
        title={
            'text': 'Stock Price Analysis',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#222', family='Segoe UI, Arial, sans-serif')
        },
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        template='plotly_white',
        height=340,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor='#f8fafc',
        paper_bgcolor='#fff',
        font=dict(color='#222', family='Segoe UI, Arial, sans-serif'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color='#222')
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            tickformat='%b %d'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            tickformat='.0f'
        )
    )
    
    volume_fig.update_layout(
        title={
            'text': 'Volume Analysis',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color='#222', family='Segoe UI, Arial, sans-serif')
        },
        xaxis_title='Date',
        yaxis_title='Volume',
        hovermode='x unified',
        template='plotly_white',
        height=280,
        margin=dict(l=40, r=40, t=60, b=40),
        barmode='group',
        font=dict(color='#222', family='Segoe UI, Arial, sans-serif'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color='#222')
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            tickformat='%b %d'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e7eb',
            tickformat='.2s'
        ),
        plot_bgcolor='#f8fafc',
        paper_bgcolor='#fff'
    )
    
    return price_fig, volume_fig, prediction_cards

@app.callback(
    [Output('price-chart', 'figure'),
     Output('volume-chart', 'figure'),
     Output('prediction-card', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('time-range-dropdown', 'value')]
)
def update_charts(selected_stocks, time_range):
    if not selected_stocks:
        return {}, {}, []
    
    try:
        return update_charts_async(selected_stocks, time_range)
    except Exception as e:
        print(f"Error updating charts: {str(e)}")
        traceback.print_exc()
        return {}, {}, []

# User data file
USERS_FILE = 'users.json'

# Helper functions for JSON user data
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# Check if users file exists, if not, create an empty one
if not os.path.exists(USERS_FILE):
    save_users({})

def calculate_technical_indicators(df):
    """Calculate technical indicators for buy/sell signals"""
    try:
        # Existing indicators
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
        df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
        
        # New Indicators
        
        # 1. Stochastic Oscillator
        df['Lowest_14'] = df['Low'].rolling(window=14).min()
        df['Highest_14'] = df['High'].rolling(window=14).max()
        df['%K'] = 100 * ((df['Close'] - df['Lowest_14']) / (df['Highest_14'] - df['Lowest_14']))
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # 2. Average True Range (ATR)
        df['TR'] = pd.DataFrame({
            'HL': df['High'] - df['Low'],
            'HC': abs(df['High'] - df['Close'].shift(1)),
            'LC': abs(df['Low'] - df['Close'].shift(1))
        }).max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # 3. Volume Weighted Average Price (VWAP)
        df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
        
        # 4. Money Flow Index (MFI)
        df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['Money_Flow'] = df['Typical_Price'] * df['Volume']
        df['Positive_Flow'] = df['Money_Flow'].where(df['Typical_Price'] > df['Typical_Price'].shift(1), 0)
        df['Negative_Flow'] = df['Money_Flow'].where(df['Typical_Price'] < df['Typical_Price'].shift(1), 0)
        df['MFI'] = 100 - (100 / (1 + df['Positive_Flow'].rolling(14).sum() / df['Negative_Flow'].rolling(14).sum()))
        
        # Generate Buy/Sell Signals
        df['Signal_RSI'] = 0
        df.loc[df['RSI'] < 30, 'Signal_RSI'] = 1  # Buy signal
        df.loc[df['RSI'] > 70, 'Signal_RSI'] = -1  # Sell signal
        
        df['Signal_MACD'] = 0
        df.loc[df['MACD'] > df['Signal'], 'Signal_MACD'] = 1  # Buy signal
        df.loc[df['MACD'] < df['Signal'], 'Signal_MACD'] = -1  # Sell signal
        
        df['Signal_MA'] = 0
        df.loc[df['SMA20'] > df['SMA50'], 'Signal_MA'] = 1  # Buy signal
        df.loc[df['SMA20'] < df['SMA50'], 'Signal_MA'] = -1  # Sell signal
        
        df['Signal_Stoch'] = 0
        df.loc[(df['%K'] < 20) & (df['%D'] < 20), 'Signal_Stoch'] = 1  # Buy signal
        df.loc[(df['%K'] > 80) & (df['%D'] > 80), 'Signal_Stoch'] = -1  # Sell signal
        
        df['Signal_MFI'] = 0
        df.loc[df['MFI'] < 20, 'Signal_MFI'] = 1  # Buy signal
        df.loc[df['MFI'] > 80, 'Signal_MFI'] = -1  # Sell signal
        
        # Combined Signal with more weight to volume-based indicators
        df['Combined_Signal'] = (
            df['Signal_RSI'] * 0.2 +
            df['Signal_MACD'] * 0.2 +
            df['Signal_MA'] * 0.2 +
            df['Signal_Stoch'] * 0.2 +
            df['Signal_MFI'] * 0.2
        )
        
        df = df.fillna(np.nan) # Fill remaining NaNs from calculations with NaN

        return df
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        return df

app.layout = html.Div([
    dcc.Store(id='session-store', data={'logged_in': False, 'username': None}), # Store user session info
    
    # Landing Page
    html.Div(id='landing-page', children=[
        html.Div(className='entry-container', children=[
            html.Div(className='entry-content', children=[
                html.H1("Welcome to Stock Market Dashboard", className='entry-title'),
                html.P("Your gateway to real-time market insights", className='entry-subtitle'),
                
                html.Div(className='features-grid', children=[
                    html.Div(className='feature-card', children=[
                        html.Img(src='https://cdn-icons-png.flaticon.com/512/2103/2103633.png', className='feature-icon'),
                        html.H3("Real-time Data"),
                        html.P("Get instant access to live market data")
                    ]),
                    html.Div(className='feature-card', children=[
                        html.Img(src='https://cdn-icons-png.flaticon.com/512/2103/2103633.png', className='feature-icon'),
                        html.H3("Market Analysis"),
                        html.P("Track and analyze stock performance")
                    ]),
                    html.Div(className='feature-card', children=[
                        html.Img(src='https://cdn-icons-png.flaticon.com/512/2103/2103633.png', className='feature-icon'),
                        html.H3("Price Predictions"),
                        html.P("Get AI-powered price predictions")
                    ])
                ]),
                
                # Buttons to show Login/Signup forms
                html.Div(className='landing-auth-buttons', children=[
                    html.Button('Login', id='show-login-from-landing', className='enter-button', style={'margin-right': '20px'}),
                    html.Button('Sign Up', id='show-signup-from-landing', className='enter-button')
                ])
            ])
        ])
    ]),
    
    # Authentication Pages (Login/Signup Forms)
    html.Div(id='auth-pages', style={'display': 'none'}, children=[
        # Login Page
        html.Div(id='login-page', children=[
            html.Div(className='animated-bg'),  # Animated background
            html.Div(className='auth-outer', children=[
                html.Div(className='auth-container', children=[
                    html.Img(src='https://cdn-icons-png.flaticon.com/512/3135/3135715.png', className='auth-logo'),  # Professional logo/icon
                    html.H1("Login", className='auth-title'),
                    html.Div(className='auth-form', children=[
                        dcc.Input(id='login-username', type='text', placeholder='Username', className='auth-input'),
                        dcc.Input(id='login-password', type='password', placeholder='Password', className='auth-input'),
                        html.Button('Login', id='login-button', className='auth-button'),
                        html.Div(id='login-error', className='auth-error'),
                        html.Div(className='auth-links', children=[
                            html.Span("Don't have an account? "),
                            html.A("Sign Up", id='show-signup', href='#')
                        ])
                    ])
                ])
            ])
        ]),
        
        # Signup Page
        html.Div(id='signup-page', style={'display': 'none'}, children=[
            html.Div(className='auth-container', children=[
                html.H1("Sign Up", className='auth-title'),
                html.Div(className='auth-form', children=[
                    dcc.Input(id='signup-username', type='text', placeholder='Username', className='auth-input'),
                    dcc.Input(id='signup-email', type='email', placeholder='Email', className='auth-input'),
                    dcc.Input(id='signup-password', type='password', placeholder='Password', className='auth-input'),
                    dcc.Input(id='signup-confirm-password', type='password', placeholder='Confirm Password', className='auth-input'),
                    html.Button('Sign Up', id='signup-button', className='auth-button'),
                    html.Div(id='signup-error', className='auth-error'),
                    html.Div(className='auth-links', children=[
                        html.Span("Already have an account? "),
                        html.A("Login", id='show-login', href='#')
                    ])
                ])
            ])
        ])
    ]),
    
    # Main Dashboard (initially hidden)
    html.Div(id='main-dashboard', style={'display': 'none'}, children=[
        html.Div(className='dashboard-container', children=[
            # Navbar (New)
            html.Div(className='navbar', children=[
                # App Name on the left
                html.Div(className='app-title-container', children=[
                    html.H1("Stock Market Dashboard", style={'textAlign': 'left', 'margin': '0'}), # Align left
                    html.P("Real-time Stock Market Data", style={'textAlign': 'left', 'margin': '5px 0 0 0', 'fontSize': '0.9em', 'color': '#e0e0e0'}), # Smaller font, lighter color
                ]),

                # Profile icon and dropdown on the right
                html.Div(className='profile-container', children=[
                    html.Div('👤', id='profile-icon', className='profile-icon'), # Profile Icon
                    html.Div(id='profile-dropdown', className='profile-dropdown', style={'display': 'none'}, children=[
                        html.Div(id='profile-username-display', className='profile-username'),
                        html.Button('Logout', id='logout-button', className='logout-button') # Reusing existing logout button ID
                    ])
                ])
            ]),

            # Quick Stats
            html.Div(className='stats-grid', children=[
                html.Div(className='stat-card', children=[
                    html.H2(className='stat-number', children=f"{len(STOCK_SYMBOLS)}"),
                    html.P(className='stat-label', children="Companies Tracked")
                ]),
                html.Div(className='stat-card', children=[
                    html.H2(className='stat-number', children="24/7"),
                    html.P(className='stat-label', children="Real-time Updates")
                ]),
                html.Div(className='stat-card', children=[
                    html.H2(className='stat-number', children=datetime.now().strftime("%H:%M")),
                    html.P(className='stat-label', children="Last Updated")
                ])
            ]),
            
            # Main Content with tabs
            html.Div(className='tabs-container', children=[
                dcc.Tabs(id="tabs", children=[
                    # Market Overview Tab
                    dcc.Tab(label='Market Overview', children=[
                        html.Div(className='card', children=[
                            html.H3("Select Companies"),
                            dcc.Dropdown(
                                id='stock-dropdown',
                                options=[{'label': f"{name} ({symbol})", 'value': symbol} 
                                       for symbol, name in STOCK_SYMBOLS.items()],
                                value=['RELIANCE.NS'],
                                multi=True
                            ),
                            html.Div(className='time-range', children=[
                                html.Label("Time Range"),
                                dcc.Dropdown(
                                    id='time-range-dropdown',
                                    options=[
                                        {'label': '1 Week', 'value': '5d'},
                                        {'label': '1 Month', 'value': '1mo'},
                                        {'label': '3 Months', 'value': '3mo'},
                                        {'label': '6 Months', 'value': '6mo'}
                                    ],
                                    value='1mo'
                                )
                            ])
                        ]),
                        html.Div(
                            id='prediction-card', className='card'
                        ),
                        # Stock Prices Graph
                        html.Div(className='card', children=[
                            html.H3("Stock Prices"),
                            dcc.Loading(
                                id="loading-price-chart",
                                type="circle",
                                color="#2563eb",
                                children=[dcc.Graph(id="price-chart", style={'height': '340px'})]
                            )
                        ]),
                        # Trading Volume Graph
                        html.Div(className='card', children=[
                            html.H3("Trading Volume"),
                            dcc.Loading(
                                id="loading-volume-chart",
                                type="circle",
                                color="#2563eb",
                                children=[dcc.Graph(id="volume-chart", style={'height': '280px'})]
                            )
                        ])
                    ]),
                    
                    # Technical Analysis Tab
                    dcc.Tab(label='Technical Analysis', children=[
                        html.Div(className='card', children=[
                            html.H3("Select Company"),
                            dcc.Dropdown(
                                id='technical-dropdown',
                                options=[{'label': name, 'value': symbol} 
                                       for symbol, name in STOCK_SYMBOLS.items()],
                                value='RELIANCE.NS'
                            )
                        ]),
                        html.Div(id='technical-signals', className='card'),
                        # Technical Indicators Graph 1
                        html.Div(className='card', children=[
                            html.H3("Technical Indicators"),
                            dcc.Graph(id="technical-graph-1", style={'height': '300px'})
                        ]),
                        # Technical Indicators Graph 2
                        html.Div(className='card', children=[
                            html.H3("More Indicators"),
                            dcc.Graph(id="technical-graph-2", style={'height': '300px'})
                        ])
                    ]),
                    
                    # F&O Analysis Tab
                    dcc.Tab(label='F&O Analysis', children=[
                        html.Div(className='card', children=[
                            html.H3("Select Company"),
                            dcc.Dropdown(
                                id='fno-dropdown',
                                options=[{'label': name, 'value': symbol} 
                                       for symbol, name in STOCK_SYMBOLS.items()],
                                value='RELIANCE.NS'
                            ),
                            html.Div(className='time-range', children=[
                                html.Label("Time Range"),
                                dcc.Dropdown(
                                    id='fno-time-range',
                                    options=[
                                        {'label': '1 Week', 'value': '5d'},
                                        {'label': '1 Month', 'value': '1mo'},
                                        {'label': '3 Months', 'value': '3mo'}
                                    ],
                                    value='1mo'
                                )
                            ])
                        ]),
                        
                        html.Div(className='card', children=[
                            html.H3("F&O Metrics"),
                            html.Div(id='fno-metrics', className='metrics-grid')
                        ]),
                        # F&O Volume and OI Graph
                        html.Div(className='card', children=[
                            html.H3("F&O Volume and Open Interest"),
                            dcc.Graph(id="fno-graph-1", style={'height': '300px'})
                        ]),
                        # F&O Implied Volatility and PCR Graph
                        html.Div(className='card', children=[
                            html.H3("F&O Implied Volatility and PCR"),
                            dcc.Graph(id="fno-graph-2", style={'height': '300px'})
                        ])
                    ]),

                    # Finance Calculator Tab
                    dcc.Tab(label='Finance Calculator', children=[
                        html.Div(className='card', children=[
                            html.H3("SIP Calculator", id='sip-calculator-header', className='calculator-header-toggle'),
                            html.Div(id='sip-calculator-content', style={'display': 'none'}, children=[
                                html.Div(className='calculator-inputs', children=[
                                    html.Div(className='input-group', children=[
                                        html.Label('Monthly Investment:'),
                                        dcc.Input(id='sip-monthly-investment', type='number', placeholder='e.g., 5000', min=0)
                                    ]),
                                    html.Div(className='input-group', children=[
                                        html.Label('Annual Return Rate (%):'),
                                        dcc.Input(id='sip-annual-rate', type='number', placeholder='e.g., 12', min=0, max=100, step=0.1)
                                    ]),
                                    html.Div(className='input-group', children=[
                                        html.Label('Investment Period (Years):'),
                                        dcc.Input(id='sip-period', type='number', placeholder='e.g., 10', min=1)
                                    ])
                                ]),
                                html.Button('Calculate SIP', id='calculate-sip', n_clicks=0, className='calculator-button'),
                                html.Div(id='sip-result', className='calculator-result')
                            ])
                        ]),
                        
                        html.Div(className='card', children=[
                            html.H3("Loan Calculator", id='loan-calculator-header', className='calculator-header-toggle'),
                            html.Div(id='loan-calculator-content', style={'display': 'none'}, children=[
                                 html.Div(className='calculator-inputs', children=[
                                     html.Div(className='input-group', children=[
                                        html.Label('Loan Amount:'),
                                        dcc.Input(id='loan-amount', type='number', placeholder='e.g., 100000', min=0)
                                    ]),
                                     html.Div(className='input-group', children=[
                                        html.Label('Annual Interest Rate (%):'),
                                        dcc.Input(id='loan-annual-rate', type='number', placeholder='e.g., 8', min=0, max=100, step=0.01)
                                    ]),
                                     html.Div(className='input-group', children=[
                                        html.Label('Loan Tenure (Years):'),
                                        dcc.Input(id='loan-tenure', type='number', placeholder='e.g., 15', min=1)
                                    ])
                                ]),
                                html.Button('Calculate Loan', id='calculate-loan', n_clicks=0, className='calculator-button'),
                                html.Div(id='loan-result', className='calculator-result')
                            ])
                        ]),

                         html.Div(className='card', children=[
                            html.H3("Stock Investment Return Calculator", id='stock-return-calculator-header', className='calculator-header-toggle'),
                            html.Div(id='stock-return-calculator-content', style={'display': 'none'}, children=[
                                 html.Div(className='calculator-inputs', children=[
                                     html.Div(className='input-group', children=[
                                        html.Label('Buy Price per Share:'),
                                        dcc.Input(id='stock-buy-price', type='number', placeholder='e.g., 1000', min=0)
                                    ]),
                                     html.Div(className='input-group', children=[
                                        html.Label('Sell Price per Share:'),
                                        dcc.Input(id='stock-sell-price', type='number', placeholder='e.g., 1200', min=0)
                                    ]),
                                     html.Div(className='input-group', children=[
                                        html.Label('Number of Shares:'),
                                        dcc.Input(id='stock-shares', type='number', placeholder='e.g., 10', min=1)
                                    ]),
                                     html.Div(className='input-group', children=[
                                        html.Label('Transaction Costs (% of total value):'),
                                        dcc.Input(id='stock-costs', type='number', placeholder='e.g., 0.5', min=0, step=0.01)
                                    ])
                                ]),
                                html.Button('Calculate Stock Return', id='calculate-stock-return', n_clicks=0, className='calculator-button'),
                                html.Div(id='stock-return-result', className='calculator-result')
                            ])
                        ])
                    ]) # Closing bracket for dcc.Tabs children list

                ]) # Closing parenthesis for dcc.Tabs
            ]) # Closing parenthesis for html.Div (tabs-container)
        ]) # Closing parenthesis for html.Div (dashboard-container)
    ]) # Closing parenthesis for html.Div (main-dashboard)
]) # Closing parenthesis for app.layout html.Div

# Callback to control page visibility based on authentication state and button clicks
@app.callback(
    [Output('landing-page', 'style'),
     Output('auth-pages', 'style'),
     Output('main-dashboard', 'style'),
     Output('login-page', 'style'),
     Output('signup-page', 'style')],
    [Input('session-store', 'data'),
     Input('show-login-from-landing', 'n_clicks'),
     Input('show-signup-from-landing', 'n_clicks'),
     Input('show-login', 'n_clicks'),
     Input('show-signup', 'n_clicks')],
    [State('login-error', 'children'), # Keep forms visible if there are errors after auth attempt
     State('signup-error', 'children')]
)
def update_page_visibility(session_data, login_from_landing_clicks, signup_from_landing_clicks, show_login_clicks, show_signup_clicks, login_error, signup_error):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    # If logged in, show dashboard
    if session_data and session_data.get('logged_in'):
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}

    # If not logged in, handle page transitions
    if triggered_id == 'show-login-from-landing' or triggered_id == 'show-login':
        return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
    elif triggered_id == 'show-signup-from-landing' or triggered_id == 'show-signup':
         return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}

    # If auth failed, stay on the respective auth page
    if triggered_id in ['login-button', 'signup-button']:
         if triggered_id == 'login-button' and login_error:
               return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}
         elif triggered_id == 'signup-button' and signup_error:
                return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'block'}
         elif triggered_id == 'login-button' and not login_error: # Successful login handled above
             pass # Should not reach here due to logged_in check
         elif triggered_id == 'signup-button' and not signup_error: # Successful signup, show login
             return {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}

    # Default: show landing page if not logged in and no specific auth trigger
    return {'display': 'block'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}

# Auth callbacks (modified to return errors and update session-store)
@app.callback(
    [Output('login-error', 'children'),
     Output('signup-error', 'children'),
     Output('session-store', 'data')], # Added session-store output
    [Input('login-button', 'n_clicks'),
     Input('signup-button', 'n_clicks')],
    [State('login-username', 'value'),
     State('login-password', 'value'),
     State('signup-username', 'value'),
     State('signup-email', 'value'),
     State('signup-password', 'value'),
     State('signup-confirm-password', 'value'),
     State('session-store', 'data')] # Added session-store state
)
def handle_auth(login_click, signup_click,
                login_username, login_password,
                signup_username, signup_email, signup_password, signup_confirm_password, session_data):
    print("DEBUG: handle_auth called")
    print(f"login_click={login_click}, signup_click={signup_click}")
    print(f"login_username={login_username}, login_password={'***' if login_password else None}")
    print(f"signup_username={signup_username}, signup_email={signup_email}, signup_password={'***' if signup_password else None}, signup_confirm_password={'***' if signup_confirm_password else None}")
    print(f"session_data={session_data}")
    ctx = dash.callback_context
    if not ctx.triggered:
        print("DEBUG: No callback triggered")
        return '', '', session_data # No change if no trigger
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"DEBUG: button_id={button_id}")
    
    login_error = ''
    signup_error = ''
    new_session_data = session_data

    if button_id == 'login-button':
        print("DEBUG: Login button pressed")
        if login_username and login_password:
            users = load_users()
            print(f"DEBUG: Loaded users: {users}")
            if login_username in users:
                print("DEBUG: Username found in users.json")
                if check_password_hash(users[login_username]['password'], login_password):
                    print("DEBUG: Password hash check passed")
                    # Update last login
                    users[login_username]['last_login'] = datetime.now().isoformat()
                    save_users(users)
                    new_session_data = {'logged_in': True, 'username': login_username}
                else:
                    print("DEBUG: Password hash check failed")
                    login_error = 'Invalid username or password'
            else:
                print("DEBUG: Username not found in users.json")
                login_error = 'Invalid username or password'
        else:
             print("DEBUG: Username or password missing")
             login_error = 'Please enter username and password'

    elif button_id == 'signup-button':
        print("DEBUG: Signup button pressed")
        if all([signup_username, signup_email, signup_password, signup_confirm_password]):
            if signup_password != signup_confirm_password:
                print("DEBUG: Passwords do not match")
                signup_error = 'Passwords do not match'
            else:
                users = load_users()
                print(f"DEBUG: Loaded users: {users}")
                if signup_username in users:
                    print("DEBUG: Username already exists")
                    signup_error = 'Username already exists'
                else:
                    try:
                        hashed_password = generate_password_hash(signup_password)
                        users[signup_username] = {
                            'email': signup_email,
                            'password': hashed_password,
                            'last_login': datetime.now().isoformat() # Set initial login on signup
                        }
                        save_users(users)
                        print("DEBUG: Signup successful, user saved")
                        # Successful signup, will redirect to login via update_page_visibility callback
                        signup_error = 'Signup successful! Please login.'
                    except Exception as e:
                        print(f"DEBUG: Exception during signup: {e}")
                        signup_error = f'Signup failed: {e}'
        else:
            print("DEBUG: Signup fields missing")
            signup_error = 'Please fill in all fields'
            
    print(f"DEBUG: Returning login_error={login_error}, signup_error={signup_error}, new_session_data={new_session_data}")
    return login_error, signup_error, new_session_data

# Callback for Logout button (redirect to landing page)
@app.callback(
    Output('session-store', 'data', allow_duplicate=True), # Use allow_duplicate=True 
    Input('logout-button', 'n_clicks'),
    prevent_initial_call=True
)
def logout(n_clicks):
    if n_clicks:
        return {'logged_in': False, 'username': None} # Clear session data
    return dash.no_update # Don't update if button not clicked

# New callback to toggle profile dropdown visibility and update username display
@app.callback(
    [Output('profile-dropdown', 'style'),
     Output('profile-username-display', 'children')],
    [Input('profile-icon', 'n_clicks')],
    [State('profile-dropdown', 'style'),
     State('session-store', 'data')],
    prevent_initial_call=True
)
def toggle_profile_dropdown(n_clicks, current_style, session_data):
    if n_clicks:
        if current_style and current_style.get('display') == 'block':
            # Hide dropdown
            return {'display': 'none'}, ''
        else:
            # Show dropdown and update username
            username = session_data.get('username', 'Guest')
            return {'display': 'block'}, f"Logged in as: {username}"
    return dash.no_update, dash.no_update # No change if icon not clicked

# Callback for Technical Analysis Tab
@app.callback(
    [Output('technical-signals', 'children'),
     Output('technical-graph-1', 'figure'),
     Output('technical-graph-2', 'figure')],
    [Input('technical-dropdown', 'value')]
)
def update_technical_analysis(selected_stock):
    if not selected_stock:
        print("update_technical_analysis: No stock selected.")
        return [], {}, {}

    print(f"update_technical_analysis: Fetching data for {selected_stock}")
    df = get_stock_data(selected_stock, '6mo')

    if df.empty:
        print(f"update_technical_analysis: No data found for {selected_stock}")
        return [html.Div(f"No technical data available for {STOCK_SYMBOLS.get(selected_stock, selected_stock)}.", style={'textAlign': 'center', 'color': 'red'})], {}, {}

    try:
        print(f"update_technical_analysis: Data fetched successfully for {selected_stock}. Rows: {len(df)}")

        df = calculate_technical_indicators(df)
        
        if df.empty:
            print(f"update_technical_analysis: Error calculating indicators for {selected_stock}.")
            return [html.Div(f"Error calculating technical indicators for {STOCK_SYMBOLS.get(selected_stock, selected_stock)}.", style={'textAlign': 'center', 'color': 'red'})], {}, {}

        # --- Create Technical Indicator Graphs ---
        
        # Graph 1: Price, SMAs, Bollinger Bands
        fig_tech1 = go.Figure()

        # Candlestick trace
        fig_tech1.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a', # Green for increasing
            decreasing_line_color='#ef5350', # Red for decreasing
        ))

        # SMA traces
        if 'SMA20' in df.columns and not df['SMA20'].isna().all():
            fig_tech1.add_trace(go.Scatter(
                x=df.index,
                y=df['SMA20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='#ff9800', width=1) # Orange
            ))
        if 'SMA50' in df.columns and not df['SMA50'].isna().all():
            fig_tech1.add_trace(go.Scatter(
                x=df.index,
                y=df['SMA50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='#2196f3', width=1) # Blue
            ))
        if 'SMA200' in df.columns and not df['SMA200'].isna().all():
             fig_tech1.add_trace(go.Scatter(
                x=df.index,
                y=df['SMA200'],
                mode='lines',
                name='SMA 200',
                line=dict(color='purple', width=1)
            ))

        if 'BB_upper' in df.columns and 'BB_middle' in df.columns and 'BB_lower' in df.columns and \
           not df['BB_upper'].isna().all() and not df['BB_middle'].isna().all() and not df['BB_lower'].isna().all():
            fig_tech1.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='red', width=1, dash='dash')
            ))
            fig_tech1.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_middle'],
                mode='lines',
                name='BB Middle',
                line=dict(color='gray', width=1)
            ))
            fig_tech1.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='red', width=1, dash='dash')
            ))
            
        fig_tech1.update_layout(
            title={
                'text': f'{STOCK_SYMBOLS.get(selected_stock, selected_stock)} Price and Moving Averages',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18)
            },
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            hovermode='x unified',
            template='plotly_white',
            height=350,
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_rangeslider_visible=False
        )

        # Graph 2: RSI, MACD, Stochastic, MFI
        fig_tech2 = go.Figure()

        # RSI trace
        if 'RSI' in df.columns and not df['RSI'].isna().all():
            fig_tech2.add_trace(go.Scatter(
                x=df.index,
                y=df['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='#4caf50', width=1) # Green
            ))
            # RSI overbought/oversold lines
            fig_tech2.add_hline(y=70, annotation_text="Overbought", line_dash="dash", line_color="#f44336", annotation_position="bottom right") # Red
            fig_tech2.add_hline(y=30, annotation_text="Oversold", line_dash="dash", line_color="#4caf50", annotation_position="top right") # Green

        # MACD traces
        if 'MACD' in df.columns and 'Signal' in df.columns and \
           not df['MACD'].isna().all() and not df['Signal'].isna().all():
            fig_tech2.add_trace(go.Scatter(
                x=df.index,
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='#2196f3', width=1) # Blue
            ))
            fig_tech2.add_trace(go.Scatter(
                x=df.index,
                y=df['Signal'],
                mode='lines',
                name='Signal Line',
                line=dict(color='#ff5722', width=1) # Deep Orange
            ))
             # MACD Histogram (Optional but helpful)
            macd_histogram_colors = ['#4caf50' if val >= 0 else '#f44336' for val in (df['MACD'] - df['Signal'])] # Green/Red
            fig_tech2.add_trace(go.Bar(
                x=df.index,
                y=(df['MACD'] - df['Signal']),
                name='MACD Hist',
                marker_color=macd_histogram_colors,
                opacity=0.6
            ))

        # Stochastic Oscillator traces
        if '%K' in df.columns and '%D' in df.columns and \
           not df['%K'].isna().all() and not df['%D'].isna().all():
            fig_tech2.add_trace(go.Scatter(
                x=df.index,
                y=df['%K'],
                mode='lines',
                name='%K',
                line=dict(color='#9c27b0', width=1) # Purple
            ))
            fig_tech2.add_trace(go.Scatter(
                x=df.index,
                y=df['%D'],
                mode='lines',
                name='%D',
                line=dict(color='#ffc107', width=1) # Amber
            ))
            # Stochastic overbought/oversold lines
            fig_tech2.add_hline(y=80, annotation_text="Overbought", line_dash="dash", line_color="#f44336", annotation_position="bottom right") # Red
            fig_tech2.add_hline(y=20, annotation_text="Oversold", line_dash="dash", line_color="#4caf50", annotation_position="top right") # Green
            
         # MFI trace
        if 'MFI' in df.columns and not df['MFI'].isna().all():
             fig_tech2.add_trace(go.Scatter(
                x=df.index,
                y=df['MFI'],
                mode='lines',
                name='MFI',
                line=dict(color='#795548', width=1) # Brown
            ))
             # MFI overbought/oversold lines
             fig_tech2.add_hline(y=80, annotation_text="Overbought (MFI)", line_dash="dot", line_color="#f44336", annotation_position="top left") # Red
             fig_tech2.add_hline(y=20, annotation_text="Oversold (MFI)", line_dash="dot", line_color="#4caf50", annotation_position="bottom left") # Green

        # Update layout for Graph 2
        fig_tech2.update_layout(
            title={
                'text': f'{STOCK_SYMBOLS.get(selected_stock, selected_stock)} Oscillators and Momentum',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18)
            },
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            height=400,
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_rangeslider_visible=False
        )
        
        current_price = df['Close'].iloc[-1] if not df.empty else None
        current_rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns and not df['RSI'].empty else None
        trend = df['Trend'].iloc[-1] if 'Trend' in df.columns and not df['Trend'].empty else 'N/A'
        current_macd = df['MACD'].iloc[-1] if 'MACD' in df.columns and not df['MACD'].isna().all() else None
        current_atr = df['ATR'].iloc[-1] if 'ATR' in df.columns and not df['ATR'].isna().all() else None
        current_mfi = df['MFI'].iloc[-1] if 'MFI' in df.columns and not df['MFI'].isna().all() else None
        
        signals_summary = html.Div(className='signals-summary', children=[
            html.H3(f"{STOCK_SYMBOLS.get(selected_stock, selected_stock)} Technical Summary"),
            html.Div(className='signal-card', children=[
                html.Div(className='signal-item', children=[
                    html.Span("Current Price: "),
                    html.Span(f"₹{current_price:.2f}" if current_price is not None else 'N/A')
                ]),
                html.Div(className='signal-item', children=[
                    html.Span("RSI: "),
                    html.Span(f"{current_rsi:.1f}" if current_rsi is not None and not np.isnan(current_rsi) else 'N/A', 
                            style={'color': 'red' if current_rsi is not None and not np.isnan(current_rsi) and current_rsi > 70 else 'green' if current_rsi is not None and not np.isnan(current_rsi) and current_rsi < 30 else 'black'})
                ]),
                html.Div(className='signal-item', children=[
                    html.Span("Trend: "),
                    html.Span([
                        html.I(className="fas fa-arrow-up trend-icon", style={"color": "#2ecc71"}) if trend == 'Up' else
                        html.I(className="fas fa-arrow-down trend-icon", style={"color": "#e74c3c"}) if trend == 'Down' else
                        html.I(className="fas fa-minus trend-icon", style={"color": "#f1c40f"}),
                        f" {trend}"
                    ])
                ]),
                html.Div(className='signal-item', children=[
                    html.Span("MACD: "),
                    html.Span(f"{current_macd:.2f}" if current_macd is not None else 'N/A')
                ]),
                html.Div(className='signal-item', children=[
                    html.Span("ATR: "),
                    html.Span(f"{current_atr:.2f}" if current_atr is not None else 'N/A')
                ]),
                html.Div(className='signal-item', children=[
                    html.Span("MFI: "),
                    html.Span(f"{current_mfi:.2f}" if current_mfi is not None else 'N/A')
                ])
            ]),
            html.Div(className='signal-explanation', children=[
                html.H4("How to Read the Signals:"),
                html.Ul([
                    html.Li("RSI above 70 suggests overbought conditions"),
                    html.Li("RSI below 30 suggests oversold conditions"),
                    html.Li("Price above 20-day MA suggests uptrend"),
                    html.Li("Price below 20-day MA suggests downtrend")
                ])
            ])
        ])

        print(f"update_technical_analysis: Returning summary and graphs for {selected_stock}")
        return [signals_summary], fig_tech1, fig_tech2

    except Exception as e:
        print(f"update_technical_analysis: Error updating technical analysis for {selected_stock}: {str(e)}")
        traceback.print_exc()
        return [html.Div(f"Error loading technical analysis for {STOCK_SYMBOLS.get(selected_stock, selected_stock)}.", style={'textAlign': 'center', 'color': 'red'})], {}, {}

# New callback for F&O analysis
@app.callback(
    [Output('fno-metrics', 'children'),
     Output('fno-graph-1', 'figure'),
     Output('fno-graph-2', 'figure')],
    [Input('fno-dropdown', 'value'),
     Input('fno-time-range', 'value')]
)
def update_fno_analysis(selected_stock, time_range):
    if not selected_stock:
        print("update_fno_analysis: No stock selected.")
        # Return empty outputs if no stock is selected
        return [], {}, {}
    
    print(f"update_fno_analysis: Fetching data for {selected_stock}, period {time_range}")
    
    try:
        # Get futures data (using the selected time_range)
        futures_df = get_futures_data(selected_stock, time_range)
        # Get options data (using the selected time_range)
        options_df = get_options_data(selected_stock, time_range)
        
        # Create metrics cards (re-using existing logic)
        metrics = []
        # Ensure the dataframes are not empty before accessing .iloc[-1]
        if not futures_df.empty and not options_df.empty:
            try:
                current_price = futures_df['Close'].iloc[-1] if 'Close' in futures_df.columns and not futures_df['Close'].empty else None
                futures_premium = futures_df['Futures Premium'].iloc[-1] if 'Futures Premium' in futures_df.columns and not futures_df['Futures Premium'].empty else None
                implied_vol = options_df['Implied Volatility'].iloc[-1] if 'Implied Volatility' in options_df.columns and not options_df['Implied Volatility'].empty else None
                put_call_ratio = options_df['Put-Call Ratio'].iloc[-1] if 'Put-Call Ratio' in options_df.columns and not options_df['Put-Call Ratio'].empty else None
                open_interest = futures_df['Open Interest'].iloc[-1] if 'Open Interest' in futures_df.columns and not futures_df['Open Interest'].empty else None
                
                metrics = [
                    html.Div(className='metric-card', children=[
                        html.H4("Current Price"),
                        html.P(f"₹{current_price:.2f}" if current_price is not None else 'N/A')
                    ]),
                    html.Div(className='metric-card', children=[
                        html.H4("Futures Premium"),
                        html.P(f"{futures_premium:.2f}%" if futures_premium is not None else 'N/A')
                    ]),
                    html.Div(className='metric-card', children=[
                        html.H4("Implied Volatility"),
                        html.P(f"{implied_vol:.2f}%" if implied_vol is not None else 'N/A')
                    ]),
                    html.Div(className='metric-card', children=[
                        html.H4("Put-Call Ratio"),
                        html.P(f"{put_call_ratio:.2f}" if put_call_ratio is not None else 'N/A')
                    ]),
                    html.Div(className='metric-card', children=[
                        html.H4("Open Interest"),
                        html.P(f"{open_interest:,.0f}" if open_interest is not None else 'N/A')
                    ])
                ]
            except IndexError: 
                 print("update_fno_analysis: IndexError: Dataframe is empty, cannot access latest metrics.")
                 metrics = [html.Div("No F&O data available for the selected period.")]
        else:
             print("update_fno_analysis: Futures or Options dataframe is empty.")
             metrics = [html.Div("No F&O data available for the selected period.")]

        # --- Create F&O Graphs ---
        
        # Graph 1: Volume and Open Interest
        fig_fno1 = go.Figure()
        
        if 'Volume' in futures_df.columns and not futures_df['Volume'].isna().all():
             fig_fno1.add_trace(go.Bar(
                x=futures_df.index,
                y=futures_df['Volume'],
                name='Volume',
                marker_color='#2196f3', # Blue
                opacity=0.7
            ))

        if 'Open Interest' in futures_df.columns and not futures_df['Open Interest'].isna().all():
            fig_fno1.add_trace(go.Scatter(
                x=futures_df.index,
                y=futures_df['Open Interest'],
                mode='lines',
                name='Open Interest',
                yaxis='y2', # Use secondary y-axis
                line=dict(color='#ff9800', width=2) # Orange
            ))

        fig_fno1.update_layout(
            title={
                'text': f'{STOCK_SYMBOLS.get(selected_stock, selected_stock)} F&O Volume and Open Interest',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18)
            },
            xaxis_title='Date',
            yaxis_title='Volume',
            hovermode='x unified',
            template='plotly_white',
            height=350, # Adjusted height
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_rangeslider_visible=False,
            yaxis2=dict( # Secondary y-axis for Open Interest
                title='Open Interest',
                overlaying='y',
                side='right'
            )
        )

        # Graph 2: Implied Volatility and Put-Call Ratio
        fig_fno2 = go.Figure()

        if 'Implied Volatility' in options_df.columns and not options_df['Implied Volatility'].isna().all():
            fig_fno2.add_trace(go.Scatter(
                x=options_df.index,
                y=options_df['Implied Volatility'],
                mode='lines',
                name='Implied Volatility',
                line=dict(color='#4caf50', width=2) # Green
            ))

        if 'Put-Call Ratio' in options_df.columns and not options_df['Put-Call Ratio'].isna().all():
            fig_fno2.add_trace(go.Scatter(
                x=options_df.index,
                y=options_df['Put-Call Ratio'],
                mode='lines',
                name='Put-Call Ratio',
                yaxis='y2', # Use secondary y-axis
                line=dict(color='#f44336', width=2) # Red
            ))
            
        fig_fno2.update_layout(
            title={
                'text': f'{STOCK_SYMBOLS.get(selected_stock, selected_stock)} Implied Volatility and Put-Call Ratio',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=18)
            },
            xaxis_title='Date',
            yaxis_title='Implied Volatility (%)',
            hovermode='x unified',
            template='plotly_white',
            height=350, # Adjusted height
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_rangeslider_visible=False,
            yaxis2=dict( # Secondary y-axis for Put-Call Ratio
                title='Put-Call Ratio',
                overlaying='y',
                side='right'
            )
        )

        print(f"update_fno_analysis: Returning metrics and graphs for {selected_stock}")
        return [html.Div(metrics, className='metrics-container')], fig_fno1, fig_fno2

    except Exception as e:
        print(f"update_fno_analysis: Error updating F&O analysis for {selected_stock}: {str(e)}")
        traceback.print_exc()
        # Return empty outputs in case of an exception
        return [html.Div(f"Error loading F&O analysis for {STOCK_SYMBOLS.get(selected_stock, selected_stock)}.", style={'textAlign': 'center', 'color': 'red'})], {}, {}

# Callback for SIP Calculator
@app.callback(
    Output('sip-result', 'children'),
    Input('calculate-sip', 'n_clicks'),
    [State('sip-monthly-investment', 'value'),
     State('sip-annual-rate', 'value'),
     State('sip-period', 'value')],
    prevent_initial_call=True
)
def calculate_sip(n_clicks, monthly_investment, annual_rate, period):
    if n_clicks and monthly_investment is not None and annual_rate is not None and period is not None:
        try:
            # Convert annual rate to monthly and percentage to decimal
            monthly_rate = (annual_rate / 12) / 100
            number_of_months = period * 12
            
            # Formula for Future Value of a series (SIP)
            # FV = P * [((1 + r)^n - 1) / r] * (1 + r)
            # P = monthly_investment, r = monthly_rate, n = number_of_months
            
            if monthly_rate == 0:
                 # Handle zero interest rate case
                 future_value = monthly_investment * number_of_months
            else:
                future_value = monthly_investment * (((1 + monthly_rate)**number_of_months - 1) / monthly_rate) * (1 + monthly_rate)

            invested_amount = monthly_investment * number_of_months
            estimated_return = future_value - invested_amount

            return html.Div([
                html.P(f"Invested Amount: ₹{invested_amount:,.2f}"),
                html.P(f"Estimated Return: ₹{estimated_return:,.2f}"),
                html.P(f"Total Value: ₹{future_value:,.2f}")
            ])
        except Exception as e:
            return html.Div(f"Error calculating SIP: {e}")
    elif n_clicks:
        return html.Div("Please enter all values for SIP calculation.")
    return dash.no_update

# Callback for Loan Calculator
@app.callback(
    Output('loan-result', 'children'),
    Input('calculate-loan', 'n_clicks'),
    [State('loan-amount', 'value'),
     State('loan-annual-rate', 'value'),
     State('loan-tenure', 'value')],
    prevent_initial_call=True
)
def calculate_loan(n_clicks, loan_amount, annual_rate, tenure):
    if n_clicks and loan_amount is not None and annual_rate is not None and tenure is not None:
        try:
            # Convert annual rate to monthly and percentage to decimal
            monthly_rate = (annual_rate / 12) / 100
            number_of_months = tenure * 12

            # Formula for EMI
            # EMI = P * r * (1 + r)^n / ((1 + r)^n - 1)
            # P = loan_amount, r = monthly_rate, n = number_of_months

            if monthly_rate == 0:
                 # Handle zero interest rate case (shouldn't happen for typical loans, but for completeness)
                 emi = loan_amount / number_of_months if number_of_months > 0 else 0
            else:
                emi = loan_amount * monthly_rate * (1 + monthly_rate)**number_of_months / (((1 + monthly_rate)**number_of_months) - 1)

            total_payable = emi * number_of_months
            total_interest = total_payable - loan_amount

            return html.Div([
                html.P(f"Estimated Monthly EMI: ₹{emi:,.2f}"),
                html.P(f"Total Interest Payable: ₹{total_interest:,.2f}"),
                html.P(f"Total Amount Payable: ₹{total_payable:,.2f}")
            ])
        except Exception as e:
            return html.Div(f"Error calculating Loan: {e}")
    elif n_clicks:
        return html.Div("Please enter all values for Loan calculation.")
    return dash.no_update

# Callback for Stock Investment Return Calculator
@app.callback(
    Output('stock-return-result', 'children'),
    Input('calculate-stock-return', 'n_clicks'),
    [State('stock-buy-price', 'value'),
     State('stock-sell-price', 'value'),
     State('stock-shares', 'value'),
     State('stock-costs', 'value')],
    prevent_initial_call=True
)
def calculate_stock_return(n_clicks, buy_price, sell_price, shares, costs_percentage):
    if n_clicks and buy_price is not None and sell_price is not None and shares is not None and costs_percentage is not None:
        try:
            total_buy_value = buy_price * shares
            total_sell_value = sell_price * shares
            
            # Calculate transaction costs (applied to both buy and sell total value)
            total_costs = (total_buy_value + total_sell_value) * (costs_percentage / 100)
            
            net_profit_loss = (total_sell_value - total_buy_value) - total_costs
            
            return_percentage = (net_profit_loss / total_buy_value) * 100 if total_buy_value > 0 else 0

            return html.Div([
                html.P(f"Total Investment: ₹{total_buy_value:,.2f}"),
                html.P(f"Total Selling Value: ₹{total_sell_value:,.2f}"),
                html.P(f"Total Transaction Costs: ₹{total_costs:,.2f}"),
                html.P(f"Net Profit/Loss: ₹{net_profit_loss:,.2f}"),
                html.P(f"Return Percentage: {return_percentage:,.2f}%")
            ])
        except Exception as e:
            return html.Div(f"Error calculating Stock Return: {e}")
    elif n_clicks:
         return html.Div("Please enter all values for Stock Return calculation.")
    return dash.no_update

# Callbacks to toggle calculator visibility
@app.callback(
    Output('sip-calculator-content', 'style'),
    Input('sip-calculator-header', 'n_clicks'),
    State('sip-calculator-content', 'style'),
    prevent_initial_call=True
)
def toggle_sip_calculator(n_clicks, current_style):
    if n_clicks:
        if current_style and current_style.get('display') == 'block':
            return {'display': 'none'}
        else:
            return {'display': 'block'}
    return dash.no_update

@app.callback(
    Output('loan-calculator-content', 'style'),
    Input('loan-calculator-header', 'n_clicks'),
    State('loan-calculator-content', 'style'),
    prevent_initial_call=True
)
def toggle_loan_calculator(n_clicks, current_style):
    if n_clicks:
        if current_style and current_style.get('display') == 'block':
            return {'display': 'none'}
        else:
            return {'display': 'block'}
    return dash.no_update

@app.callback(
    Output('stock-return-calculator-content', 'style'),
    Input('stock-return-calculator-header', 'n_clicks'),
    State('stock-return-calculator-content', 'style'),
    prevent_initial_call=True
)
def toggle_stock_return_calculator(n_clicks, current_style):
    if n_clicks:
        if current_style and current_style.get('display') == 'block':
            return {'display': 'none'}
        else:
            return {'display': 'block'}
    return dash.no_update

# Add CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Stock Market Dashboard</title>
        {%favicon%}
        {%css%}
        <!-- FontAwesome for icons -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            body {
                font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
                background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
                min-height: 100vh;
                margin: 0;
                padding: 0;
                color: #222;
                overflow-x: hidden;
            }
            .animated-bg {
                position: fixed;
                top: 0; left: 0; width: 100vw; height: 100vh;
                z-index: 0;
                background: linear-gradient(120deg, #a1c4fd 0%, #c2e9fb 100%);
                animation: gradientMove 8s ease-in-out infinite alternate;
            }
            @keyframes gradientMove {
                0% { background-position: 0% 50%; }
                100% { background-position: 100% 50%; }
            }
            .auth-outer {
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                z-index: 1;
            }
            .auth-container {
                max-width: 400px;
                width: 100%;
                background: rgba(255,255,255,0.85);
                border-radius: 28px;
                box-shadow: 0 8px 40px 0 rgba(30, 60, 114, 0.18), 0 1.5px 8px 0 rgba(30,60,114,0.10);
                padding: 56px 36px 40px 36px;
                margin: 0 20px;
                text-align: center;
                backdrop-filter: blur(8px);
                border: 1.5px solid rgba(180,200,255,0.18);
                position: relative;
            }
            .auth-logo {
                width: 64px;
                height: 64px;
                margin-bottom: 18px;
                filter: drop-shadow(0 2px 8px #a1c4fd88);
            }
            .auth-title {
                font-size: 2.3em;
                margin-bottom: 24px;
                font-weight: 700;
                color: #2563eb;
                letter-spacing: 1px;
            }
            .auth-form {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .auth-input {
                padding: 14px 18px;
                border-radius: 10px;
                border: 1.5px solid #d1d5db;
                font-size: 1.13em;
                margin-bottom: 0;
                background: #f8fafc;
                transition: border 0.2s, box-shadow 0.2s;
                box-shadow: 0 1.5px 8px 0 rgba(30,60,114,0.04);
            }
            .auth-input:focus {
                border: 1.5px solid #2563eb;
                outline: none;
                box-shadow: 0 2px 12px 0 #a1c4fd44;
            }
            .auth-button {
                background: linear-gradient(90deg, #2563eb 0%, #1e40af 100%);
                color: #fff;
                border: none;
                padding: 15px 0;
                font-size: 1.18em;
                border-radius: 10px;
                cursor: pointer;
                font-weight: 600;
                margin-top: 10px;
                box-shadow: 0 4px 15px rgba(30, 60, 114, 0.13);
                text-transform: uppercase;
                letter-spacing: 1px;
                transition: background 0.2s, box-shadow 0.2s;
            }
            .auth-button:hover {
                background: linear-gradient(90deg, #1e40af 0%, #2563eb 100%);
                box-shadow: 0 8px 25px rgba(30, 60, 114, 0.18);
            }
            .auth-error {
                color: #e74c3c;
                font-size: 1em;
                margin-top: 8px;
                min-height: 24px;
            }
            .auth-links {
                margin-top: 10px;
                font-size: 1em;
            }
            .auth-links a {
                color: #2563eb;
                text-decoration: none;
                font-weight: 600;
                margin-left: 4px;
            }
            .auth-links a:hover {
                text-decoration: underline;
            }
            .entry-container {
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                background: linear-gradient(135deg, #e3eafc 0%, #f5f7fa 100%);
                padding: 40px 0;
            }
            .entry-content {
                max-width: 900px;
                width: 100%;
                text-align: center;
                color: #222;
                background: #fff;
                border-radius: 32px;
                box-shadow: 0 8px 40px rgba(30, 60, 114, 0.10);
                padding: 60px 40px 50px 40px;
                margin: 0 20px;
            }
            .entry-title {
                font-size: 3.5em;
                margin-bottom: 20px;
                font-weight: 700;
                color: #2563eb;
                letter-spacing: 1px;
            }
            .entry-subtitle {
                font-size: 1.7em;
                margin-bottom: 40px;
                color: #444;
                font-weight: 400;
            }
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 32px;
                margin: 40px 0;
            }
            .feature-card {
                background: #f8fafc;
                padding: 32px 20px;
                border-radius: 18px;
                box-shadow: 0 2px 12px rgba(30, 60, 114, 0.07);
                transition: box-shadow 0.2s, transform 0.2s;
            }
            .feature-card:hover {
                transform: translateY(-2px) scale(1.03);
                box-shadow: 0 8px 32px rgba(30, 60, 114, 0.13);
            }
            .feature-icon {
                width: 64px;
                height: 64px;
                margin-bottom: 18px;
            }
            .feature-card h3 {
                margin: 0 0 12px 0;
                color: #2563eb;
                font-size: 1.3em;
                font-weight: 600;
            }
            .feature-card p {
                margin: 0;
                color: #444;
                font-size: 1.08em;
                line-height: 1.5;
            }
            .enter-button {
                background: linear-gradient(90deg, #2563eb 0%, #1e40af 100%);
                color: #fff;
                border: none;
                padding: 20px 60px;
                font-size: 1.4em;
                border-radius: 40px;
                cursor: pointer;
                transition: background 0.2s, box-shadow 0.2s;
                font-weight: 600;
                margin-top: 30px;
                box-shadow: 0 4px 15px rgba(30, 60, 114, 0.10);
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .enter-button:hover {
                background: linear-gradient(90deg, #1e40af 0%, #2563eb 100%);
                box-shadow: 0 8px 25px rgba(30, 60, 114, 0.13);
            }
            .dashboard-container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 30px;
                background: #fff;
                border-radius: 18px;
                box-shadow: 0 4px 24px rgba(30, 60, 114, 0.08);
            }
            .card, .stat-card, .prediction-card, .metric-card {
                background: #f8fafc;
                color: #222;
                border-radius: 16px;
                box-shadow: 0 2px 12px rgba(30, 60, 114, 0.07);
                margin-bottom: 30px;
                padding: 30px;
                transition: box-shadow 0.2s, transform 0.2s;
            }
            .card:hover, .stat-card:hover, .prediction-card:hover, .metric-card:hover {
                transform: translateY(-2px) scale(1.01);
                box-shadow: 0 8px 32px rgba(30, 60, 114, 0.13);
            }
            .stat-number {
                font-size: 2.2em;
                color: #2563eb;
                margin: 0;
                font-weight: bold;
                text-shadow: none;
            }
            .stat-label {
                color: #666;
                margin: 10px 0 0 0;
                font-size: 1.1em;
            }
            .prediction-item span:last-child {
                color: #2563eb;
                font-weight: bold;
            }
            .trend-icon {
                margin-right: 6px;
                font-size: 1.1em;
                vertical-align: middle;
            }
            .profile-icon {
                font-size: 2em;
                color: #2563eb;
                 margin-right: 10px;
            }
            .profile-dropdown {
                background: #f8fafc;
                color: #222;
                border-radius: 12px;
                box-shadow: 0 4px 24px rgba(30, 60, 114, 0.10);
            }
            .logout-button {
                background: linear-gradient(90deg, #2563eb 0%, #1e40af 100%);
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 20px;
                cursor: pointer;
                font-size: 1em;
                transition: background 0.2s, box-shadow 0.2s;
                width: 100%;
            }
            .logout-button:hover {
                background: linear-gradient(90deg, #1e40af 0%, #2563eb 100%);
                box-shadow: 0 2px 8px rgba(30, 60, 114, 0.13);
            }
            .dash-tabs {
                border-radius: 12px !important;
                overflow: hidden !important;
                background: #f1f5f9;
            }
            .dash-tab--selected {
                background: #2563eb !important;
                color: white !important;
            }
            .js-plotly-plot {
                border-radius: 14px !important;
                box-shadow: 0 2px 12px rgba(30, 60, 114, 0.07) !important;
                background: #f8fafc !important;
            }
            .calculator-header-toggle {
                color: #2563eb;
            }
            .calculator-header-toggle:hover {
                color: #1e40af;
            }
            .input-group label {
                color: #2563eb;
            }
             .calculator-button {
                background: linear-gradient(90deg, #2563eb 0%, #1e40af 100%);
                color: #fff;
                 border: none;
                border-radius: 8px;
                font-size: 1em;
                padding: 10px 20px;
                transition: background 0.2s, box-shadow 0.2s;
            }
             .calculator-button:hover {
                background: linear-gradient(90deg, #1e40af 0%, #2563eb 100%);
                color: #fff;
            }
            .trend-badge-up {
                background: #22c55e;
                color: #fff;
                border-radius: 12px;
                padding: 2px 10px;
                font-size: 0.9em;
                margin-left: 10px;
            }
            .trend-badge-down {
                background: #ef4444;
                color: #fff;
                border-radius: 12px;
                padding: 2px 10px;
                font-size: 0.9em;
                margin-left: 10px;
            }
            .trend-badge-neutral {
                background: #facc15;
                color: #fff;
                border-radius: 12px;
                padding: 2px 10px;
                font-size: 0.9em;
                margin-left: 10px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Add error handling for the app
@app.server.errorhandler(Exception)
def handle_error(e):
    print(f"Server error: {str(e)}")
    traceback.print_exc()
    return str(e), 500

if __name__ == '__main__':
    try:
        print("Starting server...")
        # Configure server settings for production
        port = int(os.environ.get('PORT', 8050))
        host = os.environ.get('HOST', '127.0.0.1')
        
        # Production settings
        if os.environ.get('ENVIRONMENT') == 'production':
            app.run_server(
                host=host,
                port=port,
                debug=False,
                use_reloader=False,
                threaded=True,
                dev_tools_ui=False,
                dev_tools_props_check=False
            )
        else:
            # Development settings
            app.run_server(
                host=host,
                port=port,
                debug=True,
                use_reloader=True,
                threaded=True,
                dev_tools_ui=True,
                dev_tools_props_check=True
            )
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()
