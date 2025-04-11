import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings
import json
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

# ========== DATA COLLECTION ==========

def get_stock_data(ticker, start_date=None, end_date=None):
    """Get stock data using yfinance"""
    stock = yf.Ticker(ticker)
    
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = stock.history(start=start_date, end=end_date)
    data = data.reset_index()
    return data

def get_news_data(query, days=30, api_key=None):
    """Get news data from NewsAPI"""
    if not api_key:
        # Try to load from .env file
        load_dotenv()
        api_key = os.environ.get("API_KEY")
        
    if not api_key:
        raise ValueError("API key not provided. Set API_KEY in .env file or pass as parameter.")
    
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
        'sortBy': 'relevancy',
        'apiKey': api_key,
        'pageSize': 100,
        'language': 'en'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if data['status'] != 'ok':
        raise Exception(f"NewsAPI error: {data['message']}")
    
    articles = data['articles']
    news_data = pd.DataFrame(articles)
    
    # Check if the required columns exist
    if 'publishedAt' in news_data.columns and 'title' in news_data.columns:
        news_data = news_data[['publishedAt', 'title']]
        news_data.columns = ['date', 'headline']
        return news_data
    else:
        raise Exception("Required columns not found in NewsAPI response")

# ========== FEATURE ENGINEERING ==========

def preprocess_headlines(news_data):
    """Clean and preprocess news headlines"""
    try:
        stop_words = set(stopwords.words('english'))
    except:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        words = word_tokenize(text)
        words = [word for word in words if word.isalpha()]
        words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(words)
    
    news_data['clean_headlines'] = news_data['headline'].apply(preprocess_text)
    return news_data

def calculate_sentiment(news_data):
    """Calculate sentiment scores for headlines"""
    analyzer = SentimentIntensityAnalyzer()
    
    def get_sentiment_score(text):
        score = analyzer.polarity_scores(text)
        return score['compound']
    
    news_data['sentiment_score'] = news_data['clean_headlines'].apply(get_sentiment_score)
    return news_data

def prepare_stock_features(stock_data):
    """Prepare stock features for ML model"""
    # Create target variable - whether tomorrow's price will be higher than today's
    stock_data["Tomorrow"] = stock_data["Close"].shift(-1)
    stock_data["Target"] = (stock_data["Tomorrow"] > stock_data["Close"]).astype(int)
    
    # Create technical features - simplified for better performance on small datasets
    horizons = [2, 5, 10, 20]  # Multiple horizon windows
    
    # Initialize new predictors list
    new_predictors = []
    
    # Numeric columns for rolling calculations
    numeric_columns = ["Close", "Volume", "Open", "High", "Low"]
    
    # Rolling average calculations
    for horizon in horizons:
        for col in numeric_columns:
            # Rolling average
            rolling_avg_column = f"{col}_RollingAvg_{horizon}"
            stock_data[rolling_avg_column] = stock_data[col].rolling(window=horizon, min_periods=1).mean()
            new_predictors.append(rolling_avg_column)
            
            # Ratio (current value / rolling average)
            ratio_column = f"{col}_Ratio_{horizon}"
            stock_data[ratio_column] = stock_data[col] / stock_data[rolling_avg_column]
            new_predictors.append(ratio_column)
    
    # Add percentage change features
    for col in numeric_columns:
        pct_change_column = f"{col}_PctChange"
        stock_data[pct_change_column] = stock_data[col].pct_change()
        new_predictors.append(pct_change_column)
    
    # Add volatility features
    for horizon in [5, 10, 20]:
        stock_data[f'Volatility_{horizon}day'] = stock_data['Close'].pct_change().rolling(horizon).std()
        new_predictors.append(f'Volatility_{horizon}day')
    
    # Add MACD
    stock_data['MACD'] = stock_data['Close'].ewm(span=12).mean() - stock_data['Close'].ewm(span=26).mean()
    new_predictors.append('MACD')
    
    # Add RSI (simplified)
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.finfo(float).eps)
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    new_predictors.append('RSI')
    
    # Add Bollinger Bands
    stock_data['Bollinger_Middle'] = stock_data['Close'].rolling(20).mean()
    stock_data['Bollinger_Std'] = stock_data['Close'].rolling(20).std()
    stock_data['Bollinger_Upper'] = stock_data['Bollinger_Middle'] + 2 * stock_data['Bollinger_Std']
    stock_data['Bollinger_Lower'] = stock_data['Bollinger_Middle'] - 2 * stock_data['Bollinger_Std']
    
    stock_data['Bollinger_Position'] = (stock_data['Close'] - stock_data['Bollinger_Lower']) / (
            stock_data['Bollinger_Upper'] - stock_data['Bollinger_Lower'])
    new_predictors.append('Bollinger_Position')
    
    # Drop any rows with NaN values
    stock_data = stock_data.dropna()
    
    return stock_data, new_predictors

def merge_stock_and_sentiment(stock_data, sentiment_data):
    """Merge stock and sentiment data on date"""
    # Convert dates to datetime format
    sentiment_data['date'] = pd.to_datetime(sentiment_data['date']).dt.date
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date
    
    # Find the most influential news item for each day (based on absolute sentiment score)
    def find_most_influential_news(group):
        if len(group) == 0:
            return pd.Series({'date': None, 'headline': None, 'clean_headlines': None, 'sentiment_score': 0})
        return group.loc[abs(group['sentiment_score']).idxmax()]
        
    most_influential_news = sentiment_data.groupby('date').apply(find_most_influential_news).reset_index(drop=True)
    
    # Merge stock data with the most influential news
    combined_data = pd.merge(stock_data, most_influential_news, left_on='Date', right_on='date', how='left')
    
    # Fill NaN values for days without news
    combined_data['sentiment_score'] = combined_data['sentiment_score'].fillna(0)
    
    return combined_data

# ========== MODEL EVALUATION & IMPROVEMENT ==========

def analyze_feature_importance(stock_data, predictors, n_top=15):
    """Analyze feature importance in the ML model"""
    print("\nAnalyzing feature importance...")
    
    # Prepare data
    X = stock_data[predictors]
    y = stock_data['Target']
    
    # Train model with more moderate balancing to prevent single-class predictions
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=5,
        max_depth=10,
        class_weight={0: 1.0, 1: 1.2},
        random_state=42
    )
    model.fit(X, y)
    
    # Get feature importances
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': predictors,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop Most Important Features:")
    print(feature_importance.head(n_top))
    
    return feature_importance.head(n_top)['Feature'].tolist()

# ========== ML MODEL ==========

def create_balanced_ml_model(stock_data, sentiment_data, predictors, test_size=30, use_selected_features=True):
    """Create a balanced ML model that addresses the UP prediction bias"""
    # Use top features if indicated
    if use_selected_features and len(predictors) > 15:
        try:
            selected_predictors = analyze_feature_importance(stock_data, predictors)
            print(f"Using selected top features: {selected_predictors}")
        except Exception as e:
            print(f"Feature selection failed: {str(e)}, using all features")
            selected_predictors = predictors
    else:
        selected_predictors = predictors
    
    # Prepare training and testing data
    train = stock_data.iloc[:-test_size].copy()
    test = stock_data.iloc[-test_size:].copy()
    
    # Make sure 'Date' is datetime format for both dataframes
    if 'Date' not in test.columns:
        test = test.reset_index()
        train = train.reset_index()
    
    # Ensure Date column is datetime
    test['Date'] = pd.to_datetime(test['Date']).dt.date
    
    # Print the class distribution to debug
    print(f"\nTarget distribution in training data: {train['Target'].value_counts().to_dict()}")
    
    # Initialize and train the model with moderate class weights
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_split=5,
        max_depth=10,
        class_weight={0: 1.0, 1: 1.2},
        random_state=42
    )
    
    model.fit(train[selected_predictors], train["Target"])
    
    # Get predictions
    test_predictions = model.predict(test[selected_predictors])
    
    # Get probabilities safely with error handling
    try:
        # Try to get probabilities for both classes
        test_probabilities = model.predict_proba(test[selected_predictors])
        
        # Check if we have probabilities for both classes
        if test_probabilities.shape[1] >= 2:
            test_confidence = test_probabilities[:, 1]  # Probability of upward movement
        else:
            # If only one class, use fixed confidence based on prediction
            print("Warning: Only one class in probabilities. Using fixed confidence values.")
            test_confidence = np.where(test_predictions == 1, 0.75, 0.25)
    except Exception as e:
        print(f"Error getting prediction probabilities: {str(e)}")
        print("Using fixed confidence values.")
        test_confidence = np.where(test_predictions == 1, 0.75, 0.25)
    
    # Create a DataFrame with predictions
    prediction_df = pd.DataFrame({
        'Date': test['Date'],
        'Actual_Close': test['Close'],
        'Open': test['Open'],
        'High': test['High'],
        'Low': test['Low'],
        'Volume': test['Volume'],
        'ML_Prediction': test_predictions,
        'ML_Confidence': test_confidence,
        'Actual_Target': test['Target']
    })
    
    # Merge with sentiment data
    sentiment_data['Date'] = pd.to_datetime(sentiment_data['date']).dt.date
    combined_signals = pd.merge(
        prediction_df, 
        sentiment_data[['Date', 'sentiment_score', 'headline']], 
        on='Date', 
        how='left'
    )
    
    # Fill missing sentiment scores with 0
    combined_signals['sentiment_score'] = combined_signals['sentiment_score'].fillna(0)
    
    # Create sentiment signal (1 for positive, 0 for negative)
    combined_signals['Sentiment_Signal'] = (combined_signals['sentiment_score'] > 0).astype(int)
    
    # Create classic combined signals (ML and sentiment agree)
    combined_signals['Combined_Signal'] = np.where(
        combined_signals['ML_Prediction'] == combined_signals['Sentiment_Signal'],
        combined_signals['ML_Prediction'],
        -1  # No trade when signals disagree
    )
    
    # Create sentiment-first signals (primary signal is sentiment, ML confirms)
    combined_signals['Sentiment_First_Signal'] = np.where(
        (combined_signals['Sentiment_Signal'] == 1) & (combined_signals['ML_Confidence'] > 0.45),
        1,  # Buy when sentiment is positive and ML confidence is moderate+
        np.where(
            (combined_signals['Sentiment_Signal'] == 0) & (combined_signals['ML_Confidence'] < 0.55),
            0,  # Sell when sentiment is negative and ML confidence is moderate-
            -1  # No trade otherwise
        )
    )
    
    # Lower thresholds for Strong Signal to get more signals
    combined_signals['Strong_Signal'] = np.where(
        (combined_signals['ML_Prediction'] == 1) & 
        (combined_signals['Sentiment_Signal'] == 1) &
        (combined_signals['ML_Confidence'] > 0.52) &
        (combined_signals['sentiment_score'] > 0.15),
        1,  # Strong buy
        np.where(
            (combined_signals['ML_Prediction'] == 0) & 
            (combined_signals['Sentiment_Signal'] == 0) &
            (combined_signals['ML_Confidence'] > 0.52) &
            (combined_signals['sentiment_score'] < -0.15),
            0,  # Strong sell
            -1  # No strong signal
        )
    )
    
    # Calculate next day's return
    combined_signals['Next_Day_Return'] = np.append(
        combined_signals['Actual_Close'].pct_change(1).values[1:], [np.nan]
    )
    
    # Calculate potential returns based on signals
    # For ML signal
    combined_signals['ML_Return'] = np.where(
        combined_signals['ML_Prediction'] == 1,
        combined_signals['Next_Day_Return'],
        -combined_signals['Next_Day_Return']  # For sell signals, we benefit from price drops
    )
    
    # For sentiment signal
    combined_signals['Sentiment_Return'] = np.where(
        combined_signals['Sentiment_Signal'] == 1,
        combined_signals['Next_Day_Return'],
        -combined_signals['Next_Day_Return']
    )
    
    # For combined signal
    combined_signals['Combined_Return'] = np.where(
        combined_signals['Combined_Signal'] == 1,
        combined_signals['Next_Day_Return'],
        np.where(
            combined_signals['Combined_Signal'] == 0,
            -combined_signals['Next_Day_Return'],
            0  # No trade
        )
    )
    
    # For sentiment-first signal
    combined_signals['Sentiment_First_Return'] = np.where(
        combined_signals['Sentiment_First_Signal'] == 1,
        combined_signals['Next_Day_Return'],
        np.where(
            combined_signals['Sentiment_First_Signal'] == 0,
            -combined_signals['Next_Day_Return'],
            0  # No trade
        )
    )
    
    # For strong signal
    combined_signals['Strong_Return'] = np.where(
        combined_signals['Strong_Signal'] == 1,
        combined_signals['Next_Day_Return'],
        np.where(
            combined_signals['Strong_Signal'] == 0,
            -combined_signals['Next_Day_Return'],
            0  # No trade
        )
    )
    
    # Store the model for future predictions
    output_dir = Path(f"./models/{stock_data['Date'].iloc[-1].strftime('%Y%m%d')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"{stock_data.iloc[0]['Date'].strftime('%Y%m%d')}_{test.iloc[-1]['Date'].strftime('%Y%m%d')}_model.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "selected_features": selected_predictors,
            "training_end_date": train.iloc[-1]['Date']
        }, f)
    
    return combined_signals, selected_predictors, model

def analyze_performance(combined_signals):
    """Analyze the performance of different trading signals"""
    # Drop the last row which will have NaN for Next_Day_Return
    df = combined_signals.dropna(subset=['Next_Day_Return']).copy()
    
    # Initialize results
    results = {
        'ML Accuracy': 0,
        'Sentiment Accuracy': 0,
        'Combined Accuracy': np.nan,
        'Sentiment First Accuracy': np.nan,
        'Strong Signal Accuracy': np.nan,
        'ML Average Return': 0,
        'Sentiment Average Return': 0,
        'Combined Average Return': np.nan,
        'Sentiment First Average Return': np.nan,
        'Strong Signal Average Return': np.nan,
        'ML Win Rate': 0,
        'Sentiment Win Rate': 0,
        'Combined Win Rate': np.nan,
        'Sentiment First Win Rate': np.nan,
        'Strong Signal Win Rate': np.nan,
        'Trade Count': {
            'ML': 0,
            'Sentiment': 0,
            'Combined': 0,
            'Sentiment First': 0,
            'Strong': 0
        }
    }
    
    if len(df) == 0:
        return results
    
    # Calculate accuracy for different signals
    results['ML Accuracy'] = accuracy_score(
        df['Actual_Target'],
        df['ML_Prediction']
    )
    
    results['Sentiment Accuracy'] = accuracy_score(
        df['Actual_Target'],
        df['Sentiment_Signal']
    )
    
    # Calculate accuracy for combined signals only when they generate a trade
    combined_trades = df[df['Combined_Signal'] != -1]
    if len(combined_trades) > 0:
        results['Combined Accuracy'] = accuracy_score(
            combined_trades['Actual_Target'],
            combined_trades['Combined_Signal']
        )
    
    # Calculate accuracy for sentiment-first signals
    sentiment_first_trades = df[df['Sentiment_First_Signal'] != -1]
    if len(sentiment_first_trades) > 0:
        results['Sentiment First Accuracy'] = accuracy_score(
            sentiment_first_trades['Actual_Target'],
            sentiment_first_trades['Sentiment_First_Signal']
        )
    
    # Calculate strong signal accuracy if there are any
    strong_trades = df[df['Strong_Signal'] != -1]
    if len(strong_trades) > 0:
        results['Strong Signal Accuracy'] = accuracy_score(
            strong_trades['Actual_Target'],
            strong_trades['Strong_Signal']
        )
    
    # Calculate average returns
    results['ML Average Return'] = df['ML_Return'].mean()
    results['Sentiment Average Return'] = df['Sentiment_Return'].mean()
    
    # Handle potential empty DataFrames
    if len(df[df['Combined_Return'] != 0]) > 0:
        results['Combined Average Return'] = df['Combined_Return'].replace(0, np.nan).mean()
    
    if len(df[df['Sentiment_First_Return'] != 0]) > 0:
        results['Sentiment First Average Return'] = df['Sentiment_First_Return'].replace(0, np.nan).mean()
    
    if len(df[df['Strong_Return'] != 0]) > 0:
        results['Strong Signal Average Return'] = df['Strong_Return'].replace(0, np.nan).mean()
    
    # Calculate win rate
    if len(df) > 0:
        results['ML Win Rate'] = (df['ML_Return'] > 0).mean()
        results['Sentiment Win Rate'] = (df['Sentiment_Return'] > 0).mean()
    
    combined_returns = df['Combined_Return'].replace(0, np.nan)
    if not combined_returns.isna().all():
        results['Combined Win Rate'] = (combined_returns > 0).mean()
    
    sentiment_first_returns = df['Sentiment_First_Return'].replace(0, np.nan)
    if not sentiment_first_returns.isna().all():
        results['Sentiment First Win Rate'] = (sentiment_first_returns > 0).mean()
    
    strong_returns = df['Strong_Return'].replace(0, np.nan)
    if not strong_returns.isna().all():
        results['Strong Signal Win Rate'] = (strong_returns > 0).mean()
    
    # Count trades
    results['Trade Count'] = {
        'ML': len(df),
        'Sentiment': len(df),
        'Combined': (df['Combined_Signal'] != -1).sum(),
        'Sentiment First': (df['Sentiment_First_Signal'] != -1).sum(),
        'Strong': (df['Strong_Signal'] != -1).sum()
    }
    
    return results

def create_portfolio_simulation(signals, initial_investment=10000):
    """Simulate portfolio performance based on different trading strategies"""
    # Drop rows with NaN returns
    df = signals.dropna(subset=['Next_Day_Return']).copy()
    
    if len(df) == 0:
        return pd.DataFrame()
    
    # Convert date to datetime for proper ordering
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Initialize portfolios with the initial investment
    strategies = ['ML', 'Sentiment', 'Combined', 'Sentiment_First', 'Strong', 'Buy_Hold']
    
    portfolio_values = pd.DataFrame({
        'Date': df['Date'],
        'ML_Portfolio': initial_investment,
        'Sentiment_Portfolio': initial_investment,
        'Combined_Portfolio': initial_investment,
        'Sentiment_First_Portfolio': initial_investment,
        'Strong_Portfolio': initial_investment,
        'Buy_Hold_Portfolio': initial_investment
    })
    
    # Calculate daily returns and portfolio values
    for i in range(1, len(df)):
        prev_values = portfolio_values.iloc[i-1].copy()
        
        # ML strategy
        ml_return = df.iloc[i-1]['ML_Return'] if not pd.isna(df.iloc[i-1]['ML_Return']) else 0
        portfolio_values.loc[portfolio_values.index[i], 'ML_Portfolio'] = prev_values['ML_Portfolio'] * (1 + ml_return)
        
        # Sentiment strategy
        sentiment_return = df.iloc[i-1]['Sentiment_Return'] if not pd.isna(df.iloc[i-1]['Sentiment_Return']) else 0
        portfolio_values.loc[portfolio_values.index[i], 'Sentiment_Portfolio'] = prev_values['Sentiment_Portfolio'] * (1 + sentiment_return)
        
        # Combined strategy
        combined_return = df.iloc[i-1]['Combined_Return'] if not pd.isna(df.iloc[i-1]['Combined_Return']) else 0
        portfolio_values.loc[portfolio_values.index[i], 'Combined_Portfolio'] = prev_values['Combined_Portfolio'] * (1 + combined_return)
        
        # Sentiment-First strategy
        sf_return = df.iloc[i-1]['Sentiment_First_Return'] if not pd.isna(df.iloc[i-1]['Sentiment_First_Return']) else 0
        portfolio_values.loc[portfolio_values.index[i], 'Sentiment_First_Portfolio'] = prev_values['Sentiment_First_Portfolio'] * (1 + sf_return)
        
        # Strong strategy
        strong_return = df.iloc[i-1]['Strong_Return'] if not pd.isna(df.iloc[i-1]['Strong_Return']) else 0
        portfolio_values.loc[portfolio_values.index[i], 'Strong_Portfolio'] = prev_values['Strong_Portfolio'] * (1 + strong_return)
        
        # Buy & Hold strategy
        buy_hold_return = df.iloc[i-1]['Next_Day_Return'] if not pd.isna(df.iloc[i-1]['Next_Day_Return']) else 0
        portfolio_values.loc[portfolio_values.index[i], 'Buy_Hold_Portfolio'] = prev_values['Buy_Hold_Portfolio'] * (1 + buy_hold_return)
    
    # Calculate performance metrics
    performance = {
        'Final Value': {},
        'Total Return': {},
        'Daily Returns': {},
        'Sharpe Ratio': {},
        'Max Drawdown': {},
        'Win Rate': {}
    }
    
    for strategy in strategies:
        col_name = f"{strategy}_Portfolio" if strategy != 'Buy_Hold' else 'Buy_Hold_Portfolio'
        
        # Final portfolio value
        performance['Final Value'][strategy] = portfolio_values[col_name].iloc[-1]
        
        # Total return percentage
        performance['Total Return'][strategy] = (portfolio_values[col_name].iloc[-1] / initial_investment - 1) * 100
        
        # Calculate daily returns
        daily_returns = portfolio_values[col_name].pct_change().dropna()
        performance['Daily Returns'][strategy] = daily_returns.mean() * 100
        
        # Sharpe ratio (assumes risk-free rate of 0 for simplicity)
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # Annualized
            performance['Sharpe Ratio'][strategy] = sharpe
        else:
            performance['Sharpe Ratio'][strategy] = 0
        
        # Maximum drawdown
        cum_returns = (1 + daily_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns / running_max - 1)
        max_drawdown = drawdown.min() * 100  # Convert to percentage
        performance['Max Drawdown'][strategy] = max_drawdown
        
        # Win rate (percentage of positive daily returns)
        performance['Win Rate'][strategy] = (daily_returns > 0).mean() * 100
    
    return portfolio_values, performance

def run_analysis(ticker, api_key=None, initial_investment=10000, ml_days=365, sentiment_days=30, save_data=True):
    """Run the full analysis pipeline for a given ticker"""
    print(f"Starting analysis for {ticker}...")
    results_dir = Path(f"./results/{ticker}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Get stock data - Default to 1 year of ML training data
    print("Fetching stock data...")
    
    # Calculate start dates
    ml_start_date = (datetime.now() - timedelta(days=ml_days)).strftime('%Y-%m-%d')
    
    # Get extended history for ML training
    stock_data = get_stock_data(ticker=ticker, start_date=ml_start_date)
    
    # 2. Prepare stock features
    print("Preparing stock features...")
    stock_data, predictors = prepare_stock_features(stock_data)
    
    # 3. Get news data (last month by default)
    print("Fetching news data...")
    try:
        news_data = get_news_data(query=ticker, days=sentiment_days, api_key=api_key)
    except Exception as e:
        print(f"Error fetching news data: {str(e)}")
        print(f"Trying with company name instead of ticker...")
        # Attempt to get actual company name if available
        try:
            company_info = yf.Ticker(ticker).info
            company_name = company_info.get('shortName') or ticker
            print(f"Using company name: {company_name}")
            news_data = get_news_data(query=company_name, days=sentiment_days, api_key=api_key)
        except Exception as e2:
            print(f"Error fetching news with company name: {str(e2)}")
            print("Creating empty news dataframe...")
            # Create an empty dataframe with the right structure
            news_data = pd.DataFrame(columns=['date', 'headline'])
    
    # 4. Process news data
    print("Processing news headlines...")
    if len(news_data) > 0:
        news_data = preprocess_headlines(news_data)
        news_data = calculate_sentiment(news_data)
    else:
        print("No news data available.")
    
    # 5. Merge stock and sentiment data
    print("Merging stock and sentiment data...")
    combined_data = merge_stock_and_sentiment(stock_data, news_data)
    
    # Print some information about the data
    print(f"\nTotal stock data points: {len(stock_data)}")
    print(f"Stock data date range: {stock_data['Date'].min()} to {stock_data['Date'].max()}")
    print(f"Sentiment data points: {len(news_data)}")
    print(f"Data points with both stock and sentiment: {len(combined_data)}")
    
    # 6. Save intermediate data if requested
    if save_data:
        stock_data.to_csv(results_dir / f'{ticker}_stock_data.csv', index=False)
        news_data.to_csv(results_dir / f'{ticker}_news_data.csv', index=False)
        combined_data.to_csv(results_dir / f'{ticker}_combined_data.csv', index=False)
    
    # 7. Create balanced ML model with sentiment-first approach
    print("Creating balanced ML model with sentiment-first approach...")
    
    # Use the last month (points with sentiment data) as test data
    test_size = min(sentiment_days, len(combined_data))
    
    # Train on longer history, test on recent month with sentiment
    signals, selected_predictors, model = create_balanced_ml_model(
        stock_data, combined_data, predictors, test_size, use_selected_features=True
    )
    
    # 8. Analyze performance
    print("Analyzing performance...")
    performance = analyze_performance(signals)
    
    # 9. Create portfolio simulation
    print("Simulating portfolio performance...")
    portfolio_values, portfolio_performance = create_portfolio_simulation(signals, initial_investment)
    
    # 10. Save results
    if save_data:
        signals.to_csv(results_dir / f'{ticker}_signals.csv', index=False)
        
        # Save performance metrics as JSON
        with open(results_dir / f'{ticker}_performance.json', 'w') as f:
            json.dump({k: v if not isinstance(v, dict) else {str(k2): float(v2) for k2, v2 in v.items()} 
                      for k, v in performance.items() if v is not None}, f, indent=4)
        
        # Save portfolio performance
        with open(results_dir / f'{ticker}_portfolio_performance.json', 'w') as f:
            # Convert portfolio_performance to JSON serializable format
            json_data = {}
            for metric, values in portfolio_performance.items():
                json_data[metric] = {k: float(v) for k, v in values.items()}
            json.dump(json_data, f, indent=4)
        
        portfolio_values.to_csv(results_dir / f'{ticker}_portfolio_values.csv', index=False)
    
    print(f"\nAnalysis for {ticker} complete!")
    
def get_latest_signals(ticker):
    """Get the latest trading signals for a ticker"""
    results_dir = Path(f"./results/{ticker}")
    
    if not results_dir.exists():
        return None
    
    try:
        signals_file = results_dir / f'{ticker}_signals.csv'
        if signals_file.exists():
            signals = pd.read_csv(signals_file)
            
            # Convert Date to datetime
            signals['Date'] = pd.to_datetime(signals['Date'])
            
            # Get the latest signals
            latest_signals = signals.iloc[-1].to_dict()
            return latest_signals
        else:
            return None
    except Exception as e:
        print(f"Error loading signals for {ticker}: {str(e)}")
        return None

def quick_test(ticker="AAPL", api_key=None):
    """Run a quick test to verify API and data fetching works"""
    print("Running quick test...")
    
    if not api_key:
        # Try to load from .env file
        load_dotenv()
        api_key = os.environ.get("API_KEY")
    
    if not api_key:
        print("ERROR: No API key found. Please set API_KEY in .env file.")
        return False
    
    print(f"Testing stock data fetch for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d")
        if len(data) == 0:
            print(f"ERROR: No stock data found for {ticker}")
            return False
        print(f"Successfully fetched {len(data)} days of {ticker} data")
    except Exception as e:
        print(f"ERROR fetching stock data: {str(e)}")
        return False
    
    print("Testing NewsAPI...")
    try:
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q': ticker,
            'apiKey': api_key,
            'pageSize': 1
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['status'] != 'ok':
            print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return False
            
        print("NewsAPI test successful")
    except Exception as e:
        print(f"ERROR with NewsAPI: {str(e)}")
        return False
    
    print("All tests passed!")
    return True

# Main execution block if run as script
if __name__ == "__main__":
    # First run a quick test to verify everything works
    load_dotenv()
    api_key = os.environ.get("API_KEY")
    
    print("Stock Analysis Pipeline")
    print("=" * 50)
    
    if not quick_test(api_key=api_key):
        print("Quick test failed. Please fix the issues above before running the analysis.")
        exit(1)
    
    # Get user input for ticker
    ticker = input("Enter stock ticker (e.g., AAPL, MSFT, AMZN): ").strip().upper()
    
    # Get initial investment
    try:
        initial_investment = float(input("Enter initial investment amount (default: $10,000): ") or 10000)
    except ValueError:
        print("Invalid input. Using default: $10,000")
        initial_investment = 10000
    
    # Get ML training period
    try:
        ml_days = int(input("Enter ML training period in days (default: 365): ") or 365)
    except ValueError:
        print("Invalid input. Using default: 365 days")
        ml_days = 365
    
    # Get sentiment analysis period
    try:
        sentiment_days = int(input("Enter sentiment analysis period in days (default: 30): ") or 30)
    except ValueError:
        print("Invalid input. Using default: 30 days")
        sentiment_days = 30
    
    # Run the analysis
    try:
        results = run_analysis(
            ticker=ticker,
            api_key=api_key,
            initial_investment=initial_investment,
            ml_days=ml_days,
            sentiment_days=sentiment_days
        )
        
        print("\nAnalysis Results Summary:")
        print(f"Ticker: {ticker}")
        print(f"Period: {results['signals']['Date'].min()} to {results['signals']['Date'].max()}")
        
        print("\nAccuracy Metrics:")
        print(f"  ML Model Accuracy: {results['performance']['ML Accuracy']:.2%}")
        print(f"  Sentiment Analysis Accuracy: {results['performance']['Sentiment Accuracy']:.2%}")
        
        print("\nFinal Portfolio Values:")
        for strategy, value in results['portfolio_performance']['Final Value'].items():
            print(f"  {strategy}: ${value:.2f}")
        
        print("\nTotal Returns:")
        for strategy, ret in results['portfolio_performance']['Total Return'].items():
            print(f"  {strategy}: {ret:.2%}")
        
        print("\nResults saved to ./results/{ticker}/ directory")
        
    except Exception as e:
        print(f"ERROR running analysis: {str(e)}")
        import traceback
        traceback.print_exc()