import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from dotenv import load_dotenv

# Import our analysis module
from app import run_analysis, quick_test

# Load environment variables
load_dotenv()
api_key = os.environ.get("API_KEY")

# Initialize the Dash app
app = dash.Dash(
    __name__,
    title="Stock Analysis Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("Stock Analysis with ML & Sentiment", className="header-title"),
        html.P("Analyze stock performance with machine learning and news sentiment")
    ], className="header"),
    
    # Input section
    html.Div([
        html.Div([
            html.Label("Stock Ticker"),
            dcc.Input(
                id="ticker-input",
                type="text",
                placeholder="Enter stock ticker (e.g., AAPL)",
                value=" ",
                className="control-input"
            ),
        ], className="control-element"),
        
        html.Div([
            html.Label("Initial Investment ($)"),
            dcc.Input(
                id="investment-input",
                type="number",
                min=1000,
                max=1000000,
                step=1000,
                value=10000,
                className="control-input"
            ),
        ], className="control-element"),
        
        html.Div([
            html.Label("ML Training Period (days)"),
            dcc.Input(
                id="ml-days-input",
                type="number",
                min=30,
                max=1000,
                step=30,
                value=" ",
                className="control-input"
            ),
        ], className="control-element"),
        
        html.Div([
            html.Label("Sentiment Analysis Period (days)"),
            dcc.Input(
                id="sentiment-days-input",
                type="number",
                min=7,
                max=100,
                step=7,
                value=30,
                className="control-input"
            ),
        ], className="control-element"),
        
        html.Button("Run Analysis", id="run-button", className="control-button"),
        
        html.Div(id="loading-div", children=[
            dcc.Loading(
                id="loading-spinner",
                type="circle",
                children=html.Div(id="loading-output")
            ),
        ]),
        
    ], className="control-panel"),
    
    # Tabs for different visualizations
    html.Div([
        dcc.Tabs(id='viz-tabs', value='candlestick', children=[
            dcc.Tab(label='Price Chart', value='candlestick', children=[
                dcc.RadioItems(
                    id='chart-type-radio',
                    options=[
                        {'label': 'Candlestick', 'value': 'candlestick'},
                        {'label': 'OHLC', 'value': 'ohlc'},
                        {'label': 'Line', 'value': 'line'}
                    ],
                    value='candlestick',
                    labelStyle={'margin-right': '15px', 'display': 'inline-block'},
                    className="radio-group"
                ),
                dcc.Graph(id='price-chart')
            ]),
            dcc.Tab(label='Signals & Predictions', value='signals', children=[
                dcc.Graph(id='signals-chart')
            ]),
            dcc.Tab(label='Portfolio Performance', value='portfolio', children=[
                html.Div([
                    html.Label("Select Trading Strategies to Display"),
                    dcc.Checklist(
                        id='strategies-checklist',
                        options=[
                            {'label': 'ML Model', 'value': 'ML'},
                            {'label': 'Sentiment', 'value': 'Sentiment'},
                            {'label': 'Combined', 'value': 'Combined'},
                            {'label': 'Sentiment-First', 'value': 'Sentiment_First'},
                            {'label': 'Strong Signals', 'value': 'Strong'},
                            {'label': 'Buy & Hold', 'value': 'Buy_Hold'}
                        ],
                        value=['ML', 'Sentiment', 'Combined', 'Buy_Hold'],
                        inline=True,
                        className="checklist-group"
                    ),
                ]),
                dcc.Graph(id='portfolio-chart')
            ]),
            dcc.Tab(label='Performance Metrics', value='metrics', children=[
                dcc.Graph(id='metrics-chart')
            ]),
            dcc.Tab(label='Feature Importance', value='features', children=[
                dcc.Graph(id='features-chart')
            ])
        ]),
    ], className="visualization-panel"),
    
    # Hidden divs to store data
    html.Div(id='signals-data-store', style={'display': 'none'}),
    html.Div(id='portfolio-data-store', style={'display': 'none'}),
    html.Div(id='performance-data-store', style={'display': 'none'}),
    html.Div(id='stock-data-store', style={'display': 'none'}),
    
    # Footer
    html.Div([
        html.P("Stock Analysis Dashboard | Data from Yahoo Finance & NewsAPI")
    ], className="footer")
    
], className="app-container")

# Callback for running the analysis
@app.callback(
    [Output('loading-output', 'children'),
     Output('signals-data-store', 'children'),
     Output('portfolio-data-store', 'children'),
     Output('performance-data-store', 'children'),
     Output('stock-data-store', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('investment-input', 'value'),
     State('ml-days-input', 'value'),
     State('sentiment-days-input', 'value')]
)
def run_stock_analysis(n_clicks, ticker, initial_investment, ml_days, sentiment_days):
    if n_clicks is None:
        # Load existing results if available
        try:
            ticker = ticker.upper().strip()
            results_dir = Path(f"./results/{ticker}")
            
            if results_dir.exists():
                signals_file = results_dir / f'{ticker}_signals.csv'
                portfolio_file = results_dir / f'{ticker}_portfolio_values.csv'
                performance_file = results_dir / f'{ticker}_performance.json'
                stock_file = results_dir / f'{ticker}_stock_data.csv'
                
                if all(f.exists() for f in [signals_file, portfolio_file, performance_file, stock_file]):
                    signals = pd.read_csv(signals_file).to_json(date_format='iso', orient='split')
                    portfolio = pd.read_csv(portfolio_file).to_json(date_format='iso', orient='split')
                    
                    with open(performance_file, 'r') as f:
                        performance = json.load(f)
                    
                    stock_data = pd.read_csv(stock_file).to_json(date_format='iso', orient='split')
                    
                    return [
                        f"Loaded existing analysis for {ticker}",
                        signals,
                        portfolio,
                        json.dumps(performance),
                        stock_data
                    ]
        except Exception as e:
            print(f"Error loading existing data: {str(e)}")
            pass
        
        # If no existing data or error loading, return empty
        return [
            "Enter a ticker and click 'Run Analysis'",
            None, None, None, None
        ]
    
    try:
        # Check if ticker is valid
        ticker = ticker.upper().strip()
        if not ticker or len(ticker) > 6:
            return [
                "Please enter a valid ticker symbol",
                None, None, None, None
            ]
        
        # Run the analysis
        results = run_analysis(
            ticker=ticker,
            api_key=api_key,
            initial_investment=float(initial_investment),
            ml_days=int(ml_days),
            sentiment_days=int(sentiment_days),
            save_data=True
        )
        
        # Convert to JSON for storage
        signals = results['signals'].to_json(date_format='iso', orient='split')
        portfolio = results['portfolio_values'].to_json(date_format='iso', orient='split')
        performance = results['performance']
        
        # Convert performance to JSON-serializable format
        json_performance = {}
        for k, v in performance.items():
            if isinstance(v, dict):
                json_performance[k] = {str(k2): float(v2) if isinstance(v2, (int, float, np.number)) else v2 
                                      for k2, v2 in v.items()}
            else:
                json_performance[k] = float(v) if isinstance(v, (int, float, np.number)) else v
        
        stock_data = results['stock_data'].to_json(date_format='iso', orient='split')
        
        return [
            f"Analysis complete for {ticker}",
            signals,
            portfolio, 
            json.dumps(json_performance),
            stock_data
        ]
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [
            f"Error: {str(e)}",
            None, None, None, None
        ]

# Callback for updating price chart
@app.callback(
    Output('price-chart', 'figure'),
    [Input('chart-type-radio', 'value'),
     Input('signals-data-store', 'children'),
     Input('stock-data-store', 'children')]
)
def update_price_chart(chart_type, signals_json, stock_json):
    if not signals_json or not stock_json:
        # Return empty figure if no data
        return go.Figure().update_layout(
            title="No data available. Run analysis first.",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white"
        )
    
    # Load stock data
    signals_df = pd.read_json(signals_json, orient='split')
    stock_df = pd.read_json(stock_json, orient='split')
    
    # Ensure Date is datetime
    signals_df['Date'] = pd.to_datetime(signals_df['Date'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
    # Filter stock data to match signals period
    min_date = signals_df['Date'].min()
    stock_df = stock_df[stock_df['Date'] >= min_date]
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Price", "Volume")
    )
    
    # Add price chart based on selection
    if chart_type == 'candlestick':
        fig.add_trace(
            go.Candlestick(
                x=stock_df['Date'],
                open=stock_df['Open'],
                high=stock_df['High'],
                low=stock_df['Low'],
                close=stock_df['Close'],
                name="OHLC"
            ),
            row=1, col=1
        )
    elif chart_type == 'ohlc':
        fig.add_trace(
            go.Ohlc(
                x=stock_df['Date'],
                open=stock_df['Open'],
                high=stock_df['High'],
                low=stock_df['Low'],
                close=stock_df['Close'],
                name="OHLC"
            ),
            row=1, col=1
        )
    else:  # line chart
        fig.add_trace(
            go.Scatter(
                x=stock_df['Date'],
                y=stock_df['Close'],
                mode='lines',
                name="Close Price",
                line=dict(width=2, color='blue')
            ),
            row=1, col=1
        )
    
    # Add volume chart
    fig.add_trace(
        go.Bar(
            x=stock_df['Date'],
            y=stock_df['Volume'],
            name="Volume",
            marker_color='rgba(0, 0, 255, 0.3)'
        ),
        row=2, col=1
    )
    
    # Add signals
    # Buy signals (ML)
    buy_signals = signals_df[signals_df['ML_Prediction'] == 1]
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals['Date'],
                y=buy_signals['Actual_Close'],
                mode='markers',
                marker=dict(
                    size=10,
                    symbol='triangle-up',
                    color='green',
                    opacity=0.7
                ),
                name="ML Buy Signal"
            ),
            row=1, col=1
        )
    
    # Sell signals (ML)
    sell_signals = signals_df[signals_df['ML_Prediction'] == 0]
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals['Date'],
                y=sell_signals['Actual_Close'],
                mode='markers',
                marker=dict(
                    size=10,
                    symbol='triangle-down',
                    color='red',
                    opacity=0.7
                ),
                name="ML Sell Signal"
            ),
            row=1, col=1
        )
    
    # Strong buy signals
    strong_buy = signals_df[signals_df['Strong_Signal'] == 1]
    if not strong_buy.empty:
        fig.add_trace(
            go.Scatter(
                x=strong_buy['Date'],
                y=strong_buy['Actual_Close'],
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='star',
                    color='gold',
                    line=dict(color='green', width=2)
                ),
                name="Strong Buy Signal"
            ),
            row=1, col=1
        )
    
    # Strong sell signals
    strong_sell = signals_df[signals_df['Strong_Signal'] == 0]
    if not strong_sell.empty:
        fig.add_trace(
            go.Scatter(
                x=strong_sell['Date'],
                y=strong_sell['Actual_Close'],
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='star',
                    color='orange',
                    line=dict(color='red', width=2)
                ),
                name="Strong Sell Signal"
            ),
            row=1, col=1
        )
    
    # Update layout
    ticker = signals_df.iloc[0]['Date'].strftime('%Y-%m-%d') if not signals_df.empty else ""
    
    fig.update_layout(
        title=f"Price Chart with Trading Signals",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=700,
    )
    
    fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
            # Hide weekends
            dict(bounds=["sat", "mon"])
        ]
    )
    
    fig.update_yaxes(
        title_text="Price ($)",
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Volume",
        row=2, col=1
    )
    
    return fig

# Callback for updating signals chart
@app.callback(
    Output('signals-chart', 'figure'),
    [Input('signals-data-store', 'children')]
)
def update_signals_chart(signals_json):
    if not signals_json:
        # Return empty figure if no data
        return go.Figure().update_layout(
            title="No data available. Run analysis first.",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white"
        )
    
    # Load signals data
    signals_df = pd.read_json(signals_json, orient='split')
    
    # Ensure Date is datetime
    signals_df['Date'] = pd.to_datetime(signals_df['Date'])
    
    # Create figure with price and ML confidence
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4],
        subplot_titles=("Price with Predictions", "Signal Confidence")
    )
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=signals_df['Date'],
            y=signals_df['Actual_Close'],
            mode='lines',
            name="Close Price",
            line=dict(width=2, color='blue')
        ),
        row=1, col=1
    )
    
    # Add ML confidence
    fig.add_trace(
        go.Scatter(
            x=signals_df['Date'],
            y=signals_df['ML_Confidence'],
            mode='lines+markers',
            name="ML Confidence",
            line=dict(width=2, color='purple')
        ),
        row=2, col=1
    )
    
    # Add sentiment score
    fig.add_trace(
        go.Scatter(
            x=signals_df['Date'],
            y=signals_df['sentiment_score'],
            mode='lines+markers',
            name="Sentiment Score",
            line=dict(width=2, color='green')
        ),
        row=2, col=1
    )
    
    # Add 0.5 threshold line for ML confidence
    fig.add_shape(
        type="line",
        x0=signals_df['Date'].min(),
        x1=signals_df['Date'].max(),
        y0=0.5,
        y1=0.5,
        line=dict(
            color="rgba(128, 0, 128, 0.5)",
            width=2,
            dash="dash",
        ),
        row=2, col=1
    )
    
    # Add 0 threshold line for sentiment
    fig.add_shape(
        type="line",
        x0=signals_df['Date'].min(),
        x1=signals_df['Date'].max(),
        y0=0,
        y1=0,
        line=dict(
            color="rgba(0, 128, 0, 0.5)",
            width=2,
            dash="dash",
        ),
        row=2, col=1
    )
    
    # Add colored backgrounds for predictions
    for i, row in signals_df.iterrows():
        # Skip last day (no next day return)
        if i == len(signals_df) - 1:
            continue
        
        # Determine if prediction was correct
        ml_correct = row['ML_Prediction'] == row['Actual_Target']
        sentiment_correct = row['Sentiment_Signal'] == row['Actual_Target']
        
        # Add ML prediction marker
        color = "green" if ml_correct else "red"
        marker_symbol = "triangle-up" if row['ML_Prediction'] == 1 else "triangle-down"
        
        fig.add_trace(
            go.Scatter(
                x=[row['Date']],
                y=[row['Actual_Close']],
                mode='markers',
                marker=dict(
                    size=12,
                    symbol=marker_symbol,
                    color=color,
                ),
                name=f"ML {'Buy' if row['ML_Prediction'] == 1 else 'Sell'} ({row['ML_Confidence']:.2f})",
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Add annotations for some headlines
    for i, row in signals_df.iterrows():
        if i % 4 == 0 and row['headline'] is not None:  # Add every 4th headline to avoid clutter
            headline = row['headline']
            if isinstance(headline, str) and len(headline) > 50:
                headline = headline[:47] + "..."
                
            fig.add_annotation(
                x=row['Date'],
                y=row['Actual_Close'],
                text=headline,
                showarrow=True,
                arrowhead=4,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#636363",
                ax=0,
                ay=-40,
                bgcolor="white",
                opacity=0.8
            )
    
    # Update layout
    ticker = signals_df.iloc[0]['Date'].strftime('%Y-%m-%d') if not signals_df.empty else ""
    
    fig.update_layout(
        title=f"Prediction Signals and Confidence",
        xaxis_title="Date",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=700
    )
    
    fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
            # Hide weekends
            dict(bounds=["sat", "mon"])
        ]
    )
    
    fig.update_yaxes(
        title_text="Price ($)",
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Signal Strength",
        row=2, col=1
    )
    
    return fig

# Callback for updating portfolio chart
@app.callback(
    Output('portfolio-chart', 'figure'),
    [Input('portfolio-data-store', 'children'),
     Input('strategies-checklist', 'value')]
)
def update_portfolio_chart(portfolio_json, strategies):
    if not portfolio_json or not strategies:
        # Return empty figure if no data
        return go.Figure().update_layout(
            title="No data available. Run analysis first.",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white"
        )
    
    # Load portfolio data
    portfolio_df = pd.read_json(portfolio_json, orient='split')
    
    # Ensure Date is datetime
    portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
    
    # Create figure
    fig = go.Figure()
    
    # Color mapping for strategies
    colors = {
        'ML': 'blue',
        'Sentiment': 'green',
        'Combined': 'purple',
        'Sentiment_First': 'orange',
        'Strong': 'red',
        'Buy_Hold': 'black'
    }
    
    # Add lines for selected strategies
    for strategy in strategies:
        col_name = f"{strategy}_Portfolio"
        if col_name in portfolio_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df['Date'],
                    y=portfolio_df[col_name],
                    mode='lines',
                    name=strategy.replace('_', ' '),
                    line=dict(
                        width=2,
                        color=colors.get(strategy, 'gray')
                    )
                )
            )
    
    # Update layout
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )
    
    fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
            # Hide weekends
            dict(bounds=["sat", "mon"])
        ]
    )
    
    return fig

# Callback for updating metrics chart
@app.callback(
    Output('metrics-chart', 'figure'),
    [Input('performance-data-store', 'children')]
)
def update_metrics_chart(performance_json):
    if not performance_json:
        # Return empty figure if no data
        return go.Figure().update_layout(
            title="No data available. Run analysis first.",
            template="plotly_white"
        )
    
    # Load performance data
    performance = json.loads(performance_json)
    
    # Create subplots for different metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Accuracy by Strategy", 
            "Average Return by Strategy",
            "Win Rate by Strategy",
            "Trade Count by Strategy"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Strategies to include
    strategies = ['ML', 'Sentiment', 'Combined', 'Sentiment First', 'Strong']
    
    # 1. Accuracy plot
    accuracies = [
        performance['ML Accuracy'], 
        performance['Sentiment Accuracy'],
        performance.get('Combined Accuracy', 0),
        performance.get('Sentiment First Accuracy', 0),
        performance.get('Strong Signal Accuracy', 0)
    ]
    
    # Replace NaN or None with 0
    accuracies = [0 if not isinstance(acc, (int, float)) or pd.isna(acc) else acc for acc in accuracies]
    
    fig.add_trace(
        go.Bar(
            x=strategies, 
            y=accuracies,
            marker_color=['blue', 'green', 'purple', 'orange', 'red'],
            text=[f"{acc:.1%}" for acc in accuracies],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # 2. Average Return plot
    returns = [
        performance['ML Average Return'],
        performance['Sentiment Average Return'],
        performance.get('Combined Average Return', 0),
        performance.get('Sentiment First Average Return', 0),
        performance.get('Strong Signal Average Return', 0)
    ]
    
    # Replace NaN or None with 0
    returns = [0 if not isinstance(ret, (int, float)) or pd.isna(ret) else ret for ret in returns]
    
    # Convert to percentage
    returns = [ret * 100 for ret in returns]
    
    fig.add_trace(
        go.Bar(
            x=strategies, 
            y=returns,
            marker_color=['blue', 'green', 'purple', 'orange', 'red'],
            text=[f"{ret:.2f}%" for ret in returns],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # 3. Win Rate plot
    win_rates = [
        performance['ML Win Rate'] * 100,
        performance['Sentiment Win Rate'] * 100,
        performance.get('Combined Win Rate', 0) * 100,
        performance.get('Sentiment First Win Rate', 0) * 100,
        performance.get('Strong Signal Win Rate', 0) * 100
    ]
    
    # Replace NaN or None with 0
    win_rates = [0 if not isinstance(rate, (int, float)) or pd.isna(rate) else rate for rate in win_rates]
    
    fig.add_trace(
        go.Bar(
            x=strategies, 
            y=win_rates,
            marker_color=['blue', 'green', 'purple', 'orange', 'red'],
            text=[f"{rate:.1f}%" for rate in win_rates],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # 4. Trade Count plot
    trade_counts = [
        performance['Trade Count']['ML'],
        performance['Trade Count']['Sentiment'],
        performance['Trade Count']['Combined'],
        performance['Trade Count']['Sentiment First'],
        performance['Trade Count']['Strong']
    ]
    
    # Replace NaN or None with 0
    trade_counts = [0 if not isinstance(count, (int, float)) or pd.isna(count) else count for count in trade_counts]
    
    fig.add_trace(
        go.Bar(
            x=strategies, 
            y=trade_counts,
            marker_color=['blue', 'green', 'purple', 'orange', 'red'],
            text=[f"{int(count)}" for count in trade_counts],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    # Add 50% line to accuracy and win rate charts
    for row, col in [(1, 1), (2, 1)]:
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=4.5,
            y0=50,
            y1=50,
            line=dict(
                color="rgba(0, 0, 0, 0.5)",
                width=2,
                dash="dash",
            ),
            row=row, col=col
        )
    
    # Add 0% line to return chart
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=4.5,
        y0=0,
        y1=0,
        line=dict(
            color="rgba(0, 0, 0, 0.5)",
            width=2,
            dash="dash",
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="Performance Metrics by Strategy",
        template="plotly_white",
        showlegend=False,
        height=700
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Accuracy (%)", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Average Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Win Rate (%)", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Number of Trades", row=2, col=2)
    
    # Update x-axis titles
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Strategy", row=i, col=j)
    
    return fig

# Callback for updating features chart
@app.callback(
    Output('features-chart', 'figure'),
    [Input('stock-data-store', 'children')]
)
def update_features_chart(stock_json):
    if not stock_json:
        # Return empty figure if no data
        return go.Figure().update_layout(
            title="No data available. Run analysis first.",
            template="plotly_white"
        )
    
    # We need to re-analyze feature importance here
    try:
        stock_df = pd.read_json(stock_json, orient='split')
        
        # Prepare features again
        stock_df, predictors = prepare_stock_features(stock_df)
        
        # Prepare data for feature importance
        X = stock_df[predictors]
        y = stock_df['Target']
        
        # Train model to get feature importance
        from sklearn.ensemble import RandomForestClassifier
        
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
        }).sort_values('Importance', ascending=False).head(15)
        
        # Create horizontal bar chart
        fig = go.Figure(
            go.Bar(
                x=feature_importance['Importance'],
                y=feature_importance['Feature'],
                orientation='h',
                marker_color='blue'
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Top 15 Most Important Features",
            xaxis_title="Importance",
            yaxis_title="Feature",
            template="plotly_white",
            height=600
        )
        
        # Reverse y-axis to show most important at the top
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        
        return fig
    
    except Exception as e:
        # If error, return empty figure with error message
        return go.Figure().update_layout(
            title=f"Error analyzing feature importance: {str(e)}",
            template="plotly_white"
        )

# CSS for styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                background-color: #f5f5f5;
            }
            
            .app-container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 15px;
            }
            
            .header {
                background-color: #2c3e50;
                color: white;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .header-title {
                margin: 0;
                font-size: 24px;
            }
            
            .control-panel {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                align-items: flex-end;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .control-element {
                flex: 1;
                min-width: 200px;
            }
            
            .control-input {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 5px;
            }
            
            .control-button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            
            .control-button:hover {
                background-color: #2980b9;
            }
            
            .visualization-panel {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .radio-group {
                margin: 10px 0;
            }
            
            .checklist-group {
                margin: 10px 0;
            }
            
            .footer {
                text-align: center;
                margin-top: 20px;
                color: #7f8c8d;
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

# Function to prepare stock features for feature importance analysis
def prepare_stock_features(stock_data):
    """Simplified version for visualization only"""
    # Create target variable - whether tomorrow's price will be higher than today's
    stock_data["Tomorrow"] = stock_data["Close"].shift(-1)
    stock_data["Target"] = (stock_data["Tomorrow"] > stock_data["Close"]).astype(int)
    
    # Create technical features
    horizons = [2, 5, 10, 20]
    new_predictors = []
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
    
    # Drop any rows with NaN values
    stock_data = stock_data.dropna()
    
    return stock_data, new_predictors

# Run the server if executed directly
if __name__ == '__main__':
    print("Starting Stock Analysis Dashboard...")
    print("API key loaded:", "Yes" if api_key else "No")
    
    # Run a quick test to verify API
    if quick_test(api_key=api_key):
        print("API test successful. Dashboard is ready.")
    else:
        print("WARNING: API test failed. Some features may not work correctly.")
    
    #Run the dashboard
    app.run(debug=True)
