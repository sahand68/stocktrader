import plotly.graph_objects as go
from datetime import timedelta

def create_candlestick_plot(df, ticker):
    """Create a candlestick plot with technical indicators."""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    
    # Add SMA
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['SMA_20'],
        name='SMA (20)',
        line=dict(color='orange', dash='dot')
    ))
    
    # Add EMA
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['EMA_20'],
        name='EMA (20)',
        line=dict(color='purple', dash='dot')
    ))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_upper'],
        name='Bollinger Upper',
        line=dict(color='gray', dash='dash'),
        opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['BB_lower'],
        name='Bollinger Lower',
        line=dict(color='gray', dash='dash'),
        opacity=0.5,
        fill='tonexty'  # Fill between upper and lower bands
    ))
    
    # Add RSI subplot
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        yaxis='y2',
        line=dict(color='cyan')
    ))
    
    # Update layout with secondary y-axis for RSI
    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        yaxis2=dict(
            title="RSI",
            overlaying="y",
            side="right",
            range=[0, 100],
            showgrid=False
        ),
        hovermode='x unified',
        showlegend=True
    )
    
    # Add RSI reference lines
    fig.add_hline(y=70, line_dash="dot", line_color="red", opacity=0.3, yref="y2")
    fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.3, yref="y2")
    
    return fig

def create_prediction_figure(
    df, predicted_prices, future_timestamps, forecast_days,
    support_levels, resistance_levels, current_price,
    confidence, pred_change
):
    """
    Create a figure showing the stock price with predictions and support/resistance levels.
    Now handles multiple prediction points.
    """
    # Create candlestick trace
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Stock Price'
    )
    
    # Create prediction trace
    prediction_trace = go.Scatter(
        x=future_timestamps,
        y=predicted_prices,
        mode='lines+markers',
        name=f'Predicted Price ({confidence:.2f} confidence)',
        line=dict(
            color='blue' if pred_change > 0 else 'red',
            dash='dot'
        ),
        marker=dict(
            size=8,
            symbol='diamond'
        )
    )
    
    # Create support level traces
    support_traces = []
    if support_levels:
        for i, level in enumerate(support_levels[-3:], 1):
            support_traces.append(go.Scatter(
                x=[df.index[-1], future_timestamps[-1]],
                y=[level, level],
                mode='lines',
                name=f'Support {i}',
                line=dict(
                    color='green',
                    dash='dash',
                    width=1
                )
            ))
    
    # Create resistance level traces
    resistance_traces = []
    if resistance_levels:
        for i, level in enumerate(resistance_levels[-3:], 1):
            resistance_traces.append(go.Scatter(
                x=[df.index[-1], future_timestamps[-1]],
                y=[level, level],
                mode='lines',
                name=f'Resistance {i}',
                line=dict(
                    color='red',
                    dash='dash',
                    width=1
                )
            ))
    
    # Combine all traces
    data = [candlestick, prediction_trace] + support_traces + resistance_traces
    
    # Create layout
    layout = go.Layout(
        title=f'{forecast_days}-Day Price Prediction',
        yaxis=dict(title='Price'),
        xaxis=dict(title='Date'),
        showlegend=True,
        hovermode='x unified'
    )
    
    return go.Figure(data=data, layout=layout) 