# dashboard/app.py
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from collections import deque
import os
import sys
import networkx as nx
from sklearn.decomposition import PCA
from scipy import stats
# Import the enhanced model
from enhanced_model import load_ensemble_model

# Add the parent directory to the path so Python can find your modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a mock model for demonstration purposes
class MockEnsembleModel:
    """Simple mock model to simulate predictions without real models"""
    def __init__(self):
        print("Initialized mock ensemble model")
        self.weights = {
            "rf": 0.25,
            "xgb": 0.25,
            "isolation_forest": 0.25,
            "autoencoder": 0.25
        }
    
    def predict(self, transaction_df):
        """Generate random prediction scores"""
        num_transactions = len(transaction_df)
        
        # Simulate model predictions
        return {
            'ensemble_score': np.random.beta(1, 10, num_transactions),
            'rf_score': np.random.beta(1, 10, num_transactions),
            'xgb_score': np.random.beta(1, 10, num_transactions),
            'isolation_forest_score_norm': np.random.beta(1, 10, num_transactions),
            'autoencoder_score_norm': np.random.beta(1, 10, num_transactions)
        }
    
    def explain_prediction(self, transaction_df):
        """Provide a mock explanation"""
        score = self.predict(transaction_df)['ensemble_score'][0]
        is_fraud = score > 0.7
        
        explanation = f"{'Suspicious' if is_fraud else 'Normal'} transaction pattern detected."
        
        return {
            'is_fraud': is_fraud,
            'score': float(score),
            'explanation': explanation,
            'model_scores': {
                'Random Forest': float(np.random.beta(1, 10, 1)[0]),
                'XGBoost': float(np.random.beta(1, 10, 1)[0]),
                'Isolation Forest': float(np.random.beta(1, 10, 1)[0]),
                'Autoencoder': float(np.random.beta(1, 10, 1)[0])
            }
        }

# Load a mock model for now (since loading the real model would require trained models)
model = load_ensemble_model("blockchain-fraud-detection/models/ensemble/fraud_detection_model.pkl")

# Initialize the Dash app
app = dash.Dash(__name__, title="Blockchain Fraud Detection Dashboard")

# Global data stores
transactions = []  # All transactions
fraud_alerts = []  # Fraud alerts
transaction_counts = {
    'timestamps': [],
    'normal': [],
    'fraud': []
}

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Blockchain Fraud Detection Dashboard", 
                style={'textAlign': 'center', 'color': '#ffffff'}),
        html.P("Real-time monitoring of blockchain transactions and fraud detection",
              style={'textAlign': 'center', 'color': 'rgba(255, 255, 255, 0.8)'})
    ], className="header"),
    
    # Stats Row
    html.Div([
        # Total Transactions
        html.Div([
            html.H4("Total Transactions"),
            html.H2(id="total-transactions", children="0"),
        ], className="stat-card"),
        
        # Fraud Transactions
        html.Div([
            html.H4("Fraud Detected"),
            html.H2(id="fraud-transactions", children="0"),
        ], className="stat-card"),
        
        # Fraud Rate
        html.Div([
            html.H4("Fraud Rate"),
            html.H2(id="fraud-rate", children="0%"),
        ], className="stat-card"),
        
        # Average Transaction Amount
        html.Div([
            html.H4("Avg Transaction"),
            html.H2(id="avg-amount", children="$0"),
        ], className="stat-card"),
    ], className="stats-container"),
    
    # Charts Row
    html.Div([
        # Transaction Volume Chart
        html.Div([
            html.H3("Transaction Volume"),
            dcc.Graph(id="transaction-volume-chart"),
        ], className="chart-card"),
        
        # Fraud Distribution Chart
        html.Div([
            html.H3("Fraud Score Distribution"),
            dcc.Graph(id="fraud-distribution-chart"),
        ], className="chart-card"),
    ], className="charts-container"),
        
    # Network visualization
    html.Div([
        html.Div([
            html.H3("Transaction Network"),
            dcc.Graph(id="transaction-network-graph"),
        ], className="chart-card full-width"),
    ], className="network-container"),
    
    # 3D visualization
    html.Div([
        html.Div([
            html.H3("3D Transaction Visualization"),
            dcc.Graph(id="3d-visualization", style={"height": "600px"}),
        ], className="chart-card full-width"),
    ], className="visualization-container"),
    
    # Tables Row
    html.Div([
        # Recent Transactions
        html.Div([
            html.H3("Recent Transactions"),
            html.Div(id="transactions-table", className="table-container"),
        ], className="table-card"),
        
        # Fraud Alerts
        html.Div([
            html.H3("Fraud Alerts"),
            html.Div(id="fraud-alerts", className="alerts-container"),
        ], className="alerts-card"),
    ], className="tables-container"),
    
    # Transaction Analysis Tool
    html.Div([
        html.H3("Transaction Analysis"),
        html.Div([
            # Input form
            html.Div([
                html.Label("Transaction ID"),
                dcc.Input(id="tx-id-input", type="text", placeholder="Enter Transaction ID"),
                
                html.Label("Sender ID"),
                dcc.Input(id="sender-id-input", type="text", placeholder="Enter Sender ID"),
                
                html.Label("Receiver ID"),
                dcc.Input(id="receiver-id-input", type="text", placeholder="Enter Receiver ID"),
                
                html.Label("Amount"),
                dcc.Input(id="amount-input", type="number", placeholder="Enter Amount"),
                
                html.Button("Analyze", id="analyze-button", n_clicks=0),
            ], className="analysis-form"),
            
            # Analysis results
            html.Div([
                html.Div(id="analysis-results", className="analysis-results"),
            ], className="analysis-results-container"),
        ], className="analysis-container"),
    ], className="analysis-card"),
    
    # Footer
    html.Footer([
        html.Div([
            html.P([
                "Blockchain Fraud Detection Dashboard",
                html.Span("v1.0", className="version-tag")
            ], className="footer-title"),
            html.P("© 2025 AI-Driven Fraud Detection System", className="copyright")
        ], className="footer-content")
    ], className="footer"),
    
    # Loading component
    dcc.Loading(
        id="loading-component",
        type="circle",
        color="#1de9b6",
        children=[
            html.Div(id="loading-output", style={"display": "none"})
        ]
    ),
    
    # Hidden divs for storing data
    html.Div(id='transactions-store', style={'display': 'none'}),
    html.Div(id='alerts-store', style={'display': 'none'}),
    
    # Update interval
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # update every 5 seconds
        n_intervals=0
    )
], className="dashboard-container")

# Callbacks

# Data update callback
@app.callback(
    [Output('transactions-store', 'children'),
     Output('alerts-store', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_data(n):
    global transactions, fraud_alerts, transaction_counts
    
    # In a real implementation, you would fetch data from your real-time pipeline
    # Here we're simulating incoming transactions
    current_time = datetime.now()
    
    # Simulate 1-5 new transactions every update
    num_new_tx = np.random.randint(1, 6)
    new_transactions = []
    
    for i in range(num_new_tx):
        # Create random transaction data
        tx_id = f"tx_{len(transactions) + i + 1}"
        sender = f"user_{np.random.randint(1, 100)}"
        receiver = f"user_{np.random.randint(1, 100)}"
        
        # Force some transactions to have high amounts and be fraudulent
        if np.random.random() < 0.15:  # Every third transaction will be fraudulent
            amount = round(np.random.gamma(10, 300), 2)  # Higher amounts for fraudulent transactions
            fraud_score = np.random.uniform(0.7, 0.95)  # Force high fraud score
            is_fraud = True
        else:
            amount = round(np.random.gamma(2, 100), 2)  # Normal amounts
            fraud_score = np.random.uniform(0.05, 0.4)  # Lower fraud scores
            is_fraud = False
        
        # Create transaction object
        tx = {
            'Transaction_ID': tx_id,
            'Sender_ID': sender,
            'Receiver_ID': receiver,
            'Transaction_Amount': amount,
            'Timestamp': current_time.isoformat(),
            'Time_Display': current_time.strftime("%H:%M:%S"),
            'Fraud_Score': fraud_score,
            'Is_Fraud': is_fraud
        }
    
    # Skip the model prediction since we're manually setting values
    new_transactions.append(tx)
    # Add to transaction history
    transactions.extend(new_transactions)
    
    # Add fraud alerts
    for tx in new_transactions:
        if tx['Is_Fraud']:
            alert = tx.copy()
            alert['Alert_Time'] = current_time.isoformat()
            alert['Description'] = "Suspicious transaction pattern detected"
            fraud_alerts.append(alert)
    
    # Update transaction counts for the time series
    normal_count = len([tx for tx in new_transactions if not tx['Is_Fraud']])
    fraud_count = len([tx for tx in new_transactions if tx['Is_Fraud']])
    
    transaction_counts['timestamps'].append(current_time.strftime("%H:%M:%S"))
    transaction_counts['normal'].append(normal_count)
    transaction_counts['fraud'].append(fraud_count)
    
    # Limit the history size to prevent memory issues
    max_history = 1000
    if len(transactions) > max_history:
        transactions = transactions[-max_history:]
    if len(fraud_alerts) > 100:
        fraud_alerts = fraud_alerts[-100:]
    
    # Limit time series data points
    max_points = 20
    if len(transaction_counts['timestamps']) > max_points:
        transaction_counts['timestamps'] = transaction_counts['timestamps'][-max_points:]
        transaction_counts['normal'] = transaction_counts['normal'][-max_points:]
        transaction_counts['fraud'] = transaction_counts['fraud'][-max_points:]
    
    # Return serialized data for the Dash store components
    return json.dumps(transactions), json.dumps(fraud_alerts)

# Stats update callback
@app.callback(
    [Output('total-transactions', 'children'),
     Output('fraud-transactions', 'children'),
     Output('fraud-rate', 'children'),
     Output('avg-amount', 'children')],
    [Input('transactions-store', 'children')]
)
def update_stats(transactions_json):
    if not transactions_json:
        return "0", "0", "0%", "$0"
    
    transactions = json.loads(transactions_json)
    
    # Calculate statistics
    total_tx = len(transactions)
    fraud_tx = len([tx for tx in transactions if tx['Is_Fraud']])
    
    if total_tx > 0:
        fraud_rate = f"{(fraud_tx / total_tx) * 100:.1f}%"
        avg_amount = f"${np.mean([tx['Transaction_Amount'] for tx in transactions]):.2f}"
    else:
        fraud_rate = "0%"
        avg_amount = "$0"
    
    return str(total_tx), str(fraud_tx), fraud_rate, avg_amount

# Transaction volume chart callback
@app.callback(
    Output('transaction-volume-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_transaction_volume(n):
    global transaction_counts
    
    # Create a more visually appealing time series chart
    fig = go.Figure()
    
    # Add area charts for better visual effect
    fig.add_trace(go.Scatter(
        x=transaction_counts['timestamps'],
        y=transaction_counts['normal'],
        name='Normal',
        mode='lines',
        line=dict(width=2, color='rgba(105, 240, 174, 1)'),
        fill='tozeroy',
        fillcolor='rgba(105, 240, 174, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=transaction_counts['timestamps'],
        y=transaction_counts['fraud'],
        name='Fraud',
        mode='lines',
        line=dict(width=2, color='rgba(255, 82, 82, 1)'),
        fill='tozeroy',
        fillcolor='rgba(255, 82, 82, 0.2)'
    ))
    
    # Add markers for the most recent data points
    if transaction_counts['timestamps']:
        last_idx = -1
        fig.add_trace(go.Scatter(
            x=[transaction_counts['timestamps'][last_idx]],
            y=[transaction_counts['normal'][last_idx]],
            mode='markers',
            marker=dict(color='rgba(105, 240, 174, 1)', size=10, symbol='circle'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=[transaction_counts['timestamps'][last_idx]],
            y=[transaction_counts['fraud'][last_idx]],
            mode='markers',
            marker=dict(color='rgba(255, 82, 82, 1)', size=10, symbol='circle'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Enhanced layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Time",
        yaxis_title="Number of Transactions",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)'
        ),
        hovermode='x unified'
    )
    
    return fig

# Fraud distribution chart callback
@app.callback(
    Output('fraud-distribution-chart', 'figure'),
    [Input('transactions-store', 'children')]
)
def update_fraud_distribution(transactions_json):
    if not transactions_json:
        return go.Figure()
    
    transactions = json.loads(transactions_json)
    
    if not transactions:
        return go.Figure()
    
    # Get fraud scores
    scores = [tx['Fraud_Score'] for tx in transactions]
    
    # Create a more visually appealing histogram
    fig = go.Figure()
    
    # Add histogram trace with enhanced styling
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=20,
        marker=dict(
            color='rgba(0, 176, 255, 0.6)',
            line=dict(color='rgba(0, 176, 255, 1)', width=1)
        ),
        opacity=0.7,
        hovertemplate='Score: %{x}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add KDE (kernel density estimation) curve for better visualization
    hist, bin_edges = np.histogram(scores, bins=20, range=(0, 1), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Smooth the histogram with a kernel density estimate
    if len(scores) > 1:  # Need at least 2 points for KDE
        x_kde = np.linspace(0, 1, 100)
        kde = stats.gaussian_kde(scores)
        y_kde = kde(x_kde)
        
        # Scale KDE to match histogram height
        scale_factor = max(hist) / max(y_kde) if max(y_kde) > 0 else 1
        y_kde = y_kde * scale_factor
        
        fig.add_trace(go.Scatter(
            x=x_kde,
            y=y_kde,
            mode='lines',
            line=dict(color='rgba(29, 233, 182, 1)', width=3),
            name='Density',
            hoverinfo='skip'
        ))
    
    # Add threshold line with animation effect
    fig.add_shape(
        type="line",
        x0=0.7, y0=0, x1=0.7, y1=1,
        yref="paper",
        line=dict(
            color="rgba(255, 82, 82, 1)",
            width=2,
            dash="dash"
        )
    )
    
    # Add threshold label
    fig.add_annotation(
        x=0.7,
        y=1,
        yref="paper",
        text="Fraud Threshold",
        showarrow=True,
        arrowhead=2,
        arrowcolor="rgba(255, 82, 82, 1)",
        arrowsize=1,
        arrowwidth=2,
        ax=50,
        ay=-30,
        font=dict(color="rgba(255, 82, 82, 1)", size=12)
    )
    
    # Enhanced layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Fraud Score",
        yaxis_title="Number of Transactions",
        bargap=0.1,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgba(255,255,255,0.2)',
            range=[0, 1]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            showline=True,
            linecolor='rgba(255,255,255,0.2)'
        ),
        showlegend=False
    )
    
    return fig

# Transaction network graph callback
@app.callback(
    Output('transaction-network-graph', 'figure'),
    [Input('transactions-store', 'children')]
)
def update_network_graph(transactions_json):
    if not transactions_json:
        return go.Figure()
    
    transactions = json.loads(transactions_json)
    
    if not transactions:
        return go.Figure()
    
    # Limit to last 50 transactions for performance
    recent_tx = transactions[-50:]
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes and edges
    for tx in recent_tx:
        sender = tx['Sender_ID']
        receiver = tx['Receiver_ID']
        amount = tx['Transaction_Amount']
        is_fraud = tx['Is_Fraud']
        
        # Add nodes if they don't exist
        if sender not in G.nodes:
            G.add_node(sender, type='sender')
        if receiver not in G.nodes:
            G.add_node(receiver, type='receiver')
        
        # Add edge
        G.add_edge(sender, receiver, amount=amount, is_fraud=is_fraud, 
                  weight=1.0 + amount/100.0)  # Weight affects edge length
    
    # Create positions - use a layout algorithm
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Color nodes by type
        if G.nodes[node]['type'] == 'sender':
            node_colors.append('rgba(0, 176, 255, 0.8)')
        else:
            node_colors.append('rgba(106, 27, 154, 0.8)')
        
        # Size nodes by degree
        degree = G.degree(node)
        node_sizes.append(10 + 5 * degree)
        
        # Node hover text
        node_text.append(f"ID: {node}<br>Transactions: {degree}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color='rgba(255, 255, 255, 0.5)')
        )
    )
    
    # Create edge traces (separate trace for normal and fraud)
    edge_x_normal = []
    edge_y_normal = []
    edge_x_fraud = []
    edge_y_fraud = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        is_fraud = G.edges[edge]['is_fraud']
        amount = G.edges[edge]['amount']
        
        edge_text.append(f"From: {edge[0]}<br>To: {edge[1]}<br>Amount: ${amount:.2f}")
        
        if is_fraud:
            edge_x_fraud.extend([x0, x1, None])
            edge_y_fraud.extend([y0, y1, None])
        else:
            edge_x_normal.extend([x0, x1, None])
            edge_y_normal.extend([y0, y1, None])
    
    edge_trace_normal = go.Scatter(
        x=edge_x_normal, y=edge_y_normal,
        line=dict(width=1, color='rgba(105, 240, 174, 0.4)'),
        hoverinfo='none',
        mode='lines'
    )
    
    edge_trace_fraud = go.Scatter(
        x=edge_x_fraud, y=edge_y_fraud,
        line=dict(width=2, color='rgba(255, 82, 82, 0.8)'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace_normal, edge_trace_fraud, node_trace],
                  layout=go.Layout(
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      plot_bgcolor='rgba(0,0,0,0)',
                      paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='white'),
                      title=dict(
                          text="Blockchain Transaction Network",
                          font=dict(size=16),
                          x=0.5,
                          y=0.95
                      ),
                      annotations=[
                          dict(
                              text="• Sender • Receiver<br>Red lines indicate potential fraud",
                              showarrow=False,
                              xref="paper", yref="paper",
                              x=0.01, y=0.01,
                              font=dict(size=12, color='rgba(255,255,255,0.7)')
                          )
                      ]
                  ))
    
    return fig

# 3D visualization callback
@app.callback(
    Output('3d-visualization', 'figure'),
    [Input('transactions-store', 'children')]
)
def update_3d_visualization(transactions_json):
    if not transactions_json:
        return go.Figure()
    
    transactions = json.loads(transactions_json)
    
    if len(transactions) < 5:  # Need at least a few points for PCA
        return go.Figure()
    
    # Extract features for visualization
    features = []
    for tx in transactions:
        # Create numerical features from the transaction
        amount = tx['Transaction_Amount']
        fraud_score = tx['Fraud_Score']
        
        # Create additional features from transaction ID and timestamp
        tx_id_hash = hash(tx['Transaction_ID']) % 1000 / 1000.0
        timestamp = datetime.fromisoformat(tx['Timestamp'])
        time_feature = timestamp.hour + timestamp.minute / 60.0
        
        # Create synthetic features for visualization
        sender_hash = hash(tx['Sender_ID']) % 1000 / 1000.0
        receiver_hash = hash(tx['Receiver_ID']) % 1000 / 1000.0
        
        features.append([
            amount, 
            fraud_score, 
            tx_id_hash,
            time_feature,
            sender_hash,
            receiver_hash
        ])
    
    # Convert to array
    feature_array = np.array(features)
    
    # Use PCA to reduce to 3 dimensions
    if len(feature_array) > 3:
        pca = PCA(n_components=3)
        coords = pca.fit_transform(feature_array)
    else:
        # If not enough samples, just use first 3 features
        coords = feature_array[:, :3]
    
    # Extract coordinates
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    
    # Create colors based on fraud score
    colors = [
        f'rgba({int(255*score)}, {int(255*(1-score))}, 100, 0.8)'
        for score in [tx['Fraud_Score'] for tx in transactions]
    ]
    
    # Create hover text
    hover_text =  [
        f"ID: {tx['Transaction_ID']}<br>" +
        f"Amount: ${tx['Transaction_Amount']:.2f}<br>" +
        f"Fraud Score: {tx['Fraud_Score']:.3f}<br>" +
        f"Sender: {tx['Sender_ID']}<br>" +
        f"Receiver: {tx['Receiver_ID']}"
        for tx in transactions
    ]
    
    # Create figure
    fig = go.Figure(data=[
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                opacity=0.8,
                line=dict(
                    color='rgba(255, 255, 255, 0.5)',
                    width=0.5
                )
            ),
            text=hover_text,
            hoverinfo='text'
        )
    ])
    
    # Add clusters for better visualization
    # Separate fraud and non-fraud points
    fraud_indices = [i for i, tx in enumerate(transactions) if tx['Is_Fraud']]
    non_fraud_indices = [i for i, tx in enumerate(transactions) if not tx['Is_Fraud']]
    
    if fraud_indices:
        # Calculate centroid of fraud points
        fraud_centroid = np.mean(coords[fraud_indices], axis=0)
        
        # Add a subtle sphere around fraud cluster
        fig.add_trace(go.Mesh3d(
            x=[fraud_centroid[0]], 
            y=[fraud_centroid[1]], 
            z=[fraud_centroid[2]],
            alphahull=5,
            opacity=0.1,
            color='red',
            hoverinfo='none'
        ))
    
    # Add animated camera paths
    camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.5)
    )
    
    # Enhanced layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='rgba(255, 255, 255, 0.1)',
                showbackground=False,
                zerolinecolor='rgba(255, 255, 255, 0.1)'
            ),
            yaxis=dict(
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='rgba(255, 255, 255, 0.1)',
                showbackground=False,
                zerolinecolor='rgba(255, 255, 255, 0.1)'
            ),
            zaxis=dict(
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='rgba(255, 255, 255, 0.1)',
                showbackground=False,
                zerolinecolor='rgba(255, 255, 255, 0.1)'
            ),
            camera=camera
        ),
        scene_aspectmode='cube'
    )
    
    return fig

# Transactions table callback
@app.callback(
    Output('transactions-table', 'children'),
    [Input('transactions-store', 'children')]
)
def update_transactions_table(transactions_json):
    if not transactions_json:
        return html.P("No transactions available")
    
    transactions = json.loads(transactions_json)
    
    if not transactions:
        return html.P("No transactions available")
    
    # Get most recent transactions (limit to 10)
    recent_tx = transactions[-10:][::-1]  # Reverse to show newest first
    
    # Create table
    table = html.Table([
        html.Thead(
            html.Tr([
                html.Th("ID"),
                html.Th("Time"),
                html.Th("Sender"),
                html.Th("Receiver"),
                html.Th("Amount"),
                html.Th("Fraud Score")
            ])
        ),
        html.Tbody([
            html.Tr([
                html.Td(tx['Transaction_ID']),
                html.Td(tx['Time_Display']),
                html.Td(tx['Sender_ID']),
                html.Td(tx['Receiver_ID']),
                html.Td(f"${tx['Transaction_Amount']:.2f}"),
                html.Td(
                    f"{tx['Fraud_Score']:.3f}",
                    style={'color': '#ff5252' if tx['Is_Fraud'] else '#69f0ae'}
                )
            ]) for tx in recent_tx
        ])
    ])
    
    return table

# Fraud alerts callback
@app.callback(
    Output('fraud-alerts', 'children'),
    [Input('alerts-store', 'children')]
)
def update_fraud_alerts(alerts_json):
    if not alerts_json:
        return html.P("No fraud alerts")
    
    alerts = json.loads(alerts_json)
    
    if not alerts:
        return html.P("No fraud alerts")
    
    # Take most recent alerts (limit to 5)
    recent_alerts = alerts[-5:][::-1]  # Reverse to show newest first
    
    # Create alert cards
    alerts_list = [
        html.Div([
            html.Div([
                html.Span("FRAUD ALERT", className="alert-badge"),
                html.Span(alert['Time_Display'], className="alert-time")
            ], className="alert-header"),
            html.P(f"Transaction {alert['Transaction_ID']} from {alert['Sender_ID']} to {alert['Receiver_ID']}"),
            html.P(f"Amount: ${alert['Transaction_Amount']:.2f} • Score: {alert['Fraud_Score']:.3f}"),
            html.P(alert['Description'], className="alert-description")
        ], className="alert-card") for alert in recent_alerts
    ]
    
    return alerts_list

# Transaction analysis helper functions
def get_color_for_score(score):
    """Get color for score based on value"""
    if score > 0.7:
        return 'var(--danger-color)'
    elif score > 0.4:
        return 'var(--warning-color)'
    else:
        return 'var(--success-color)'

def get_recommendation(risk_class, option):
    """Get recommendations based on risk class"""
    recommendations = {
        'high-risk': [
            "Block transaction immediately and flag for review",
            "Conduct enhanced due diligence on both sender and receiver",
            "Notify compliance department for investigation"
        ],
        'medium-risk': [
            "Apply additional verification steps before processing",
            "Monitor sender's account for additional suspicious activity",
            "Consider reducing transaction limits temporarily"
        ],
        'low-risk': [
            "Process transaction normally",
            "Include in routine monitoring cycles",
            "No additional action required"
        ]
    }
    return recommendations.get(risk_class, ["No recommendation available"])[option-1]
# Transaction analysis callback
@app.callback(
    Output('analysis-results', 'children'),
    [Input('analyze-button', 'n_clicks')],
    [State('tx-id-input', 'value'),
     State('sender-id-input', 'value'),
     State('receiver-id-input', 'value'),
     State('amount-input', 'value')]
)
def analyze_transaction(n_clicks, tx_id, sender_id, receiver_id, amount):
    if n_clicks == 0:
        return html.Div([
            html.Div(className="analysis-placeholder", children=[
                html.Div(className="analysis-icon"),
                html.P("Enter transaction details and click Analyze to evaluate fraud risk")
            ])
        ])
    
    if not tx_id or not sender_id or not receiver_id or not amount:
        return html.Div([
            html.P("Please fill in all fields", 
                  style={'color': 'var(--danger-color)', 'fontWeight': 'bold'})
        ])
    
    # Create transaction object for analysis
    tx = {
        'Transaction_ID': tx_id,
        'Sender_ID': sender_id,
        'Receiver_ID': receiver_id,
        'Transaction_Amount': float(amount),
        'Timestamp': datetime.now().isoformat(),
    }
    
    # Analyze transaction
    tx_df = pd.DataFrame([tx])
    
    try:
        # Calculate a deterministic fraud score based on transaction properties instead of random
        # This approach uses properties of the transaction to determine the score
        
        # Use amount as a primary factor - higher amounts have higher risk
        amount_factor = min(0.6, float(amount) / 1000)  # Cap at 0.6
        
        # Use sender/receiver IDs as factors
        sender_hash = hash(sender_id) % 100 / 100
        receiver_hash = hash(receiver_id) % 100 / 100
        
        # Calculate fraud score based on these factors
        if "suspicious" in sender_id.lower() or "suspicious" in receiver_id.lower():
            # Keywords trigger high score
            fraud_score = 0.85
        elif float(amount) > 800:
            # Very high amounts have higher scores
            fraud_score = 0.7 + (amount_factor * 0.2)
        elif float(amount) > 400:
            # High amounts have medium-high scores
            fraud_score = 0.5 + (amount_factor * 0.3)
        else:
            # Normal transactions have lower scores
            fraud_score = amount_factor + (sender_hash * 0.2) + (receiver_hash * 0.1)
        
        # Ensure score is between 0 and 1
        fraud_score = max(0, min(0.95, fraud_score))
        is_fraud = fraud_score > 0.7
        
        # Create a consistent explanation based on the inputs
        if is_fraud:
            explanation_text = f"Transaction of ${float(amount):.2f} from {sender_id} to {receiver_id} shows unusual patterns compared to normal transactions. The high amount and sender profile indicate potential risk."
        elif fraud_score > 0.4:
            explanation_text = f"Transaction shows some unusual characteristics based on the amount (${float(amount):.2f}) and transaction pattern. Additional verification may be needed."
        else:
            explanation_text = f"Transaction of ${float(amount):.2f} shows normal patterns. No suspicious activity detected in the transaction flow."
        
        # Create mock model scores that align with the overall fraud score
        model_scores = {
            'Random Forest': max(0.1, min(0.9, fraud_score + np.random.uniform(-0.1, 0.1))),
            'XGBoost': max(0.1, min(0.9, fraud_score + np.random.uniform(-0.1, 0.1))),
            'Isolation Forest': max(0.1, min(0.9, fraud_score + np.random.uniform(-0.15, 0.05))),
            'Autoencoder': max(0.1, min(0.9, fraud_score + np.random.uniform(-0.05, 0.15)))
        }
        
        # Create explanation object similar to what explain_prediction would return
        explanation = {
            'is_fraud': is_fraud,
            'score': float(fraud_score),
            'explanation': explanation_text,
            'model_scores': model_scores
        }
        
        # Set result styling based on score
        if is_fraud:
            risk_level = "High Risk"
            risk_color = 'var(--danger-color)'
            risk_icon = "⚠️"
            risk_class = "high-risk"
        elif fraud_score > 0.4:
            risk_level = "Medium Risk"
            risk_color = 'var(--warning-color)'
            risk_icon = "⚠️"
            risk_class = "medium-risk"
        else:
            risk_level = "Low Risk"
            risk_color = 'var(--success-color)'
            risk_icon = "✓"
            risk_class = "low-risk"
        
        # Create result display with visual gauge for the score
        gauge_percentage = min(100, max(0, fraud_score * 100))
        
        return html.Div([
            # Risk header with icon
            html.Div([
                html.Span(risk_icon, className="risk-icon"),
                html.H4(risk_level, style={'color': risk_color, 'margin': '0 0 0 10px'})
            ], className="risk-header"),
            
            # Visual gauge
            html.Div([
                html.Div(className="gauge-background"),
                html.Div(className="gauge-fill", 
                         style={'width': f'{gauge_percentage}%', 'background-color': risk_color}),
                html.Div(f"Fraud Score: {fraud_score:.3f}", className="gauge-label")
            ], className="gauge-container"),
            
            # Risk factors
            html.Div([
                html.H5("Risk Factors:"),
                html.P(explanation.get('explanation', "No detailed explanation available"), 
                      className="explanation-text")
            ], className="risk-factors"),
            
            # Model contributions as horizontal bars
            html.Div([
                html.H5("Model Contributions:"),
                html.Div([
                    html.Div([
                        html.Span(model_name, className="model-name"),
                        html.Div(className="bar-container", children=[
                            html.Div(className="bar-fill", 
                                    style={'width': f'{score*100}%', 
                                          'background-color': get_color_for_score(score)})
                        ]),
                        html.Span(f"{score:.3f}", className="score-value")
                    ], className="model-score-item")
                    for model_name, score in explanation.get('model_scores', {}).items()
                ], className="model-scores-container")
            ], className="model-contributions"),
            
            # Action recommendations based on risk level
            html.Div([
                html.H5("Recommended Actions:"),
                html.Ul([
                    html.Li(get_recommendation(risk_class, 1)),
                    html.Li(get_recommendation(risk_class, 2)),
                    html.Li(get_recommendation(risk_class, 3))
                ], className="recommendations-list")
            ], className="recommendations")
            
        ], className=f"analysis-result {risk_class}")
    
    except Exception as e:
        return html.Div([
            html.P(f"Error analyzing transaction: {str(e)}", 
                  style={'color': 'var(--danger-color)', 'fontWeight': 'bold'})
        ])

# Loading state callback
@app.callback(
    Output("loading-output", "children"),
    [Input("interval-component", "n_intervals")]
)
def update_loading_state(n):
    # This function doesn't need to do anything,
    # it just triggers the loading animation
    time.sleep(0.5)  # Simulate loading delay
    return ""

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)