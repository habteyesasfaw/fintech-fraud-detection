import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import requests
from dash.dependencies import Input, Output
from flask import Flask

# Initialize Flask server and then add Dash app
server = Flask(__name__)
app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Fetch summary data from Flask API
summary_url = "http://127.0.0.1:5000/summary"
summary_data = requests.get(summary_url).json()

# Dash layout
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    
    # Summary Statistics Section
    html.Div([
        html.H2("Summary Statistics"),
        html.P(f"Total Transactions: {summary_data['total_transactions']}"),
        html.P(f"Total Fraud Cases: {summary_data['total_fraud']}"),
        html.P(f"Fraud Percentage: {summary_data['fraud_percentage']:.2f}%")
    ], style={'marginBottom': 50}),
    
    # Fraud Trends Line Chart
    html.Div([
        html.H2("Fraud Trends Over Time"),
        dcc.Graph(id="fraud-trend-graph")
    ])
])

# Fetch fraud trend data for visualization
fraud_trends_url = "http://127.0.0.1:5000/fraud_trends"
fraud_trends_data = requests.get(fraud_trends_url).json()
trend_df = pd.DataFrame(fraud_trends_data)

@app.callback(
    Output("fraud-trend-graph", "figure"),
    Input("fraud-trend-graph", "id")
)
def update_fraud_trends(_):
    fig = px.line(
        trend_df,
        x="date",
        y="fraud_cases",
        title="Detected Fraud Cases Over Time",
        labels={'date': 'Date', 'fraud_cases': 'Fraud Cases'},
        template="plotly_dark"
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
