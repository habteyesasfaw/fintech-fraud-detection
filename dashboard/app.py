from flask import Flask, jsonify, request
import pandas as pd

app = Flask(__name__)

# Load data
fraud_data = pd.read_csv('data/Fraud_Data.csv')  # Load your main fraud data
credit_data = pd.read_csv('data/creditcard.csv')  # Load credit card fraud data

@app.route('/summary', methods=['GET'])
def get_summary():
    total_transactions = len(fraud_data)
    total_fraud = fraud_data['class'].sum()  # Assuming 1 represents fraud
    fraud_percentage = (total_fraud / total_transactions) * 100
    
    summary = {
        "total_transactions": total_transactions,
        "total_fraud": total_fraud,
        "fraud_percentage": fraud_percentage
    }
    return jsonify(summary)

@app.route('/fraud_trends', methods=['GET'])
def fraud_trends():
    trends = fraud_data.groupby(fraud_data['purchase_time'].str[:10]).agg(
        fraud_cases=('class', 'sum'),
        total_transactions=('class', 'count')
    ).reset_index()
    trends.columns = ['date', 'fraud_cases', 'total_transactions']
    trends['fraud_percentage'] = (trends['fraud_cases'] / trends['total_transactions']) * 100
    return trends.to_json(orient="records")

if __name__ == '__main__':
    app.run(port=5000)
