from flask import Flask, render_template
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import pandas as pd
from io import BytesIO, StringIO
import base64
import subprocess
import os
pd.options.display.max_columns = None

def read_ledger(file_path):
    # Call hledger CLI to get transactions in CSV format
    cmd = ['hledger', '-f', file_path, 'register', '--real', '-O', 'csv']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = result.stdout

        if not data.strip():  # Check if the output is empty
            print("No data returned from ledger command.")
            return pd.DataFrame(columns=['txnidx', 'date', 'code', 'description', 'account', 'amount', 'total'])

        try:
            # Convert CSV to DataFrame
            df = pd.read_csv(StringIO(data))
            if df.empty:
                raise ValueError("DataFrame is empty after parsing CSV")
            return df
        except Exception as e:
            print(f"Error parsing CSV data: {e}")
            return pd.DataFrame(columns=['txnidx', 'date', 'code', 'description', 'account', 'amount', 'total'])
    except subprocess.CalledProcessError as e:
        print(f"Error running ledger command: {e}")
        return pd.DataFrame(columns=['txnidx', 'date', 'code', 'description', 'account', 'amount', 'total'])

# Combine transactions based on txnidx
def combine_rows(group):
    combined = {
        #'txnidx': group['txnidx'].iloc[0],
        'date': group['date'].iloc[0],
        'code': group['code'].iloc[0],
        'description': group['description'].iloc[0],
        'debit_account': group.loc[group['amount'] > 0, 'account'].values[0] if not group.loc[group['amount'] > 0].empty else None,
        'credit_account': group.loc[group['amount'] < 0, 'account'].values[0] if not group.loc[group['amount'] < 0].empty else None,
        'amount': group['amount'].iloc[0],  # Assuming amount is the same for all rows in a transaction
        'total': group['total'].iloc[0]
    }
    return pd.Series(combined)

def process_data(df):    
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

    # Remove rows where 'date' is NaT
    df = df.dropna(subset=['date'])

    # Remove non-numeric characters and convert to float
    df['amount'] = df['amount'].replace(r'[^\d.-]', '', regex=True).astype(float)
    
    df = df.groupby('txnidx').apply(combine_rows, include_groups=False).reset_index(drop=True)
    
    # Define income and expense account categories
    income_accounts = ['Erträge']
    expense_accounts = ['Aufwendungen']
    
    # Normalize
    df['amount'] = df['amount'].abs()
    
    # Classify transactions
    # total = 0 if income and expense
    # total = -|amount| if expense
    # total = +|amount| if income
    df['income'] = df.apply(
        lambda x: x['amount'] if x['credit_account'] and any(acc in x['credit_account'] for acc in income_accounts) else 0, axis=1
    )
    df['expenses'] = df.apply(
        lambda x: x['amount'] if x['debit_account'] and any(acc in x['debit_account'] for acc in expense_accounts) else 0, axis=1
    )
    # Calculate the total amount
    df['total'] = df['income'] - df['expenses']
    # Drop the 'income' and 'expenses' columns
    df = df.drop(columns=['income', 'expenses'])
    
    return df

app = Flask(__name__)

file_path = '/home/niklas/org/accounting/accounting.ledger'
print("Loading data...")
df = read_ledger(file_path)
print("Done.")
print("Processing data...")
df = process_data(df)
print("Done.")
print(df[:20])

@app.route('/')
def index():
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)
