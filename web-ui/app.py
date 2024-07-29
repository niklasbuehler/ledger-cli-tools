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
    cmd = ['hledger', '-f', file_path, 'register', '--real', '-O', 'csv'] # , '-b', '2024-03-01', '-e', '2024-03-31']
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

def process_data(df):    
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')

    # Remove rows where 'date' is NaT
    df = df.dropna(subset=['date'])

    # Remove non-numeric characters and convert to float
    df['amount'] = df['amount'].replace(r'[^\d.-]', '', regex=True).astype(float)
    df['total'] = df['total'].replace(r'[^\d.-]', '', regex=True).astype(float)
    
    # Define income and expense account categories
    income_accounts = ['Erträge']
    expense_accounts = ['Aufwendungen']
    
    # Classify transactions
    df['income'] = df.apply(
        lambda x: -x['amount'] if x['account'] and any(acc in x['account'] for acc in income_accounts) else 0, axis=1
    )
    df['expenses'] = df.apply(
        lambda x: x['amount'] if x['account'] and any(acc in x['account'] for acc in expense_accounts) else 0, axis=1
    )
    # Calculate the total amount
    df['change'] = df['income'] - df['expenses']
    # Drop the 'income' and 'expenses' columns
    #df = df.drop(columns=['income', 'expenses'])
    
    return df
    
def get_plot_income_expenses(df):
    if df.empty:
        return ''  # Return empty string if DataFrame is empty

    plt.figure(figsize=(10, 5))

    # Set 'date' column as the index
    df.set_index('date', inplace=True)

    if df.empty:
        return ''  # Return empty string if DataFrame is empty

    # Resample by month and sum the amounts
    df_resampled = df['change'].resample('MS').sum()

    if df_resampled.empty:
        return ''  # Return empty string if resampled DataFrame is empty
    
    print(df_resampled)

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 5))

    # Define color based on amount
    colors = df_resampled.apply(lambda x: 'green' if x > 0 else 'red')

    #df_resampled.plot(kind='bar', color=colors, ax=ax)
    bars = ax.bar(df_resampled.index, df_resampled.values, width=20, color=colors)

    # Add a horizontal line at y=0
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Set the locator for the x-axis to be every month
    ax.xaxis.set_major_locator(MonthLocator())
    
    # Format the month labels
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    
    # Ensure the x-axis limits cover the entire date range
    #ax.set_xlim(df_resampled.index.min(), df_resampled.index.max())
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.title('Monthly Income/Expenses')
    plt.xlabel('Month')
    plt.ylabel('Amount')
    plt.tight_layout()

    # Save plot to a BytesIO object and encode as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

app = Flask(__name__)

file_path = '/home/niklas/org/accounting/accounting.ledger'
print("Loading data...")
df_raw = read_ledger(file_path)
print("Done.")
print("Processing data...")
df = process_data(df_raw)
print("Done.")

@app.route('/')
def index():
    plot_income_expenses = get_plot_income_expenses(df)
    df_raw_html = df_raw.to_html(classes='table table-striped')
    df_html = df.to_html(classes='table table-striped')
    return render_template('index.html', plot_income_expenses=plot_income_expenses, df_raw_html=df_raw_html, df_html=df_html)
    
if __name__ == '__main__':
    app.run(debug=True)
