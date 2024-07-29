from flask import Flask, render_template
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
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
    
    df_plot = df.copy()

    plt.figure(figsize=(10, 5))

    # Set 'date' column as the index
    df_plot.set_index('date', inplace=True)

    if df_plot.empty:
        return ''  # Return empty string if DataFrame is empty

    # Resample by month and sum the amounts
    df_plot = df_plot['change'].resample('MS').sum()

    if df_plot.empty:
        return ''  # Return empty string if resampled DataFrame is empty

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 5))

    # Define color based on amount
    colors = df_plot.apply(lambda x: 'green' if x > 0 else 'red')

    #df_plot.plot(kind='bar', color=colors, ax=ax)
    bars = ax.bar(df_plot.index, df_plot.values, width=20, color=colors)

    # Add a horizontal line at y=0
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    # Set the locator for the x-axis to be every month
    ax.xaxis.set_major_locator(MonthLocator())
    
    # Format the month labels
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    
    # Ensure the x-axis limits cover the entire date range
    #ax.set_xlim(df_plot.index.min(), df_plot.index.max())
    
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

def get_plot_net_worth(df):
    if df.empty:
        return ''  # Return empty string if DataFrame is empty    
    
    df_plot = df.copy()

    plt.figure(figsize=(10, 5))

    # Ensure 'date' column is in datetime format and set as index
    df_plot.set_index('date', inplace=True)

    if df_plot.empty:
        return ''  # Return empty string if DataFrame is empty

    # Sort the DataFrame by date
    df_plot.sort_index(inplace=True)

    # Extract initial net worth from "Anfangsbestand" account
    initial_net_worth = -df_plot.loc[df_plot['account'] == 'Anfangsbestand']['amount'].sum()

    # Calculate the cumulative sum to get net worth over time, including initial net worth
    df_plot['net_worth'] = df_plot['change'].cumsum() + initial_net_worth

    # Resample by month and get the last value of the month for net worth
    df_plot = df_plot['net_worth'].resample('MS').last().ffill()

    if df_plot.empty:
        return ''  # Return empty string if resampled DataFrame is empty

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df_plot.index, df_plot.values, marker='o', linestyle='-', color='blue', label='Net Worth')

    # Set the locator for the x-axis to be every month
    ax.xaxis.set_major_locator(MonthLocator())

    # Format the month labels
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.title('Net Worth Over Time')
    plt.xlabel('Month')
    plt.ylabel('Net Worth')
    plt.tight_layout()

    # Save plot to a BytesIO object and encode as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def get_plot_expenses(df):
    if df.empty:
        return ''  # Return empty string if DataFrame is empty

    # Filter the DataFrame to include only positive expenses
    expenses_df = df[df['expenses'] > 0]

    if expenses_df.empty:
        return ''  # Return empty string if there are no positive expenses
    
    # Extract the last part of each account name
    expenses_df['subaccount'] = expenses_df['account'].apply(lambda x: x.split(':')[-1])

    # Group by subaccounts and sum the expenses
    expenses_grouped = expenses_df.groupby('subaccount')['expenses'].sum()

    # Sort expenses in descending order and get the least important accounts
    expenses_grouped = expenses_grouped.sort_values(ascending=False)
    threshold = 0.05 * expenses_grouped.sum()  # 5% of the total expenses
    other_expenses = expenses_grouped[expenses_grouped < threshold].sum()
    expenses_grouped = expenses_grouped[expenses_grouped >= threshold]
    expenses_grouped['Other'] = other_expenses

    # Plot the pie chart
    fig, ax = plt.subplots(figsize=(10, 7))

    wedges, texts, autotexts = ax.pie(
        expenses_grouped, labels=expenses_grouped.index, autopct='%1.1f%%', startangle=140,
        colors=plt.cm.tab20.colors[:len(expenses_grouped)], textprops=dict(color="w")
    )

    # Add a legend on the side
    ax.legend(wedges, expenses_grouped.index, title="Accounts", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.title('Expenses by Subaccount')
    plt.tight_layout()

    # Save plot to a BytesIO object and encode as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def get_plot_income(df):
    if df.empty:
        return ''  # Return empty string if DataFrame is empty

    # Filter the DataFrame to include only positive incomes
    income_df = df[df['income'] > 0]

    if income_df.empty:
        return ''  # Return empty string if there are no positive incomes
    
    # Extract the last part of each account name
    income_df['subaccount'] = income_df['account'].apply(lambda x: x.split(':')[-1])

    # Group by subaccounts and sum the income
    income_grouped = income_df.groupby('subaccount')['income'].sum()

    # Sort incomes in descending order and get the least important accounts
    income_grouped = income_grouped.sort_values(ascending=False)
    threshold = 0.05 * income_grouped.sum()  # 5% of the total income
    other_incomes = income_grouped[income_grouped < threshold].sum()
    income_grouped = income_grouped[income_grouped >= threshold]
    income_grouped['Other'] = other_incomes

    # Plot the pie chart
    fig, ax = plt.subplots(figsize=(10, 7))

    wedges, texts, autotexts = ax.pie(
        income_grouped, labels=income_grouped.index, autopct='%1.1f%%', startangle=140,
        colors=plt.cm.tab20.colors[:len(income_grouped)], textprops=dict(color="w")
    )

    # Add a legend on the side
    ax.legend(wedges, income_grouped.index, title="Accounts", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.title('Income by Subaccount')
    plt.tight_layout()

    # Save plot to a BytesIO object and encode as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def get_plot_cash_flow_histogram(df):
    if df.empty:
        return ''  # Return empty string if DataFrame is empty

    plt.figure(figsize=(10, 5))

    # Plot histogram for the 'change' column
    plt.hist(df['change'], bins=30, edgecolor='black', alpha=0.7)
    plt.title('Cash Flow Distribution')
    plt.xlabel('Cash Flow')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()

    # Save plot to a BytesIO object and encode as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def get_plot_cash_flow_heatmap(df, threshold=50):
    if df.empty:
        return ''  # Return empty string if DataFrame is empty

    # Extract the year and month from the date for the heatmap
    df['YearMonth'] = df['date'].dt.to_period('M').astype(str)
    
    # Filter out small values
    df = df[df['change'].abs() >= threshold]

    heatmap_data = df.pivot_table(index='YearMonth', columns='account', values='change', aggfunc='sum', fill_value=0)

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
    plt.title('Cash Flow Heatmap (Filtered)')
    plt.xlabel('Account')
    plt.ylabel('Year-Month')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save plot to a BytesIO object and encode as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')
    
def get_plot_yearly_expense_summary(df, threshold=0.05):
    if df.empty:
        return ''  # Return empty string if DataFrame is empty

    # Extract year from date
    df['Year'] = df['date'].dt.year

    # Filter out negative expenses
    df_expenses = df[df['expenses'] > 0]

    # Simplify account names to the last segment
    df_expenses['account'] = df_expenses['account'].apply(lambda x: x.split(':')[-1] if pd.notnull(x) else x)

    # Group by year and account, then sum expenses
    yearly_expenses = df_expenses.groupby(['Year', 'account'])['expenses'].sum().unstack().fillna(0)
    
    # Calculate total expenses by account across all years
    total_expenses_by_account = yearly_expenses.sum(axis=0)
    
    # Apply threshold to filter accounts
    significant_accounts = total_expenses_by_account[total_expenses_by_account > threshold * total_expenses_by_account.sum()].index
    df_expenses['account'] = df_expenses['account'].apply(lambda x: x if x in significant_accounts else 'Other')
    
    # Regroup by year and account with the 'Other' category
    yearly_expenses_filtered = df_expenses.groupby(['Year', 'account'])['expenses'].sum().unstack().fillna(0)

    plt.figure(figsize=(10, 5))
    
    # Plot horizontal bar chart
    yearly_expenses_filtered.plot(kind='barh', stacked=True)
    
    plt.title('Yearly Expense Summary')
    plt.xlabel('Total Expenses')
    plt.ylabel('Year')
    plt.legend(title='Account', bbox_to_anchor=(1.05, 1), loc='upper left')
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
    plot_net_worth = get_plot_net_worth(df)
    plot_expenses = get_plot_expenses(df)
    plot_income = get_plot_income(df)
    plot_cash_flow_histogram = get_plot_cash_flow_histogram(df)
    plot_cash_flow_heatmap = get_plot_cash_flow_heatmap(df)
    plot_yearly_expense_summary = get_plot_yearly_expense_summary(df)
    
    df_raw_html = df_raw.to_html(classes='table table-striped')
    df_html = df.to_html(classes='table table-striped')
    
    return render_template('index.html', plot_income_expenses=plot_income_expenses, plot_net_worth=plot_net_worth, plot_expenses=plot_expenses, plot_income=plot_income, plot_cash_flow_histogram=plot_cash_flow_histogram, plot_cash_flow_heatmap=plot_cash_flow_heatmap, plot_yearly_expense_summary=plot_yearly_expense_summary,
        df_raw_html=df_raw_html, df_html=df_html)
    
if __name__ == '__main__':
    app.run(debug=True)
