import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import subprocess
from io import StringIO

# Initialize Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

def read_ledger(file_path):
    cmd = ['hledger', '-f', file_path, 'register', '--real', '-O', 'csv']
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = result.stdout

        if not data.strip():
            print("No data returned from ledger command.")
            return pd.DataFrame(columns=['txnidx', 'date', 'code', 'description', 'account', 'amount', 'total'])

        try:
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
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['date'])

    df['amount'] = df['amount'].replace(r'[^\d.-]', '', regex=True).astype(float)
    df['total'] = df['total'].replace(r'[^\d.-]', '', regex=True).astype(float)
    
    income_accounts = ['Erträge']
    expense_accounts = ['Aufwendungen']
    
    df['income'] = df.apply(
        lambda x: -x['amount'] if x['account'] and any(acc in x['account'] for acc in income_accounts) else 0, axis=1
    )
    df['expenses'] = df.apply(
        lambda x: x['amount'] if x['account'] and any(acc in x['account'] for acc in expense_accounts) else 0, axis=1
    )
    df['change'] = df['income'] - df['expenses']
    
    return df

file_path = '/home/niklas/org/accounting/accounting.ledger'
print("Loading data...")
df_raw = read_ledger(file_path)
print("Done.")
print("Processing data...")
df = process_data(df_raw)
print("Done.")

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname'),
    Input('date-picker-global', 'start_date'),
    Input('date-picker-global', 'end_date')
)
def display_page(pathname, start_date, end_date):
    df_filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    
    if pathname == '/':
        return html.Div([
            dcc.Graph(
                id='general-income-expenses-bar',
                figure=update_general_plots(start_date, end_date)[0]
            ),
            dcc.Graph(
                id='general-net-worth-line',
                figure=update_general_plots(start_date, end_date)[1]
            )
        ])
    
    elif pathname == '/income':
        return html.Div([
            dcc.Graph(
                id='income-pie-chart',
                figure=update_income_plots(start_date, end_date)[0]
            ),
            dcc.Graph(
                id='income-summary-month',
                figure=update_income_plots(start_date, end_date)[1]
            ),
            dcc.Graph(
                id='income-summary-year',
                figure=update_income_plots(start_date, end_date)[2]
            )
        ])
    
    elif pathname == '/expenses':
        return html.Div([
            dcc.Graph(
                id='expense-pie-chart',
                figure=update_expense_plots(start_date, end_date)[0]
            ),
            dcc.Graph(
                id='expense-summary-month',
                figure=update_expense_plots(start_date, end_date)[1]
            ),
            dcc.Graph(
                id='expense-summary-year',
                figure=update_expense_plots(start_date, end_date)[2]
            )
        ])
    
    return html.Div([
        html.H3("404: Page Not Found")
    ])

@app.callback(
    Output('general-income-expenses-bar', 'figure'),
    Output('general-net-worth-line', 'figure'),
    Input('date-picker-global', 'start_date'),
    Input('date-picker-global', 'end_date')
)
def update_general_plots(start_date, end_date):
    df_filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

    # Income/Expenses Bar Plot
    if df_filtered.empty:
        return go.Figure(), go.Figure()

    df_plot = df_filtered.set_index('date')['change'].resample('MS').sum()
    colors = df_plot.apply(lambda x: 'green' if x > 0 else 'red')

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=df_plot.index,
        y=df_plot.values,
        marker_color=colors,
        name='Income/Expenses'
    ))
    bar_fig.update_layout(
        title='Monthly Income/Expenses',
        xaxis_title='Month',
        yaxis_title='Amount',
        xaxis_tickformat='%Y-%m',
        xaxis_dtick='M1',
        xaxis_tickangle=-45,
        barmode='relative'
    )

    # Net Worth Line Plot
    initial_net_worth = -df_filtered.loc[df_filtered['account'] == 'Anfangsbestand']['amount'].sum()
    df_filtered['net_worth'] = df_filtered['change'].cumsum() + initial_net_worth
    colors = df_filtered['net_worth'].apply(lambda x: 'green' if x > 0 else 'red')
    
    net_worth_fig = go.Figure(data=[
        go.Scatter(x=df_filtered['date'], y=df_filtered['net_worth'], mode='lines+markers', marker_color=colors, name='Net Worth')
    ])
    net_worth_fig.update_layout(
        title='Net Worth Over Time',
        xaxis_title='Date',
        yaxis_title='Net Worth'
    )

    return bar_fig, net_worth_fig

@app.callback(
    Output('income-pie-chart', 'figure'),
    Output('income-summary-month', 'figure'),
    Output('income-summary-year', 'figure'),
    Input('date-picker-global', 'start_date'),
    Input('date-picker-global', 'end_date')
)
def update_income_plots(start_date, end_date):
    df_filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    df_income = df_filtered[df_filtered['income'] != 0]
    
    df_income['account'] = df_income['account'].apply(lambda x: x.split(':')[-1])
    df_income['Year'] = df_income['date'].dt.year
    df_income['Month'] = df_income['date'].dt.strftime('%Y-%m')

    # Income Pie Chart
    income_summary = df_income.groupby('account')['income'].sum().reset_index()
    pie_fig = go.Figure(data=[go.Pie(labels=income_summary['account'], values=income_summary['income'])])
    pie_fig.update_layout(title='Income Distribution')

    # Income Summary (Monthly)
    monthly_income = df_income.groupby(['Month', 'account'])['income'].sum().unstack().fillna(0)
    
    monthly_income_summary_fig = go.Figure()
    for col in monthly_income.columns:
        monthly_income_summary_fig.add_trace(go.Bar(
            x=monthly_income.index,
            y=monthly_income[col],
            name=col
        ))

    monthly_income_summary_fig.update_layout(
        title='Monthly Income Summary',
        xaxis_title='Month',
        yaxis_title='Total Income',
        xaxis_tickformat='%Y-%m',
        xaxis_dtick='M1',
        xaxis_tickangle=-45,
        barmode='stack',
    )
    
    # Income Summary (Yearly)
    yearly_income = df_income.groupby(['Year', 'account'])['income'].sum().unstack().fillna(0)
    yearly_income = yearly_income.loc[:, (yearly_income > 0).any(axis=0)]
    
    yearly_income_summary_fig = go.Figure()
    for col in yearly_income.columns:
        yearly_income_summary_fig.add_trace(go.Bar(
            x=yearly_income.index,
            y=yearly_income[col],
            name=col
        ))

    yearly_income_summary_fig.update_layout(
        title='Yearly Income Summary',
        xaxis_title='Year',
        yaxis_title='Total Income',
        barmode='stack',
        xaxis_dtick='M1'
    )

    return pie_fig, monthly_income_summary_fig, yearly_income_summary_fig

@app.callback(
    Output('expense-pie-chart', 'figure'),
    Output('expense-summary-month', 'figure'),
    Output('expense-summary-year', 'figure'),
    Input('date-picker-global', 'start_date'),
    Input('date-picker-global', 'end_date')
)
def update_expense_plots(start_date, end_date):
    df_filtered = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    df_expenses = df_filtered[df_filtered['expenses'] != 0]
    
    df_expenses['account'] = df_expenses['account'].apply(lambda x: x.split(':')[-1])
    df_expenses['Year'] = df_expenses['date'].dt.year
    df_expenses['Month'] = df_expenses['date'].dt.strftime('%Y-%m')

    # Expense Pie Chart
    expense_summary = df_expenses.groupby('account')['expenses'].sum().reset_index()
    pie_fig = go.Figure(data=[go.Pie(labels=expense_summary['account'], values=expense_summary['expenses'])])
    pie_fig.update_layout(title='Expense Distribution')

    # Expense Summary (Monthly)
    monthly_expenses = df_expenses.groupby(['Month', 'account'])['expenses'].sum().unstack().fillna(0)
    
    monthly_expense_summary_fig = go.Figure()
    for col in monthly_expenses.columns:
        monthly_expense_summary_fig.add_trace(go.Bar(
            x=monthly_expenses.index,
            y=monthly_expenses[col],
            name=col
        ))

    monthly_expense_summary_fig.update_layout(
        title='Monthly Expense Summary',
        xaxis_title='Month',
        yaxis_title='Total Expenses',
        xaxis_tickformat='%Y-%m',
        xaxis_dtick='M1',
        xaxis_tickangle=-45,
        barmode='stack',
    )
    
    # Expense Summary (Yearly)
    yearly_expenses = df_expenses.groupby(['Year', 'account'])['expenses'].sum().unstack().fillna(0)
    yearly_expenses = yearly_expenses.loc[:, (yearly_expenses > 0).any(axis=0)]

    yearly_expense_summary_fig = go.Figure()
    for col in yearly_expenses.columns:
        yearly_expense_summary_fig.add_trace(go.Bar(
            x=yearly_expenses.index,
            y=yearly_expenses[col],
            name=col
        ))

    yearly_expense_summary_fig.update_layout(
        title='Yearly Expense Summary',
        xaxis_title='Year',
        yaxis_title='Total Expenses',
        barmode='stack',
        xaxis_dtick='M1'
    )

    return pie_fig, monthly_expense_summary_fig, yearly_expense_summary_fig

# Layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.NavItem(dbc.NavLink("Income", href="/income")),
            dbc.NavItem(dbc.NavLink("Expenses", href="/expenses")),
            dbc.NavItem(dbc.NavLink(
                dcc.DatePickerRange(
                    id='date-picker-global',
                    start_date=df['date'].min().date(),
                    end_date=df['date'].max().date(),
                    display_format='YYYY-MM-DD'
                ),
                href="",
                style={'padding': '5px'}
            )),
        ],
        brand="Financial Dashboard",
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-4",
        expand="md"
    ),
    html.Div(id='page-content')
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

