import csv
from datetime import datetime
from collections import defaultdict

def parse_number(number_str):
    # Replace thousand separators
    number_str = number_str.replace(',', '')
    return float(number_str)

def extract_accounts(csv_file):
    accounts = set()
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            accounts.add(row['Full Account Name'])
    return sorted(accounts)

def parse_gnucash_csv_to_ledger(csv_file, ledger_file):
    transactions = defaultdict(list)
    accounts = extract_accounts(csv_file)

    # Read the CSV and group by Transaction ID
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            transactions[row['Transaction ID']].append(row)

    # Collect all transactions for sorting
    sorted_transactions = []
    for transaction_id, entries in transactions.items():
        date = datetime.strptime(entries[0]['Date'], '%m/%d/%Y')
        description = entries[0]['Description']
        sorted_transactions.append((date, description, entries))

    # Sort transactions by date
    sorted_transactions.sort(key=lambda x: x[0])

    # Write to Ledger file
    with open(ledger_file, 'w', encoding='utf-8') as ledger:
        # Write account definitions
        ledger.write("; Account Definitions\n")
        for account in accounts:
            ledger.write(f"account {account}\n")
        ledger.write("\n")

        # Write transactions
        for date, description, entries in sorted_transactions:
            date_str = date.strftime('%Y/%m/%d')
            ledger.write(f"{date_str} * {description}\n")

            for entry in entries:
                account = entry['Full Account Name']
                amount = parse_number(entry['Amount Num.'])
                currency = entry['Commodity/Currency'].replace('CURRENCY::', '')
                amount_with_currency = f"{amount:.2f} {currency}"

                ledger.write(f"    {account}    {amount_with_currency}\n")

            ledger.write("\n")

# Usage example
csv_file = 'gnucash_data.csv'
ledger_file = 'ledger_data.dat'
parse_gnucash_csv_to_ledger(csv_file, ledger_file)
