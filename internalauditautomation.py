import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from thefuzz import fuzz
from scipy.stats import chisquare
from sklearn.ensemble import IsolationForest
import plotly.express as px
import os
import sys

# Jupyter/Colab-specific imports for download
try:
    from IPython.display import Javascript
    from google.colab import files
    IN_NOTEBOOK = True
except ImportError:
    IN_NOTEBOOK = False

# Set up logging
logging.basicConfig(
    filename='audit_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define schemas for data validation
LEDGER_SCHEMA = DataFrameSchema({
    "Date": Column("datetime64[ns]", Check(lambda x: x <= pd.Timestamp.now())),
    "Amount": Column(float, Check(lambda x: x > 0)),
    "Description": Column(str),
    "Transaction_ID": Column(str, regex=r"^TX\d{10}$", nullable=True)
})

BANK_SCHEMA = DataFrameSchema({
    "Date": Column("datetime64[ns]", Check(lambda x: x <= pd.Timestamp.now())),
    "Amount": Column(float),  # Allow negatives for bank fees
    "Description": Column(str),
    "Transaction_ID": Column(str, regex=r"^TX\d{10}$", nullable=True)
})

class InternalAuditAutomation:
    def __init__(self):
        self.discrepancies = []

    def load_data(self, ledger_file, bank_file):
        """Load and validate ledger and bank statement data."""
        try:
            ledger = pd.read_csv(ledger_file, parse_dates=['Date']).pipe(LEDGER_SCHEMA.validate)
            bank = pd.read_csv(bank_file, parse_dates=['Date']).pipe(BANK_SCHEMA.validate)
            logging.info("Data loaded and validated successfully.")
            return ledger, bank
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def reconcile_accounts(self, ledger, bank):
        """Reconcile accounts with exact matching on Transaction_ID."""
        discrepancies = []
        
        if 'Transaction_ID' in ledger.columns and 'Transaction_ID' in bank.columns:
            # Perform outer merge to find matches and mismatches
            matched = pd.merge(
                ledger, bank,
                on=['Transaction_ID'],
                how='outer',
                indicator=True,
                suffixes=('_ledger', '_bank')
            )
            
            # Log matched transactions
            matched_count = len(matched[matched['_merge'] == 'both'])
            logging.info(f"Matched {matched_count} transactions by Transaction_ID.")
            
            # Process unmatched transactions
            unmatched = matched[matched['_merge'] != 'both']
            
            for _, row in unmatched.iterrows():
                if row['_merge'] == 'left_only':
                    discrepancies.append({
                        'Type': 'Ledger Only',
                        'Transaction_ID': row['Transaction_ID'],
                        'Date': row['Date_ledger'],
                        'Amount': row['Amount_ledger'],
                        'Description': row['Description_ledger'],
                        'Severity': 'High',
                        'Notes': 'Transaction in ledger but not in bank statement'
                    })
                elif row['_merge'] == 'right_only':
                    discrepancies.append({
                        'Type': 'Bank Only',
                        'Transaction_ID': row['Transaction_ID'],
                        'Date': row['Date_bank'],
                        'Amount': row['Amount_bank'],
                        'Description': row['Description_bank'],
                        'Severity': 'High',
                        'Notes': 'Transaction in bank but not in ledger'
                    })
        
        # Fuzzy matching only for transactions without Transaction_ID
        unmatched_ledger = ledger[~ledger['Transaction_ID'].isin(matched['Transaction_ID']) & ledger['Transaction_ID'].notna()]
        unmatched_bank = bank[~bank['Transaction_ID'].isin(matched['Transaction_ID']) & bank['Transaction_ID'].notna()]
        
        for _, l_row in unmatched_ledger.iterrows():
            best_score = 0
            best_match = None
            for _, b_row in unmatched_bank.iterrows():
                score = fuzz.token_sort_ratio(l_row['Description'], b_row['Description'])
                if score > 80 and abs((l_row['Date'] - b_row['Date']).days) <= 2:
                    if score > best_score:
                        best_score = score
                        best_match = b_row
            if best_match is None:
                discrepancies.append({
                    'Type': 'Ledger Unmatched',
                    'Transaction_ID': l_row['Transaction_ID'],
                    'Date': l_row['Date'],
                    'Amount': l_row['Amount'],
                    'Description': l_row['Description'],
                    'Severity': 'Medium',
                    'Notes': 'No matching bank transaction found within date window'
                })
        
        self.discrepancies.extend(discrepancies)
        logging.info(f"Found {len(discrepancies)} reconciliation discrepancies.")
        return discrepancies

    def cross_check_transactions(self, ledger, bank):
        """Check for duplicate transactions."""
        issues = []
        
        for data, source in [(ledger, 'Ledger'), (bank, 'Bank')]:
            duplicates = data[data.duplicated(subset=['Date', 'Amount', 'Description'], keep=False)]
            seen = set()
            for _, row in duplicates.iterrows():
                key = (row['Date'], row['Amount'], row['Description'])
                if key not in seen:
                    seen.add(key)
                    issues.append({
                        'Type': f'Duplicate in {source}',
                        'Transaction_ID': row.get('Transaction_ID', 'N/A'),
                        'Date': row['Date'],
                        'Amount': row['Amount'],
                        'Description': row['Description'],
                        'Severity': 'Medium',
                        'Notes': f'Potential duplicate transaction in {source.lower()}'
                    })
        
        self.discrepancies.extend(issues)
        logging.info(f"Found {len(issues)} duplicate issues.")
        return issues

    def validate_amounts(self, ledger, bank):
        """Validate amount ranges (ledger-specific due to schema)."""
        issues = []
        
        for _, row in bank.iterrows():
            if row['Amount'] == 0:
                issues.append({
                    'Type': 'Zero Bank Amount',
                    'Transaction_ID': row.get('Transaction_ID', 'N/A'),
                    'Date': row['Date'],
                    'Amount': row['Amount'],
                    'Description': row['Description'],
                    'Severity': 'Low',
                    'Notes': 'Bank transaction with zero amount'
                })
        
        self.discrepancies.extend(issues)
        logging.info(f"Found {len(issues)} amount validation issues.")
        return issues

    def detect_anomalous_patterns(self, ledger):
        """Apply Benford's Law to detect anomalies."""
        issues = []
        first_digits = ledger['Amount'].apply(lambda x: str(abs(int(x)))[0] if str(abs(int(x)))[0].isdigit() else None)
        first_digits = first_digits.dropna()
        if len(first_digits) < 30:
            logging.info("Insufficient data for Benford's Law analysis.")
            return issues
        
        observed = first_digits.value_counts(normalize=True).reindex([str(i) for i in range(1, 10)], fill_value=0)
        benford = [np.log10(1 + 1/d) for d in range(1, 10)]
        stat, p = chisquare(observed, benford)
        
        if p < 0.05:
            issues.append({
                'Type': 'Benford Anomaly',
                'Transaction_ID': 'N/A',
                'Date': None,
                'Amount': None,
                'Description': f"First-digit distribution deviates",
                'Severity': 'High',
                'Notes': f'Possible data manipulation or error in ledger amounts (p={p:.4f})'
            })
        
        self.discrepancies.extend(issues)
        logging.info(f"Benford's Law found {len(issues)} issues.")
        return issues

    def calculate_risk_scores(self, ledger):
        """Predict high-risk transactions using Isolation Forest."""
        features = ledger[['Amount']].copy()
        features['Date_Hour'] = ledger['Date'].dt.hour
        model = IsolationForest(contamination=0.05, random_state=42)
        scores = model.fit_predict(features)
        
        high_risk = ledger[scores == -1]
        issues = []
        for _, row in high_risk.iterrows():
            issues.append({
                'Type': 'High Risk Transaction',
                'Transaction_ID': row.get('Transaction_ID', 'N/A'),
                'Date': row['Date'],
                'Amount': row['Amount'],
                'Description': row['Description'],
                'Severity': 'High',
                'Notes': 'Transaction flagged as anomalous by Isolation Forest'
            })
        
        self.discrepancies.extend(issues)
        logging.info(f"Found {len(issues)} high-risk transactions.")
        return issues

    def generate_report(self, output_file='audit_report.csv'):
        """Generate detailed CSV report, summary text file, and interactive visualization."""
        try:
            report_df = pd.DataFrame(self.discrepancies)
            
            if not report_df.empty:
                # Remove duplicates
                report_df = report_df.drop_duplicates(
                    subset=['Type', 'Transaction_ID', 'Date', 'Amount', 'Description'],
                    keep='first'
                )
                
                # Ensure consistent columns
                report_df = report_df.reindex(
                    columns=['Type', 'Transaction_ID', 'Date', 'Amount', 'Description', 'Severity', 'Notes'],
                    fill_value=None
                )
                
                # Save detailed report to CSV
                report_df.to_csv(output_file, index=False)
                abs_path = os.path.abspath(output_file)
                logging.info(f"Saved audit report to {abs_path}")
                print(f"Audit report saved to: {abs_path}")
                
                # Trigger automatic download in Jupyter/Colab
                if IN_NOTEBOOK:
                    try:
                        if 'google.colab' in sys.modules:
                            files.download(output_file)
                            logging.info("Triggered download for audit_report.csv in Colab.")
                        else:
                            # Jupyter download using JavaScript
                            js_code = f"""
                            function downloadFile() {{
                                var link = document.createElement('a');
                                link.href = '{output_file}';
                                link.download = '{output_file}';
                                document.body.appendChild(link);
                                link.click();
                                document.body.removeChild(link);
                            }}
                            downloadFile();
                            """
                            display(Javascript(js_code))
                            logging.info("Triggered download for audit_report.csv in Jupyter.")
                    except Exception as e:
                        logging.warning(f"Download trigger failed: {str(e)}")
                        print(f"Download trigger failed, but CSV saved to: {abs_path}")
                
                # Generate summary statistics
                summary = report_df.groupby(['Type', 'Severity']).agg(
                    Count=('Type', 'size'),
                    Total_Amount=('Amount', lambda x: x.abs().sum() if x.notna().any() else 0)
                ).reset_index()
                summary['Total_Amount'] = summary['Total_Amount'].round(2)
                
                # Write summary to text file
                with open('audit_summary.txt', 'w') as f:
                    f.write("Audit Summary Report\n")
                    f.write("=" * 20 + "\n\n")
                    f.write(f"Total Discrepancies: {len(report_df)}\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write("Breakdown by Type and Severity:\n")
                    f.write(summary.to_string(index=False) + "\n\n")
                    f.write(f"See {output_file} for detailed findings and audit_dashboard.html for visualization.\n")
                
                # Generate visualization (treemap)
                fig = px.treemap(
                    report_df,
                    path=['Type', 'Severity', 'Description'],
                    values=report_df.get('Amount', pd.Series([1]*len(report_df))).abs(),
                    color='Severity',
                    color_discrete_map={'Low': '#00CC96', 'Medium': '#FFBB28', 'High': '#FF5733'},
                    title='Audit Findings Breakdown by Type and Severity'
                )
                fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
                fig.write_html('audit_dashboard.html')
                
                logging.info(f"Report, summary, and dashboard generated: {output_file}, audit_summary.txt, audit_dashboard.html")
            else:
                logging.info("No discrepancies found.")
                with open('audit_summary.txt', 'w') as f:
                    f.write("Audit Summary Report\n")
                    f.write("=" * 20 + "\n\n")
                    f.write("No discrepancies found.\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                # Create empty CSV
                pd.DataFrame(columns=['Type', 'Transaction_ID', 'Date', 'Amount', 'Description', 'Severity', 'Notes']).to_csv(output_file, index=False)
                abs_path = os.path.abspath(output_file)
                logging.info(f"Saved empty audit report to {abs_path}")
                print(f"Empty audit report saved to: {abs_path}")
                
                if IN_NOTEBOOK:
                    try:
                        if 'google.colab' in sys.modules:
                            files.download(output_file)
                            logging.info("Triggered download for empty audit_report.csv in Colab.")
                        else:
                            js_code = f"""
                            function downloadFile() {{
                                var link = document.createElement('a');
                                link.href = '{output_file}';
                                link.download = '{output_file}';
                                document.body.appendChild(link);
                                link.click();
                                document.body.removeChild(link);
                            }}
                            downloadFile();
                            """
                            display(Javascript(js_code))
                            logging.info("Triggered download for empty audit_report.csv in Jupyter.")
                    except Exception as e:
                        logging.warning(f"Download trigger failed: {str(e)}")
                        print(f"Download trigger failed, but empty CSV saved to: {abs_path}")
                
            return report_df
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            print(f"Error generating report: {str(e)}")
            raise

    def run_audit(self, ledger_file, bank_file):
        """Run the full audit process."""
        start_time = datetime.now()
        logging.info("Starting audit process.")
        
        try:
            ledger, bank = self.load_data(ledger_file, bank_file)
            self.reconcile_accounts(ledger, bank)
            self.cross_check_transactions(ledger, bank)
            self.validate_amounts(ledger, bank)
            self.detect_anomalous_patterns(ledger)
            self.calculate_risk_scores(ledger)
            report = self.generate_report()
            
            duration = (datetime.now() - start_time).total_seconds()
            logging.info(f"Audit completed in {duration} seconds.")
            return report, duration
        except Exception as e:
            logging.error(f"Audit failed: {str(e)}")
            print(f"Audit failed: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    # Sample data
    ledger_data = pd.DataFrame({
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-03']),
        'Amount': [100.00, 200.00, 300.00, 300.00],  # Duplicate
        'Description': ['Sale', 'Refund', 'Sale', 'Sale'],
        'Transaction_ID': ['TX1234567890', 'TX1234567891', 'TX1234567892', 'TX1234567892']
    })
    bank_data = pd.DataFrame({
        'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-04']),
        'Amount': [100.00, 200.00, -50.00],
        'Description': ['Deposit', 'Withdrawal', 'Bank Fee'],
        'Transaction_ID': ['TX1234567890', 'TX1234567891', 'TX1234567893']
    })
    
    try:
        ledger_data.to_csv('ledger.csv', index=False)
        bank_data.to_csv('bank.csv', index=False)
        logging.info("Sample data CSVs created.")
    except Exception as e:
        logging.error(f"Error creating sample CSVs: {str(e)}")
        print(f"Error creating sample CSVs: {str(e)}")
        raise
    
    # Run audit
    auditor = InternalAuditAutomation()
    report, duration = auditor.run_audit('ledger.csv', 'bank.csv')
    
    print("Audit Report:")
    print(report)
    print(f"\nAudit completed in {duration} seconds.")

"""
README
======

Purpose
-------
This script automates the internal audit process for reconciling financial ledger and bank statement data. It identifies discrepancies, detects anomalies, and flags high-risk transactions, producing detailed reports and visualizations. The script ensures robust data validation, accurate matching, and user-friendly outputs, including automatic downloading of the audit report CSV in browser-based environments (e.g., Jupyter, Colab).

Key functionalities:
- Validates input data using strict schemas.
- Reconciles transactions via exact Transaction_ID matching.
- Detects duplicates, zero amounts, statistical anomalies, and risky transactions.
- Generates a CSV report, text summary, and interactive treemap visualization.
- Automatically downloads the CSV in Jupyter/Colab.

Requirements
------------
- **Python Version**: 3.8 or higher
- **Required Libraries**:
  - pandas: Data manipulation and CSV handling
  - numpy: Numerical computations
  - pandera: Schema-based data validation
  - thefuzz: Fuzzy string matching (optional)
  - scipy: Statistical tests (Benford's Law)
  - scikit-learn: Machine learning (Isolation Forest)
  - plotly: Interactive visualizations
  - IPython, google-colab (optional, for Jupyter/Colab downloads)
- **Installation**:

  pip install pandas numpy pandera thefuzz scipy scikit-learn plotly
"""