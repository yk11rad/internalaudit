# internalaudit
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
