import openai
from openai import OpenAI 
import streamlit as st
import requests
import os
import pandas as pd
from apikey import OPENAI_API_KEY, FMP_API_KEY
openai.api_key = OPENAI_API_KEY

def get_jsonparsed_data(url):
    """
    Fetch JSON data from a URL and parse it
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return []

def get_financial_statements(ticker, limit, period, statement_type):
    if statement_type == "Income Statement":
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    elif statement_type == "Balance Sheet":
        url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    elif statement_type == "Cash Flow":
        url = f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{ticker}?period={period}&limit={limit}&apikey={FMP_API_KEY}"
    
    data = get_jsonparsed_data(url)

    if isinstance(data, list) and data:
        return pd.DataFrame(data)
    else:
        st.error("Unable to fetch financial statements. Please ensure the ticker is correct and try again.")
        return pd.DataFrame()
    
def generate_financial_summary(financial_statements, statement_type):
    """
    Generate a summary of financial statements using OpenAI's API.
    """
    # Create a summary of key financial metrics for all periods
    summaries = []
    
    for i in range(len(financial_statements)):
        if statement_type == "Income Statement":
            summary = f"""
            For the period ending {financial_statements['date'].iloc[i]}, the company reported the following:
            - Revenue: ${financial_statements['revenue'].iloc[i]:,.2f}
            - Gross Profit: ${financial_statements['grossProfit'].iloc[i]:,.2f}
            - Operating Income: ${financial_statements['operatingIncome'].iloc[i]:,.2f}
            - Net Income: ${financial_statements['netIncome'].iloc[i]:,.2f}
            - EPS: ${financial_statements['eps'].iloc[i]:.2f}
            """
        elif statement_type == "Balance Sheet":
            summary = f"""
            For the period ending {financial_statements['date'].iloc[i]}, the company reported the following:
            - Total Assets: ${financial_statements['totalAssets'].iloc[i]:,.2f}
            - Total Liabilities: ${financial_statements['totalLiabilities'].iloc[i]:,.2f}
            - Total Equity: ${financial_statements['totalEquity'].iloc[i]:,.2f}
            - Cash and Equivalents: ${financial_statements['cashAndCashEquivalents'].iloc[i]:,.2f}
            - Total Debt: ${financial_statements['totalDebt'].iloc[i]:,.2f}
            """
        elif statement_type == "Cash Flow":
            summary = f"""
            For the period ending {financial_statements['date'].iloc[i]}, the company reported the following:
            - Operating Cash Flow: ${financial_statements['netCashProvidedByOperatingActivities'].iloc[i]:,.2f}
            - Capital Expenditure: ${financial_statements.get('capitalExpenditure', pd.Series([0]*len(financial_statements))).iloc[i]:,.2f}
            - Free Cash Flow: ${financial_statements.get('freeCashFlow', pd.Series([0]*len(financial_statements))).iloc[i]:,.2f}
            - Dividends Paid: ${abs(financial_statements.get('dividendsPaid', pd.Series([0]*len(financial_statements))).iloc[i]):,.2f}
            """
        summaries.append(summary)
    
    all_summaries = "\n\n".join(summaries)
    
    # Updated OpenAI API call for v1.0.0+
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an AI trained to provide financial analysis based on financial statements."
            },
            {
                "role": "user",
                "content": f"""
                Please analyze the following financial data and provide insights:\n{all_summaries}\n
                For your analysis:
                1. Identify significant trends across the time periods
                2. Highlight notable changes in key metrics
                3. Assess overall financial health based on these statements
                4. Provide potential explanations for major changes
                5. Summarize with 2-3 key takeaways for investors
                """
            }
        ]
    )
    
    # Updated response handling
    return response.choices[0].message.content

st.title('Financial Statements Analyst')

statement_type = st.selectbox("Select financial statement type:", ["Income Statement", "Balance Sheet", "Cash Flow"])

col1, col2 = st.columns(2)

with col1:
    period = st.selectbox("Select period:", ["Annual", "Quarterly"]).lower()

with col2:
    limit = st.number_input("Number of past financial statements to analyze:", min_value=1, max_value=10, value=4)
    

ticker = st.text_input("Please enter the company ticker:")

if st.button('Run'):
    if ticker:
        ticker = ticker.upper()
        financial_statements = get_financial_statements(ticker, limit, period, statement_type)

        with st.expander("View Financial Statements"):
            st.dataframe(financial_statements)

        financial_summary = generate_financial_summary(financial_statements, statement_type)

        st.write(f'Summary for {ticker}:\n {financial_summary}\n')  