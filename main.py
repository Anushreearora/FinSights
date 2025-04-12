import os
import sys

import streamlit as st

# Data handling imports
import pandas as pd
import plotly.express as px 
import random
from pygwalker.api.streamlit import StreamlitRenderer

# Configuration imports
import tickers as cfg

#get data path
def get_data_path() -> str:
    return os.path.join(os.getcwd(), 'datasets')

#load csv 
def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(get_data_path(), filename))

#get data from csv
def get_data():
    """Load all financial data from CSV files."""
    annual_files = [
        'annually_income_statement.csv',
        'annually_balance_sheet.csv',
        'annually_cash_flow.csv'
    ]
    quarterly_files = [
        'quarterly_income_statement.csv',
        'quarterly_balance_sheet.csv',
        'quarterly_cash_flow.csv'
    ]
    return [load_data(file) for file in annual_files + quarterly_files]

@st.cache_data
def get_cached_data() -> list:
    """Get cached financial data."""
    return get_data()

#plot chart
def plot_charts(df: pd.DataFrame, metrics: list, period: str = "annually",  show_percentage: bool = False, tickers: list = None) -> None:
    filtered_df = df.copy()
    if tickers:
        filtered_df = filtered_df[filtered_df['Symbol'].isin(tickers)]
    
    filtered_df['Year'] = pd.to_datetime(filtered_df['Date']).dt.year
    if period == 'quarterly':
        filtered_df['Quarter'] = pd.to_datetime(filtered_df['Date']).dt.to_period('Q').astype(str)
        filtered_df['Quarter'] = filtered_df['Year'].astype(str) + ' ' + filtered_df['Quarter'].str[-2:]

    # Remove rows with NaN values in the metric columns
    filtered_df = filtered_df.dropna(subset=metrics)

    # Sort the dataframe by Year and Quarter
    sort_columns = ['Year'] if period == 'annually' else ['Year', 'Quarter']
    filtered_df = filtered_df.sort_values(sort_columns)

    # Round the metrics to 2 decimal places
    filtered_df[metrics] = filtered_df[metrics].round(2)

    if show_percentage:
        filtered_df[metrics] *= 100

    # Ensure each year has all four quarters
    if period == 'quarterly':
        all_quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        filtered_df = filtered_df[filtered_df['Quarter'].str[-2:].isin(all_quarters)]

    fig = px.line(filtered_df, x='Year' if period == 'annually' else 'Quarter', y=metrics, color='Symbol',
                  title=f'{", ".join(metrics)} Trends ({period.capitalize()})',
                  color_discrete_map= cfg.COLOR_THEME)
    
    # Set the x-axis properties
    if period == 'annually':
        fig.update_xaxes(title_text='Year', tickmode='linear', dtick=1, range=[filtered_df['Year'].min() - 0.5, filtered_df['Year'].max() + 0.5])
    else:
        fig.update_xaxes(title_text='Quarter', tickmode='linear', dtick=1, categoryorder='category ascending')
    
    fig.update_yaxes(title_text=', '.join(metrics))

    # Set y-axis range
    y_min, y_max = filtered_df[metrics].min().min(), filtered_df[metrics].max().max()
    fig.update_yaxes(range=[y_min * 1.1, y_max * 1.1] if y_min < 0 else [0, y_max * 1.1])
    
    # Add hover data
    hover_template = '%{y:.2f}%' if show_percentage else '%{y:.2f}'
    fig.update_traces(hovertemplate=hover_template)

    st.plotly_chart(fig, use_container_width=True)

#income statement tab
def display_income_statement_tab(annual_income_statement_df: pd.DataFrame, quarterly_income_statement_df: pd.DataFrame) -> None:
    """Display the income statement analysis tab."""

    st.title("💰 Annual Income Statement Analysis")
    # create_custom_chart(annual_income_statement_df)
    
    selected_tickers = st.multiselect('Select tickers to analyze', sorted(annual_income_statement_df['Symbol'].unique()), default=['ALC'])

    col1, col2 = st.columns(2)
    with col1:
        for metric in ['Total Revenue', 'Normalized EBITDA', 'Normalized Income']:
            plot_charts(annual_income_statement_df, [metric], tickers=selected_tickers)
    with col2:
        for metric in ['Net Income', 'Basic EPS', 'Operating Expense']:
            plot_charts(annual_income_statement_df, [metric], tickers=selected_tickers)

    st.markdown("### Quarterly Income Statement Analysis")
    col1, col2 = st.columns(2)
    with col1:
        for metric in ['Total Revenue', 'Normalized EBITDA', 'Normalized Income']:
            plot_charts(quarterly_income_statement_df, [metric], period='quarterly', tickers=selected_tickers)
    with col2:
        for metric in ['Net Income', 'Basic EPS', 'Operating Expense']:
            plot_charts(quarterly_income_statement_df, [metric], period='quarterly', tickers=selected_tickers)

#balance sheet tab

#cash flows tab

def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="FinSight", page_icon="💬")

    # Get the data
    (
        annual_income_statement_df,
        annual_balance_sheet_df,
        annual_cash_flow_df,
        quarterly_income_statement_df,
        quarterly_balance_sheet_df,
        quarterly_cash_flow_df
    ) = get_data()
    
    topic = st.sidebar.radio("Select an option", ["Income Statement", "Balance Sheet", "Cash Flow"])

    if topic == "Income Statement":
        display_income_statement_tab(annual_income_statement_df, quarterly_income_statement_df)
    # elif topic == "Balance Sheet":
    #     display_balance_sheet_tab(annual_balance_sheet_df, quarterly_balance_sheet_df)
    # elif topic == "Cash Flow":
    #     display_cash_flow_tab(annual_cash_flow_df, quarterly_cash_flow_df)

if __name__ == "__main__":
    main()