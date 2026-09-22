import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import Dict, List, Tuple
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os


def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute high-level KPIs from the sales data.
    
    Args:
        df: DataFrame with sales data
        
    Returns:
        Dictionary with computed KPIs
    """
    kpis = {}
    
    # Total Sales
    if 'unit_sales' in df.columns:
        kpis['total_sales'] = df['unit_sales'].sum()
    
    # Total Transactions
    if 'transactions' in df.columns:
        kpis['total_transactions'] = df['transactions'].sum()
    
    # Average Daily Sales
    if 'unit_sales' in df.columns and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        daily_sales = df.groupby('date')['unit_sales'].sum()
        kpis['average_daily_sales'] = daily_sales.mean()
    
    # Average Daily Transactions
    if 'transactions' in df.columns and 'date' in df.columns:
        daily_transactions = df.groupby('date')['transactions'].sum()
        kpis['average_daily_transactions'] = daily_transactions.mean()
    
    # Top Product Families
    if 'family' in df.columns and 'unit_sales' in df.columns:
        family_sales = df.groupby('family')['unit_sales'].sum().sort_values(ascending=False)
        kpis['top_families'] = family_sales.head(5).to_dict()
    
    # Top Stores
    if 'store' in df.columns and 'unit_sales' in df.columns:
        store_sales = df.groupby('store')['unit_sales'].sum().sort_values(ascending=False)
        kpis['top_stores'] = store_sales.head(5).to_dict()
    
    # Date Range
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        kpis['date_range'] = (df['date'].min().strftime('%Y-%m-%d'), 
                              df['date'].max().strftime('%Y-%m-%d'))
        kpis['total_days'] = (df['date'].max() - df['date'].min()).days + 1
    
    return kpis


def create_sales_trend_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a line chart showing sales trends over time.
    
    Args:
        df: DataFrame with sales data
        
    Returns:
        Plotly figure object
    """
    if 'date' not in df.columns or 'unit_sales' not in df.columns:
        return None
    
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    daily_sales = df_copy.groupby('date')['unit_sales'].sum().reset_index()
    
    fig = px.line(daily_sales, x='date', y='unit_sales',
                  title='Sales Trend Over Time',
                  labels={'unit_sales': 'Unit Sales', 'date': 'Date'})
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Unit Sales',
        hovermode='x unified'
    )
    
    return fig


def create_family_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing sales distribution by product family.
    
    Args:
        df: DataFrame with sales data
        
    Returns:
        Plotly figure object
    """
    if 'family' not in df.columns or 'unit_sales' not in df.columns:
        return None
    
    family_sales = df.groupby('family')['unit_sales'].sum().sort_values(ascending=False)
    
    fig = px.bar(x=family_sales.index, y=family_sales.values,
                 title='Sales Distribution by Product Family',
                 labels={'x': 'Product Family', 'y': 'Unit Sales'})
    
    fig.update_layout(
        xaxis_title='Product Family',
        yaxis_title='Unit Sales',
        xaxis_tickangle=-45
    )
    
    return fig


def create_store_performance_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing sales by store.
    
    Args:
        df: DataFrame with sales data
        
    Returns:
        Plotly figure object
    """
    if 'store' not in df.columns or 'unit_sales' not in df.columns:
        return None
    
    store_sales = df.groupby('store')['unit_sales'].sum().sort_values(ascending=False)
    
    fig = px.bar(x=store_sales.index, y=store_sales.values,
                 title='Sales by Store',
                 labels={'x': 'Store', 'y': 'Unit Sales'})
    
    fig.update_layout(
        xaxis_title='Store',
        yaxis_title='Unit Sales',
        xaxis_tickangle=-45
    )
    
    return fig


def generate_executive_summary(kpis: dict, api_key: str) -> str:
    """
    Generate an executive summary using Gemini AI.
    
    Args:
        kpis: Dictionary of computed KPIs
        api_key: Google API key for Gemini
        
    Returns:
        Generated executive summary text
    """
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        
        # Format KPIs for the prompt
        kpi_text = f"""
        Total Sales: {kpis.get('total_sales', 'N/A'):,.0f}
        Total Transactions: {kpis.get('total_transactions', 'N/A'):,.0f}
        Average Daily Sales: {kpis.get('average_daily_sales', 'N/A'):,.0f}
        Average Daily Transactions: {kpis.get('average_daily_transactions', 'N/A'):,.0f}
        Date Range: {kpis.get('date_range', ('N/A', 'N/A'))[0]} to {kpis.get('date_range', ('N/A', 'N/A'))[1]}
        Total Days: {kpis.get('total_days', 'N/A')}
        
        Top Product Families:
        """
        
        for family, sales in kpis.get('top_families', {}).items():
            kpi_text += f"\n  - {family}: {sales:,.0f}"
        
        kpi_text += f"\n\nTop Stores:"
        for store, sales in kpis.get('top_stores', {}).items():
            kpi_text += f"\n  - {store}: {sales:,.0f}"
        
        prompt = f"""
        You are a retail analytics expert. Based on the following KPIs from a retail sales dataset, 
        write a concise executive summary (3-4 paragraphs) that highlights:
        1. Overall performance trends
        2. Key insights about top-performing product families
        3. Store performance observations
        4. Any notable patterns or recommendations
        
        KPIs:
        {kpi_text}
        
        Keep the summary professional, data-driven, and actionable.
        """
        
        message = HumanMessage(content=prompt)
        response = llm.invoke([message])
        
        # Extract text content from response
        if hasattr(response, 'content'):
            content = response.content
            # If content is a dict with 'text' field, extract it
            if isinstance(content, dict) and 'text' in content:
                extracted_text = content['text']
            # If content is a string, return it directly
            elif isinstance(content, str):
                extracted_text = content
            else:
                extracted_text = str(content)
            
            # Remove the 'extras' and 'signature' part using regex
            extracted_text = re.sub(r", 'extras': \{[^}]+\}", '', extracted_text)
            extracted_text = re.sub(r"'extras': \{[^}]+\}", '', extracted_text)
            return extracted_text
        else:
            return str(response)
        
    except Exception as e:
        import traceback
        error_details = f"Error generating executive summary: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(f"DEBUG: {error_details}")  # Print to console
        return error_details


def format_kpis_display(kpis: dict) -> str:
    """
    Format KPIs for display in Streamlit.
    
    Args:
        kpis: Dictionary of computed KPIs
        
    Returns:
        Formatted string for display
    """
    display_text = ""
    
    if 'total_sales' in kpis:
        display_text += f"**Total Sales:** {kpis['total_sales']:,.0f}\n\n"
    
    if 'total_transactions' in kpis:
        display_text += f"**Total Transactions:** {kpis['total_transactions']:,.0f}\n\n"
    
    if 'average_daily_sales' in kpis:
        display_text += f"**Average Daily Sales:** {kpis['average_daily_sales']:,.0f}\n\n"
    
    if 'average_daily_transactions' in kpis:
        display_text += f"**Average Daily Transactions:** {kpis['average_daily_transactions']:,.0f}\n\n"
    
    if 'date_range' in kpis:
        display_text += f"**Date Range:** {kpis['date_range'][0]} to {kpis['date_range'][1]}\n\n"
    
    if 'total_days' in kpis:
        display_text += f"**Total Days:** {kpis['total_days']}\n\n"
    
    return display_text
