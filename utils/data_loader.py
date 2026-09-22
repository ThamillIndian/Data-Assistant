import pandas as pd
import streamlit as st
from typing import Optional, Tuple


@st.cache_data
def load_csv_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load and cache CSV data from uploaded file.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        DataFrame with loaded data or None if loading fails
    """
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Auto-map common column names to expected schema
        column_mapping = {
            'Date': 'date',
            'Qty': 'unit_sales',
            'Amount': 'transactions',
            'Category': 'family',
            'Sales Channel': 'store'
        }
        
        # Apply mapping for columns that exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
            else:
                st.warning(f"Column '{old_name}' not found in CSV. Skipping mapping to '{new_name}'.")
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None


def validate_sales_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the DataFrame contains required columns for sales data.
    
    Expected columns: date, store, family, unit_sales, transactions
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_columns = ['date', 'store', 'family', 'unit_sales', 'transactions']
    
    # Show actual columns found in the file
    actual_columns = list(df.columns)
    st.info(f"**Columns found in your CSV:** {', '.join(actual_columns)}")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check if date column can be converted to datetime
    try:
        pd.to_datetime(df['date'])
    except Exception as e:
        return False, f"Invalid date format: {str(e)}"
    
    # Check if numeric columns are actually numeric
    numeric_columns = ['unit_sales', 'transactions']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return False, f"Column '{col}' must be numeric"
    
    return True, ""


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': list(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'date_range': None,
        'unique_stores': None,
        'unique_families': None
    }
    
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
        info['date_range'] = (dates.min().strftime('%Y-%m-%d'), dates.max().strftime('%Y-%m-%d'))
    
    if 'store' in df.columns:
        info['unique_stores'] = df['store'].nunique()
    
    if 'family' in df.columns:
        info['unique_families'] = df['family'].nunique()
    
    return info
