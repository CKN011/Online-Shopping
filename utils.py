import pandas as pd
import base64
from io import BytesIO

def export_dataframe_to_csv(df):
    """
    Convert a pandas DataFrame to a CSV string for download
    
    Args:
        df (pandas.DataFrame): The DataFrame to export
        
    Returns:
        str: CSV string
    """
    return df.to_csv(index=False).encode('utf-8')

def create_download_link(buffer, filename, link_text):
    """
    Create an HTML download link for file
    
    Args:
        buffer (BytesIO): Buffer containing file data
        filename (str): Name of the file to download
        link_text (str): Text to display for the download link
        
    Returns:
        str: HTML link
    """
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'

def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a BytesIO object
    
    Args:
        uploaded_file: The uploaded file object from Streamlit
        
    Returns:
        BytesIO: BytesIO object containing the file data
    """
    bytes_data = uploaded_file.getvalue()
    buffer = BytesIO(bytes_data)
    return buffer

def format_percentage(value):
    """
    Format a value as a percentage string
    
    Args:
        value (float): The value to format
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value:.2%}"

def format_decimal(value, precision=2):
    """
    Format a value as a decimal string with specified precision
    
    Args:
        value (float): The value to format
        precision (int, optional): Number of decimal places
        
    Returns:
        str: Formatted decimal string
    """
    format_str = f"{{:.{precision}f}}"
    return format_str.format(value)

def highlight_max(s):
    """
    Highlight the maximum value in a Series
    
    Args:
        s (pandas.Series): The Series to highlight
        
    Returns:
        list: List of styles
    """
    is_max = s == s.max()
    return ['background-color: #ffffb3' if v else '' for v in is_max]

def highlight_min(s):
    """
    Highlight the minimum value in a Series
    
    Args:
        s (pandas.Series): The Series to highlight
        
    Returns:
        list: List of styles
    """
    is_min = s == s.min()
    return ['background-color: #ffb3b3' if v else '' for v in is_min]
