import streamlit as st
import psycopg2
import pandas as pd
import os
import numpy as np
import json
import re
import time
import threading
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import requests

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="DNS Chatbot",
    page_icon="🛍️",
    layout="wide",
)

# Database schema information
DB_SCHEMA = {
    "tables": [
        {
            "name": "pos_order",
            "description": "Point of Sale orders",
            "columns": [
                "id", "name", "date_order", "user_id", "amount_tax", "amount_total", 
                "amount_paid", "amount_return", "create_date", "location_id"
            ]
        },
        {
            "name": "pos_order_line",
            "description": "Lines for each product sold on a POS order",
            "columns": [
                "id", "name", "product_id", "price_unit", "qty", "price_subtotal", 
                "price_subtotal_incl", "total_cost", "is_total_cost_computed", "discount", 
                "order_id", "full_product_name", "refunded_orderline_id", "create_date", 
                "write_date", "sale_order_line_id", "down_payment_details", "product_cost_computed", 
                "product_cost", "product_type", "warehouse_id"
            ]
        },
        {
            "name": "pos_payment",
            "description": "Payments recorded against POS orders",
            "columns": [
                "id", "name", "pos_order_id", "amount", "payment_date", "card_type", 
                "payment_status", "create_date", "location_id", "customer_name"
            ]
        },
        {
            "name": "pos_payment_method",
            "description": "Payment method definitions (e.g., Cash, Card)",
            "columns": [
                "id", "name", "is_cash_count", "create_date"
            ]
        },
        {
            "name": "product_product",
            "description": "Variant-level product info",
            "columns": [
                "id", "product_tmpl_id", "barcode", "create_date"
            ]
        },
        {
            "name": "product_template",
            "description": "Template-level product info",
            "columns": [
                "id", "product_tmpl_id", "barcode", "create_date", "name", "uom_po_id", 
                "purchase_method", "available_in_pos", "pos_categ_id", "online", "categ_id", "vendor_id"
            ]
        },
        {
            "name": "product_category",
            "description": "Product categories (POS or general(AC, etc))",
            "columns": [
                "id", "name", "create_date"
            ]
        },
        {
            "name": "res_partner",
            "description": "Vendor/partner details(if supplier_rank=0 then it is a customer and if it is 1, then it is a vendor)",
            "columns": [
                "id", "display_name", "create_date", "city", "supplier_rank", "is_vendor", "is_customer"
            ]
        },
        {
            "name": "sale_order_line",
            "description": "Lines for Sales Orders (non-POS)",
            "columns": [
                "id", "name", "price_unit", "price_subtotal", "price_tax", "price_total", 
                "price_reduce", "price_reduce_taxinc", "price_reduce_taxexcl", "discount", 
                "product_id", "product_uom_qty", "product_uom", "qty_delivered_method", "create_date"
            ]
        },
        {
            "name": "stock_warehouse",
            "description": "Warehouse/Location definitions",
            "columns": [
                "id", "name", "partner_id", "view_location_id", "create_date"
            ]
        },
        {
            "name": "stock_valuation_layer",
            "description": "Records for stock valuation layers",
            "columns": [
                "id", "company_id", "stock_move_id", "product_id", "remaining_qty", 
                "remaining_value", "create_date"
            ]
        },
        {
            "name": "stock_move",
            "description": "Records of stock movements",
            "columns": [
                "id", "company_id", "product_id", "date", "location_id", "location_dest_id", 
                "product_uom_qty", "state"
            ]
        },
        {
            "name": "res_company",
            "description": "Company details",
            "columns": [
                "id", "name", "currency_id", "create_date"
            ]
        },
        {
            "name": "ir_property",
            "description": "System properties and configuration",
            "columns": [
                "id", "res_id", "name", "value_text", "value_float", "company_id"
            ]
        },
        {
            "name": "stock_quant",
            "description": "Records of stock quantities available at locations",
            "columns": [
                "id", "company_id", "product_id", "location_id", "quantity", "reserved_quantity", "create_date"
            ]
        },
        {
            "name": "stock_location",
            "description": "Details of stock locations",
            "columns": [
                "id", "name", "usage", "parent_path", "create_date"
            ]
        },
        {
            "name": "purchase_order_line",
            "description": "Purchase order line items",
            "columns": [
                "id", "order_id", "product_id", "product_uom_qty", "price_unit", "create_date"
            ]
        },
        {
            "name": "purchase_order",
            "description": "Purchase order details",
            "columns": [
                "id", "date_order", "partner_id", "company_id", "currency_id", "create_date"
            ]
        },
        {
            "name": "stock_picking_type",
            "description": "Types of stock picking for orders",
            "columns": [
                "id", "warehouse_id", "code", "create_date"
            ]
        },
        {
            "name": "res_currency",
            "description": "Currency details",
            "columns": [
                "id", "name", "symbol", "create_date"
            ]
        }
    ]
}

# Initialize or load conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add visualization settings to session state
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "bar"

if "current_data" not in st.session_state:
    st.session_state.current_data = None

@contextmanager
def timeout_spinner(seconds, message="Thinking..."):
    """A safe timeout spinner that works with Streamlit's threading model"""
    status_placeholder = st.empty()
    status_placeholder.info(message)
    completion_status = {"completed": False}
    def check_timeout():
        time.sleep(seconds)
        if not completion_status["completed"]:
            completion_status["timed_out"] = True
    thread = threading.Thread(target=check_timeout, daemon=True)
    thread.start()
    try:
        yield
        completion_status["completed"] = True
        status_placeholder.empty()
    finally:
        completion_status["completed"] = True
        status_placeholder.empty()
        if completion_status.get("timed_out", False):
            st.error(f"Operation timed out after {seconds} seconds. The API might be overloaded. Please try again.")

@st.cache_resource
def get_db_connection():
    """Create and cache the database connection"""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("PGDATABASE", "your_database"),
            user=os.getenv("PGUSER", "your_username"),
            password=os.getenv("PGPASSWORD", "your_password"),
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", "5432")
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None
    
def to_excel(df):
    """Convert dataframe to Excel file"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    processed_data = output.getvalue()
    return processed_data

def to_pdf(fig):
    """Convert matplotlib figure to PDF"""
    output = BytesIO()
    fig.savefig(output, format='pdf', bbox_inches='tight')
    processed_data = output.getvalue()
    return processed_data

def get_download_link(df, filename, file_format="excel"):
    """Generate download link for data"""
    if file_format == "excel":
        data = to_excel(df)
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel File</a>'
        return href
    elif file_format == "pdf":
        fig, ax = plt.subplots(figsize=(12, 6))
        if df.shape[1] <= 10:  # For tables that are not too wide
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.2)
        else:
            # For wider tables, just create a simple title
            ax.text(0.5, 0.5, f"Data Export - {filename}", ha='center', va='center', fontsize=14)
            ax.axis('off')
        data = to_pdf(fig)
        plt.close(fig)
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}.pdf">Download PDF File</a>'
        return href

def query_groq_api(prompt, api_key, model="llama-3.3-70b-versatile", timeout=20):
    """Query the Groq API using Llama 3 models with timeout"""
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful database analyst who answers questions about retail/inventory data. Your responses should include data insights, clear business recommendations, and when appropriate, ABC/XYZ analysis explanations and forecasting interpretations. Format your response with clear sections for Findings and Recommendations only."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 2048,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error calling Groq API: {str(e)}")
        return None
    
def perform_abc_analysis(df, value_column, item_column='product'):
    """
    Perform ABC Analysis on the provided dataframe
    A: Top 80% of value
    B: Next 15% of value
    C: Bottom 5% of value
    """
    if df.empty or value_column not in df.columns:
        return None
    
    # Create a copy to avoid modifying the original
    df_sorted = df.copy()
    
    # Sort by value in descending order
    df_sorted = df_sorted.sort_values(by=value_column, ascending=False)
    
    # Calculate the cumulative percentage
    total_value = df_sorted[value_column].sum()
    df_sorted['cumulative_value'] = df_sorted[value_column].cumsum()
    df_sorted['cumulative_percentage'] = df_sorted['cumulative_value'] / total_value * 100
    
    # Assign ABC classes
    df_sorted['abc_class'] = 'C'
    df_sorted.loc[df_sorted['cumulative_percentage'] <= 80, 'abc_class'] = 'A'
    df_sorted.loc[(df_sorted['cumulative_percentage'] > 80) & 
                 (df_sorted['cumulative_percentage'] <= 95), 'abc_class'] = 'B'
    
    # Count items in each class
    class_counts = df_sorted['abc_class'].value_counts().sort_index()
    class_percentages = (class_counts / len(df_sorted) * 100).round(1)
    
    # Calculate value per class
    class_values = df_sorted.groupby('abc_class')[value_column].sum()
    class_value_percentages = (class_values / total_value * 100).round(1)
    
    # Create summary dataframe
    summary = pd.DataFrame({
        'Class': ['A', 'B', 'C'],
        'Count': [class_counts.get('A', 0), class_counts.get('B', 0), class_counts.get('C', 0)],
        'Count_Percentage': [class_percentages.get('A', 0), class_percentages.get('B', 0), class_percentages.get('C', 0)],
        'Value': [class_values.get('A', 0), class_values.get('B', 0), class_values.get('C', 0)],
        'Value_Percentage': [class_value_percentages.get('A', 0), class_value_percentages.get('B', 0), class_value_percentages.get('C', 0)]
    })
    
    return {
        'detailed': df_sorted,
        'summary': summary
    }

def perform_xyz_analysis(df, item_column, value_column, date_column, cv_threshold_x=0.5, cv_threshold_y=1.0):
    """
    Perform XYZ Analysis based on coefficient of variation
    X: CV <= 0.5 (highly predictable)
    Y: 0.5 < CV <= 1.0 (moderately predictable)
    Z: CV > 1.0 (unpredictable)
    """
    if df.empty or item_column not in df.columns or value_column not in df.columns or date_column not in df.columns:
        return None
    
    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Group by product and month/year to get time series data
    df['month_year'] = df[date_column].dt.to_period('M')
    time_series = df.groupby([item_column, 'month_year'])[value_column].sum().reset_index()
    
    # Calculate mean, std and CV for each product
    product_stats = time_series.groupby(item_column)[value_column].agg(['mean', 'std']).reset_index()
    product_stats['cv'] = product_stats['std'] / product_stats['mean']
    product_stats['cv'] = product_stats['cv'].replace([np.inf, -np.inf, np.nan], 2.0)  # Handle divisions by zero
    
    # Classify products
    product_stats['xyz_class'] = 'Z'
    product_stats.loc[product_stats['cv'] <= cv_threshold_x, 'xyz_class'] = 'X'
    product_stats.loc[(product_stats['cv'] > cv_threshold_x) & 
                 (product_stats['cv'] <= cv_threshold_y), 'xyz_class'] = 'Y'
    
    # Count items in each class
    class_counts = product_stats['xyz_class'].value_counts().sort_index()
    class_percentages = (class_counts / len(product_stats) * 100).round(1)
    
    # Calculate average CV per class
    class_cv = product_stats.groupby('xyz_class')['cv'].mean().round(2)
    
    # Create summary dataframe
    summary = pd.DataFrame({
        'Class': ['X', 'Y', 'Z'],
        'Count': [class_counts.get('X', 0), class_counts.get('Y', 0), class_counts.get('Z', 0)],
        'Count_Percentage': [class_percentages.get('X', 0), class_percentages.get('Y', 0), class_percentages.get('Z', 0)],
        'Avg_CV': [class_cv.get('X', 0), class_cv.get('Y', 0), class_cv.get('Z', 0)]
    })
    
    return {
        'detailed': product_stats,
        'summary': summary
    }

def perform_abc_xyz_analysis(df, item_column, value_column, date_column):
    """Combine ABC and XYZ analyses"""
    abc_result = perform_abc_analysis(df, value_column, item_column)
    xyz_result = perform_xyz_analysis(df, item_column, value_column, date_column)
    
    if abc_result is None or xyz_result is None:
        return None
    
    # Merge ABC and XYZ classifications
    abc_class = abc_result['detailed'][[item_column, 'abc_class']].copy()
    xyz_class = xyz_result['detailed'][[item_column, 'xyz_class']].copy()
    
    combined = pd.merge(abc_class, xyz_class, on=item_column)
    combined['abc_xyz_class'] = combined['abc_class'] + combined['xyz_class']
    
    # Create cross-table summary
    cross_table = pd.crosstab(combined['abc_class'], combined['xyz_class'])
    
    return {
        'detailed': combined,
        'abc_summary': abc_result['summary'],
        'xyz_summary': xyz_result['summary'],
        'cross_table': cross_table
    }

def perform_time_series_forecasting(df, date_column, value_column, forecast_periods=3, method='arima'):
    """Perform time series forecasting using various methods"""
    if df.empty or date_column not in df.columns or value_column not in df.columns:
        return None
    
    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Aggregate data by month
    df['month_year'] = df[date_column].dt.to_period('M')
    monthly_data = df.groupby('month_year')[value_column].sum().reset_index()
    monthly_data['month_year'] = monthly_data['month_year'].dt.to_timestamp()
    
    # Set index to date column for time series analysis
    ts_data = monthly_data.set_index('month_year')[value_column]
    
    # Make predictions based on the chosen method
    if method == 'arima':
        # Simple ARIMA model
        try:
            # Fit ARIMA model (p,d,q)=(1,1,1) as a reasonable default
            model = ARIMA(ts_data, order=(1, 1, 1))
            model_fit = model.fit()
            
            # Make forecast
            forecast = model_fit.forecast(steps=forecast_periods)
            
            # Create result dataframe
            last_date = ts_data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast
            })
            
            historical_df = pd.DataFrame({
                'date': ts_data.index,
                'actual': ts_data.values
            })
            
            return {
                'historical': historical_df,
                'forecast': forecast_df,
                'model': model_fit
            }
        except Exception as e:
            st.warning(f"ARIMA forecasting error: {str(e)}. Trying linear regression instead.")
            method = 'linear'
    
    if method == 'linear':
        # Linear regression with time as feature
        try:
            # Create features (X) as time index and target (y)
            X = np.array(range(len(ts_data))).reshape(-1, 1)
            y = ts_data.values
            
            # Train model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict for future time steps
            X_future = np.array(range(len(ts_data), len(ts_data) + forecast_periods)).reshape(-1, 1)
            forecast = model.predict(X_future)
            
            # Create result dataframe
            last_date = ts_data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'forecast': forecast
            })
            
            historical_df = pd.DataFrame({
                'date': ts_data.index,
                'actual': ts_data.values
            })
            
            return {
                'historical': historical_df,
                'forecast': forecast_df,
                'model': model
            }
        except Exception as e:
            st.warning(f"Linear regression forecasting error: {str(e)}")
            return None
    
    return None

def create_visualization(df, chart_type, x_column=None, y_column=None, category_column=None, title=None):
    """Create visualization based on selected chart type using Matplotlib"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Apply common styling
    plt.style.use('seaborn-v0_8-darkgrid')
    
    if chart_type == 'bar':
        if x_column and y_column:
            sns.barplot(data=df, x=x_column, y=y_column, hue=category_column, ax=ax)
    elif chart_type == 'line':
        if x_column and y_column:
            sns.lineplot(data=df, x=x_column, y=y_column, hue=category_column, marker='o', ax=ax)
    elif chart_type == 'pie':
        if y_column and x_column:
            # Convert to numeric if possible
            if df[y_column].dtype == 'object':
                try:
                    df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
                    df = df.dropna(subset=[y_column])
                except:
                    st.warning(f"Could not convert column '{y_column}' to numeric for pie chart.")
                    return fig
            
            if df.shape[0] > 10:  # Limit to top 10 for readability
                # Ensure the column is numeric before using nlargest
                if pd.api.types.is_numeric_dtype(df[y_column]):
                    top_data = df.nlargest(10, y_column)
                    remaining_sum = df[y_column].sum() - top_data[y_column].sum()
                    if remaining_sum > 0:
                        remaining = pd.DataFrame({
                            x_column: ['Others'],
                            y_column: [remaining_sum]
                        })
                        plot_data = pd.concat([top_data, remaining])
                    else:
                        plot_data = top_data
                else:
                    # If not numeric, just take first 10 rows
                    plot_data = df.head(10)
            else:
                plot_data = df
                
            wedges, texts, autotexts = ax.pie(
                plot_data[y_column], 
                labels=None,  # No labels on the pie to avoid cluttering
                autopct='%1.1f%%', 
                startangle=90, 
                shadow=True,
                explode=[0.05] * len(plot_data)  # Slightly explode all slices
            )
            # Place legend outside the pie chart
            ax.legend(
                wedges, 
                plot_data[x_column],
                title=x_column,
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
            ax.axis('equal')
    elif chart_type == 'scatter':
        if x_column and y_column:
            sns.scatterplot(data=df, x=x_column, y=y_column, hue=category_column, ax=ax)
    elif chart_type == 'histogram':
        if y_column:
            # Convert to numeric if possible
            if df[y_column].dtype == 'object':
                try:
                    df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
                    df = df.dropna(subset=[y_column])
                except:
                    st.warning(f"Could not convert column '{y_column}' to numeric for histogram.")
                    return fig
            
            sns.histplot(data=df, x=y_column, kde=True, ax=ax)
    elif chart_type == 'heatmap':
        if df.shape[0] <= 20 and df.shape[1] <= 20:  # Limit size for readability
            sns.heatmap(df, annot=True, cmap='YlGnBu', ax=ax)
        else:
            ax.text(0.5, 0.5, "Heatmap too large to display", 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
    elif chart_type == 'abc':
        # ABC analysis pareto chart
        if 'cumulative_percentage' in df.columns and x_column and y_column:
            # Convert to numeric if possible
            if df[y_column].dtype == 'object':
                try:
                    df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
                    df = df.dropna(subset=[y_column])
                except:
                    st.warning(f"Could not convert column '{y_column}' to numeric for ABC analysis.")
                    return fig
                    
            sorted_df = df.sort_values(y_column, ascending=False).reset_index(drop=True)
            bars = ax.bar(sorted_df[x_column], sorted_df[y_column], color='skyblue')
            
            # Create second y-axis for cumulative percentage
            ax2 = ax.twinx()
            ax2.plot(sorted_df[x_column], sorted_df['cumulative_percentage'], 
                    'r-', marker='o', markersize=4)
            
            # Draw 80% and 95% horizontal lines
            ax2.axhline(y=80, color='g', linestyle='--', alpha=0.7)
            ax2.axhline(y=95, color='orange', linestyle='--', alpha=0.7)
            
            # Add ABC zones
            idx_80 = (sorted_df['cumulative_percentage'] <= 80).sum()
            idx_95 = (sorted_df['cumulative_percentage'] <= 95).sum()
            
            for i, bar in enumerate(bars):
                if i < idx_80:
                    bar.set_color('green')
                elif i < idx_95:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
            
            ax.set_ylabel('Value')
            ax2.set_ylabel('Cumulative Percentage')
            ax2.set_ylim(0, 105)
    elif chart_type == 'xyz':
        # XYZ analysis scatter plot
        if 'cv' in df.columns and 'mean' in df.columns:
            # Create scatter plot with color coding by XYZ class
            colors = {'X': 'green', 'Y': 'orange', 'Z': 'red'}
            for xyz_class, group in df.groupby('xyz_class'):
                ax.scatter(group['mean'], group['cv'], 
                          label=xyz_class, color=colors.get(xyz_class, 'blue'), alpha=0.7)
            
            # Add horizontal lines at CV thresholds
            ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.5)
            ax.axhline(y=1.0, color='orange', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Mean Value')
            ax.set_ylabel('Coefficient of Variation (CV)')
            ax.legend(title='XYZ Class')
    elif chart_type == 'forecast':
        # Time series forecast plot
        if 'historical' in df and 'forecast' in df:
            # Plot historical data
            ax.plot(df['historical']['date'], df['historical']['actual'], 
                   'b-', marker='o', markersize=4, label='Historical')
            
            # Plot forecast
            ax.plot(df['forecast']['date'], df['forecast']['forecast'], 
                   'r--', marker='x', markersize=4, label='Forecast')
            
            # Add confidence intervals if available
            if 'lower_ci' in df['forecast'].columns and 'upper_ci' in df['forecast'].columns:
                ax.fill_between(df['forecast']['date'], 
                               df['forecast']['lower_ci'],
                               df['forecast']['upper_ci'],
                               color='red', alpha=0.2)
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
    
    # Set title if provided
    if title:
        ax.set_title(title)
    
    # Rotate x-axis labels if there are many categories
    if x_column and df.shape[0] > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Ensure legend is placed correctly and doesn't overlap with the chart
    if chart_type not in ['pie', 'heatmap'] and category_column is not None:
        plt.legend(title=category_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig

def create_plotly_visualization(df, chart_type, x_column=None, y_column=None, category_column=None, title=None):
    """Create interactive plotly visualization based on selected chart type"""
    # Initialize fig as a default figure to prevent UnboundLocalError
    fig = go.Figure()
    
    if df is None or df.empty:
        fig.add_annotation(text="No data available", showarrow=False)
        return fig
    
    # Convert to numeric if possible for relevant columns
    if y_column and y_column in df.columns and df[y_column].dtype == 'object':
        try:
            df[y_column] = pd.to_numeric(df[y_column], errors='coerce')
            df = df.dropna(subset=[y_column])
        except:
            pass
    
    # Create appropriate chart based on type
    if chart_type == 'bar':
        if x_column and y_column:
            if category_column:
                fig = px.bar(df, x=x_column, y=y_column, color=category_column, 
                            title=title, hover_data=df.columns)
            else:
                fig = px.bar(df, x=x_column, y=y_column, 
                            title=title, hover_data=df.columns)
    
    elif chart_type == 'line':
        if x_column and y_column:
            if category_column:
                fig = px.line(df, x=x_column, y=y_column, color=category_column, 
                             markers=True, title=title, hover_data=df.columns)
            else:
                fig = px.line(df, x=x_column, y=y_column, 
                             markers=True, title=title, hover_data=df.columns)
    
    elif chart_type == 'pie':
        if y_column and x_column:
            # Limit to top 10 for readability
            if df.shape[0] > 10:
                if pd.api.types.is_numeric_dtype(df[y_column]):
                    top_data = df.nlargest(10, y_column)
                    remaining_sum = df[y_column].sum() - top_data[y_column].sum()
                    if remaining_sum > 0:
                        remaining = pd.DataFrame({
                            x_column: ['Others'],
                            y_column: [remaining_sum]
                        })
                        plot_data = pd.concat([top_data, remaining])
                    else:
                        plot_data = top_data
                else:
                    plot_data = df.head(10)
            else:
                plot_data = df
                
            fig = px.pie(plot_data, values=y_column, names=x_column, title=title,
                        hover_data=[y_column])
                
    elif chart_type == 'scatter':
        if x_column and y_column:
            if category_column:
                fig = px.scatter(df, x=x_column, y=y_column, color=category_column, 
                                title=title, hover_data=df.columns)
            else:
                fig = px.scatter(df, x=x_column, y=y_column, 
                                title=title, hover_data=df.columns)
    
    elif chart_type == 'histogram':
        if y_column:
            fig = px.histogram(df, x=y_column, title=title)
            
    elif chart_type == 'heatmap':
        if df.shape[0] <= 20 and df.shape[1] <= 20:
            # For heatmap, we need numeric data in a matrix form
            try:
                # Try to find appropriate pivot structure
                if x_column and y_column and category_column:
                    pivot_data = df.pivot(index=x_column, columns=y_column, values=category_column)
                    fig = px.imshow(pivot_data, title=title, labels=dict(color=category_column),
                                  x=pivot_data.columns, y=pivot_data.index)
                else:
                    # Just use the dataframe as is
                    fig = px.imshow(df.select_dtypes(include=['number']).corr(), title=title)
            except:
                fig = go.Figure()
                fig.add_annotation(text="Cannot create heatmap with this data structure", showarrow=False)
        else:
            fig = go.Figure()
            fig.add_annotation(text="Heatmap too large to display", showarrow=False)
    
    elif chart_type == 'abc':
        # ABC analysis pareto chart
        if 'cumulative_percentage' in df.columns and x_column and y_column:
            sorted_df = df.sort_values(y_column, ascending=False).reset_index(drop=True)
            
            # Create a figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bars for values
            fig.add_trace(
                go.Bar(
                    x=sorted_df[x_column],
                    y=sorted_df[y_column],
                    name="Value",
                    hovertemplate="%{x}: %{y}<extra></extra>"
                ),
                secondary_y=False
            )
            
            # Add line for cumulative percentage
            fig.add_trace(
                go.Scatter(
                    x=sorted_df[x_column],
                    y=sorted_df['cumulative_percentage'],
                    name="Cumulative %",
                    line=dict(color='red'),
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>"
                ),
                secondary_y=True
            )
            
            # Add horizontal lines at 80% and 95%
            fig.add_shape(
                type="line",
                x0=0,
                y0=80,
                x1=1,
                y1=80,
                xref="paper",
                yref="y2",
                line=dict(color="green", width=2, dash="dash")
            )
            
            fig.add_shape(
                type="line",
                x0=0,
                y0=95,
                x1=1,
                y1=95,
                xref="paper",
                yref="y2",
                line=dict(color="orange", width=2, dash="dash")
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=x_column,
                yaxis_title="Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Update y-axis
            fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
            fig.update_yaxes(title_text="Value", secondary_y=False)
    
    elif chart_type == 'xyz':
        # XYZ analysis scatter plot
        if 'cv' in df.columns and 'mean' in df.columns and 'xyz_class' in df.columns:
            fig = px.scatter(df, x='mean', y='cv', color='xyz_class',
                           title=title or "XYZ Analysis",
                           labels={'mean': 'Mean Value', 'cv': 'Coefficient of Variation (CV)'},
                           hover_data=df.columns)
            
            # Add horizontal lines at CV thresholds
            fig.add_shape(
                type="line",
                x0=0,
                y0=0.5,
                x1=1,
                y1=0.5,
                xref="paper",
                yref="y",
                line=dict(color="green", width=2, dash="dash")
            )
            
            fig.add_shape(
                type="line",
                x0=0,
                y0=1.0,
                x1=1,
                y1=1.0,
                xref="paper",
                yref="y",
                line=dict(color="orange", width=2, dash="dash")
            )
    
    elif chart_type == 'forecast':
        # Time series forecast plot
        if 'historical' in df and 'forecast' in df:
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(
                go.Scatter(
                    x=df['historical']['date'],
                    y=df['historical']['actual'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue'),
                    hovertemplate="%{x}: %{y}<extra>Historical</extra>"
                )
            )
            
            # Add forecast
            fig.add_trace(
                go.Scatter(
                    x=df['forecast']['date'],
                    y=df['forecast']['forecast'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', dash='dash'),
                    hovertemplate="%{x}: %{y}<extra>Forecast</extra>"
                )
            )
            
            # Add confidence intervals if available
            if 'lower_ci' in df['forecast'].columns and 'upper_ci' in df['forecast'].columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['forecast']['date'].tolist() + df['forecast']['date'].tolist()[::-1],
                        y=df['forecast']['upper_ci'].tolist() + df['forecast']['lower_ci'].tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,0,0,0)'),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )
            
            # Update layout
            fig.update_layout(
                title=title or "Time Series Forecast",
                xaxis_title="Date",
                yaxis_title="Value",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
    else:
        # Default to a simple table view if chart type not supported
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[df[col] for col in df.columns],
                       fill_color='lavender',
                       align='left'))
        ])
        fig.update_layout(title=title or "Data Table")
    
    # Improve layout and hover settings
    fig.update_layout(
        hovermode="closest",
        hoverlabel=dict(
            font_size=12,
            font_family="Arial"
        ),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

class DatabaseAgent:
    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.schema = DB_SCHEMA
        self.api_key = os.getenv("GROQ_API_KEY", "")
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.error_patterns = {
            r"column\s+(\w+\.\w+)\s+does not exist": self._fix_column_not_exist,
            r"relation\s+\"(\w+)\"\s+does not exist": self._fix_relation_not_exist,
            r"missing FROM-clause entry for table \"(\w+)\"": self._fix_missing_from,
            r"column\s+\"(\w+)\"\s+must appear in the GROUP BY clause": self._fix_group_by,
            r"syntax error at or near \"(\w+)\"": self._fix_syntax_error,
            r"function to_char\(unknown, unknown\) is not unique": self._fix_to_char_error
        }
    
    def _fix_to_char_error(self, sql, error_details):
        corrected_sql = re.sub(
            r"TO_CHAR\('(\d{4}-\d{2}-\d{2})',",
            r"TO_CHAR('\1'::date,",
            sql
        )
        return corrected_sql

    def _format_chat_history(self, history):
        """Format conversation history with improved context management"""
        formatted = ""
        # Include more context from history (up from 5 to 8 messages)
        for entry in history[-8:]:
            role = "User" if entry["role"] == "user" else "Assistant"
            formatted += f"{role}: {entry['content']}\n\n"
        return formatted
    
    def _add_case_insensitive_guidelines(self, prompt):
        """Add guidelines for case-insensitive string matching to prompts"""
        guidelines = """
IMPORTANT DATABASE QUERY GUIDELINES:
- Always use case-insensitive string comparisons for text fields
- For PostgreSQL, use ILIKE instead of LIKE for pattern matching
- For exact string matching, use LOWER() function on both sides: 
  Example: WHERE LOWER(location.name) = LOWER('RedHill')
- For partial matching, use: 
  Example: WHERE location.name ILIKE '%redhill%'
- When joining tables with text fields, ensure case handling is consistent
"""
        return prompt + guidelines
    
    def _format_entity_context(self, entities):
        """Format entity information for better SQL generation"""
        context_parts = []
        
        if entities["products"]:
            products_str = ", ".join([f"'{p}'" for p in entities["products"]])
            context_parts.append(f"Products: {products_str}")
        
        if entities["categories"]:
            categories_str = ", ".join([f"'{c}'" for c in entities["categories"]])
            context_parts.append(f"Categories: {categories_str}")
        
        if entities["locations"]:
            locations_str = ", ".join([f"'{l}'" for l in entities["locations"]])
            context_parts.append(f"Locations/Warehouses: {locations_str}")
        
        if entities["payment_methods"]:
            payments_str = ", ".join([f"'{p}'" for p in entities["payment_methods"]])
            context_parts.append(f"Payment Methods: {payments_str}")
        
        if entities["time_period"]:
            context_parts.append(f"Time Period: {entities['time_period']}")
        
        if entities["limit"]:
            context_parts.append(f"Result Limit: {entities['limit']}")
        
        if entities["metrics"]:
            metrics_str = ", ".join(entities["metrics"])
            context_parts.append(f"Metrics Requested: {metrics_str}")
            
        if not context_parts:
            return "No specific entities identified. Processing as a general query."
            
        return "\n".join(context_parts)
    
    def _get_complex_query_instructions(self, intent):
        """Generate specialized instructions for complex multi-condition queries"""
        
        # Detect if this is a complex query with multiple conditions
        has_time_filter = intent["entities"]["time_period"] is not None
        has_location_filter = len(intent["entities"]["locations"]) > 0
        has_category_filter = len(intent["entities"]["categories"]) > 0
        has_product_filter = len(intent["entities"]["products"]) > 0
        has_vendor_filter = len(intent["entities"]["vendors"]) > 0
        
        # If it has multiple conditions, it's complex
        condition_count = sum([has_time_filter, has_location_filter, has_category_filter, has_product_filter, has_vendor_filter])
        
        if condition_count >= 2:
            # This is a complex query with multiple conditions
            conditions = []
            
            if has_time_filter:
                conditions.append(f"Time period: {intent['entities']['time_period']}")
                
            if has_location_filter:
                locations = ", ".join([f"'{loc}'" for loc in intent["entities"]["locations"]])
                conditions.append(f"Location/store: {locations}")
                
            if has_category_filter:
                categories = ", ".join([f"'{cat}'" for cat in intent["entities"]["categories"]])
                conditions.append(f"Product category: {categories}")
                
            if has_product_filter:
                products = ", ".join([f"'{prod}'" for prod in intent["entities"]["products"]])
                conditions.append(f"Products: {products}")
                
            if has_vendor_filter:
                vendors = ", ".join([f"'{v}'" for v in intent["entities"]["vendors"]])
                conditions.append(f"Vendors: {vendors}")
                
            conditions_text = "\n- ".join(conditions)
            
            return f"""
This is a complex query with multiple conditions:
- {conditions_text}

For complex multi-condition queries:
1. Make sure to join all necessary tables (pos_order, pos_order_line, product_template, product_category, stock_warehouse, etc.)
2. Apply case-insensitive filtering for text fields using LOWER() or ILIKE
3. Use DATE_PART('year', date_column) = XXXX for year filtering
4. Apply filters in the WHERE clause for each condition
5. Group by necessary dimensions based on the aggregation level requested

Example SQL for a multi-condition query like this:
```sql
SELECT 
    SUM(pol.price_subtotal) AS total_sales
FROM 
    pos_order po
    INNER JOIN pos_order_line pol ON po.id = pol.order_id
    INNER JOIN product_product pp ON pol.product_id = pp.id
    INNER JOIN product_template pt ON pp.product_tmpl_id = pt.id
    INNER JOIN product_category pc ON pt.categ_id = pc.id
    INNER JOIN stock_warehouse sw ON po.location_id = sw.id
WHERE 
    LOWER(sw.name) = LOWER('RedHill')
    AND LOWER(pc.name) = LOWER('AC')
    AND DATE_PART('year', po.date_order) = 2024
```
"""
        return ""
    
    def _get_specialized_instructions(self, query_type, intent):
        """Generate specialized instructions based on query intent"""
        
        # Base instructions for all query types
        instructions = f"""
Query Type Identified: {query_type}
        """
        
        if query_type == "abc_xyz_analysis":
            instructions += """
For ABC-XYZ analysis, write SQL that:
1. For ABC: Calculate total sales/quantity per product, sort descending, compute cumulative percentages
2. For XYZ: Calculate mean and standard deviation of sales/quantity over time periods, compute coefficient of variation (CV)
3. Assign ABC classes (A: top 80%, B: next 15%, C: bottom 5%) and XYZ classes (X: CV<0.5, Y: 0.5<CV<1, Z: CV>1)
4. Return both classifications with product details
"""
        elif query_type == "abc_analysis":
            instructions += """
For ABC analysis, write SQL that:
1. Calculates total sales/quantity per product
2. Sorts products by value in descending order
3. Computes cumulative percentage contribution
4. Assigns classes: A (top 80%), B (next 15%), C (bottom 5%)
"""
        elif query_type == "xyz_analysis":
            instructions += """
For XYZ analysis, write SQL that:
1. Aggregates sales/quantity by product and time period (month/quarter)
2. Calculates mean and standard deviation for each product
3. Computes coefficient of variation (CV = std/mean)
4. Classifies products: X (CV<0.5), Y (0.5<CV<1), Z (CV>1)
"""
        elif query_type == "forecasting":
            instructions += """
For forecasting queries, write SQL that:
1. Retrieves historical time series data with consistent time periods
2. Ensures data is ordered chronologically
3. Includes enough historical periods for meaningful forecasting (12+ months if possible)
4. Aggregates at appropriate level (daily/weekly/monthly) based on the query
"""
        elif query_type == "inventory_analysis":
            instructions += """
For inventory analysis, write SQL that:
1. Joins necessary inventory tables (stock_quant, stock_location, product_template)
2. Calculates key inventory metrics (on-hand quantity, reserved quantity, available quantity)
3. Groups by relevant dimensions (product, location, etc.)
4. Includes inventory valuation where relevant
"""
        elif query_type == "sales_by_payment_method":
            instructions += """
For payment method analysis, write SQL that:
1. Joins pos_payment, pos_payment_method, and pos_order tables
2. Groups by payment method (card_type or payment method name)
3. Calculates total sales and transaction counts by payment method
4. Orders results appropriately based on the query intent
"""
        elif query_type == "sales_report" or query_type == "sales_by_warehouse" or query_type == "vendor_sales":
            # Enhanced instructions for any sales-related query
            specific_entities = []
            
            # Add entity-specific guidance
            if intent["entities"]["products"]:
                specific_entities.append("specific products")
                instructions += "\nInclude WHERE clauses to filter for the specific products mentioned in the query."
                
            if intent["entities"]["categories"]:
                specific_entities.append("product categories")
                instructions += "\nJoin with product_category table and filter for the specific categories mentioned."
                
            if intent["entities"]["locations"]:
                specific_entities.append("locations/warehouses")
                instructions += "\nJoin with stock_warehouse and/or stock_location tables and filter for specific locations."
                
            if intent["entities"]["time_period"]:
                specific_entities.append("time period")
                instructions += "\nInclude date filtering for the specific time period mentioned."
            
            # Combine all entity mentions into a summary
            if specific_entities:
                entity_summary = ", ".join(specific_entities)
                instructions += f"\n\nThis query involves {entity_summary}. Make sure to handle these specific entities in the SQL query."
            
            instructions += """
For sales analysis, write SQL that:
1. Joins the necessary tables to get complete sales information
2. Includes product, category, location, and/or time dimensions as appropriate
3. Calculates sales metrics (revenue, quantity, profit) based on the request
4. Orders results based on the most relevant metric
5. Uses appropriate filters for any specific entities mentioned
"""
        
        return instructions
    
    def _build_prompt(self, query, history):
        schema_str = json.dumps(self.schema, indent=2)
        history_str = self._format_chat_history(history)
        
        # Extract entities and classify intent with LLM assistance
        intent = self.understand_intent(query)
        query_type = intent["query_type"]
        
        # Extract entities for targeted prompting
        entity_context = self._format_entity_context(intent["entities"])
        
        # Create improved instructions for LLM
        special_instructions = self._get_specialized_instructions(query_type, intent)
        
        # Add complex query instructions if applicable
        complex_instructions = self._get_complex_query_instructions(intent)
        if complex_instructions:
            special_instructions += "\n" + complex_instructions
        
        prompt = f"""You are a database analyst who answers questions about a retail/inventory PostgreSQL database. You provide detailed analysis and actionable business recommendations based on the data.

Database Schema:
{schema_str}

Your task is to:
1. Understand the user's question in the context of the database schema and conversation history
2. Generate an appropriate PostgreSQL query to answer the question
3. Execute the query (I'll handle this part)
4. Interpret the results and provide a clear, thorough analysis
5. Provide 2-3 specific, actionable business recommendations based on the data

Relevant entities identified in the query:
{entity_context}

{special_instructions}

Important SQL query tips for PostgreSQL:
- ALWAYS use case-insensitive text comparisons with ILIKE or LOWER() for text matching
- For warehouse names, use: WHERE LOWER(sw.name) = LOWER('RedHill') or sw.name ILIKE '%redhill%'
- For product categories, use: WHERE LOWER(pc.name) = LOWER('AC') or pc.name ILIKE '%ac%'
- For vendors, use: WHERE LOWER(rp.name) = LOWER('Vendor') or rp.name ILIKE '%vendor%'
- When combining multiple filters, use LEFT JOIN instead of INNER JOIN when appropriate
- Always use the SQL keyword AS for aliases
- Use explicit INNER JOIN or LEFT JOIN syntax (not just JOIN)
- Avoid using ORDER BY in subqueries
- For calculations, use explicit CAST when mixing numeric types
- When using aliases in GROUP BY or ORDER BY, refer to the column name, not the alias
- Remember to add GROUP BY clauses for all non-aggregated columns in SELECT
- Product names are stored in product_template.name, not product_product.name
- Use appropriate table aliases like 'pt' for product_template, 'pp' for product_product, 'pol' for pos_order_line, 'ppm' for pos_payment_method
- Total sales can be calculated using either qty * price_unit or using price_subtotal from pos_order_line
- Payment methods can be analyzed using the card_type field from pos_payment_method table

Previous conversation for context:
{history_str}

User question: {query}

First, think about the best SQL query to answer this question based on the schema and conversation context.
Then, provide your SQL query within ```sql ``` tags.
"""
        # Add case-insensitive guidelines
        prompt = self._add_case_insensitive_guidelines(prompt)
        return prompt
    
    def _build_error_correction_prompt(self, sql, error_msg):
        prompt = f"""The following SQL query for PostgreSQL has an error:

```sql
{sql}
```

The error message is: {error_msg}

Please fix the SQL query while preserving its original intent. Return only the corrected SQL query within sql tags, with no additional text. Focus on fixing just the specific error mentioned, making minimal changes to the original query.

Make sure to use case-insensitive comparisons (ILIKE or LOWER()) for any string matching operations.
"""
        return prompt
    
    def _generate_sql(self, prompt):
        if self.api_key:
            return query_groq_api(prompt, self.api_key, self.model)
        else:
            st.error("No Groq API key provided. Please set the GROQ_API_KEY environment variable.")
        return None

    def _extract_sql_query(self, llm_response):
        if not llm_response:
            return None
        sql_match = re.search(r"```sql\s*(.*?)\s*```", llm_response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        return None

    def _fix_column_not_exist(self, sql, error_details):
        match = re.search(r"column\s+(\w+\.\w+)\s+does not exist", error_details)
        if match:
            bad_column = match.group(1)
            table_alias = bad_column.split('.')[0]
            column_name = bad_column.split('.')[1]
            replacements = {
                "p.name": "pt.name",
                "pr.name": "pt.name",
                "pp.name": "pt.name",
                "c.name": "pc.name",
                "w.name": "sw.name",
                "pm.name": "ppm.card_type"
            }
            if bad_column in replacements:
                return sql.replace(bad_column, replacements[bad_column])
            if "HINT" in error_details:
                hint_match = re.search(r"Perhaps you meant to reference the column \"(\w+\.\w+)\"", error_details)
                if hint_match:
                    suggested_column = hint_match.group(1)
                    return sql.replace(bad_column, suggested_column)
            correction_prompt = self._build_error_correction_prompt(sql, error_details)
            correction_response = self._generate_sql(correction_prompt)
            corrected_sql = self._extract_sql_query(correction_response)
            if corrected_sql:
                return corrected_sql
        return sql

    def _fix_relation_not_exist(self, sql, error_details):
        match = re.search(r"relation\s+\"(\w+)\"\s+does not exist", error_details)
        if match:
            bad_relation = match.group(1)
            replacements = {
                "product": "product_product",
                "template": "product_template",
                "category": "product_category",
                "warehouse": "stock_warehouse",
                "order": "pos_order",
                "order_line": "pos_order_line",
                "partner": "res_partner",
                "payment_method": "pos_payment_method",
                "payment": "pos_payment"
            }
            for wrong, correct in replacements.items():
                if wrong == bad_relation or wrong in bad_relation:
                    return sql.replace(bad_relation, correct)
            correction_prompt = self._build_error_correction_prompt(sql, error_details)
            correction_response = self._generate_sql(correction_prompt)
            corrected_sql = self._extract_sql_query(correction_response)
            if corrected_sql:
                return corrected_sql
        return sql

    def _fix_missing_from(self, sql, error_details):
        match = re.search(r"missing FROM-clause entry for table \"(\w+)\"", error_details)
        if match:
            missing_table_alias = match.group(1)
            correction_prompt = self._build_error_correction_prompt(sql, error_details)
            correction_response = self._generate_sql(correction_prompt)
            corrected_sql = self._extract_sql_query(correction_response)
            if corrected_sql:
                return corrected_sql
        return sql

    def _fix_group_by(self, sql, error_details):
        match = re.search(r"column\s+\"(\w+)\"\s+must appear in the GROUP BY clause", error_details)
        if match:
            missing_column = match.group(1)
            group_by_match = re.search(r"GROUP BY\s+(.*?)(?:ORDER BY|LIMIT|$)", sql, re.IGNORECASE | re.DOTALL)
            if group_by_match:
                group_by_clause = group_by_match.group(1).strip()
                new_group_by = f"GROUP BY {group_by_clause}, {missing_column}"
                return sql.replace(f"GROUP BY {group_by_clause}", new_group_by)
            else:
                order_by_match = re.search(r"(ORDER BY\s+.*?)(?:LIMIT|$)", sql, re.IGNORECASE | re.DOTALL)
                if order_by_match:
                    return sql.replace(order_by_match.group(0), f"GROUP BY {missing_column} {order_by_match.group(0)}")
                limit_match = re.search(r"(LIMIT\s+\d+)", sql, re.IGNORECASE)
                if limit_match:
                    return sql.replace(limit_match.group(0), f"GROUP BY {missing_column} {limit_match.group(0)}")
                return f"{sql} GROUP BY {missing_column}"
        correction_prompt = self._build_error_correction_prompt(sql, error_details)
        correction_response = self._generate_sql(correction_prompt)
        corrected_sql = self._extract_sql_query(correction_response)
        if corrected_sql:
            return corrected_sql
        return sql

    def _fix_syntax_error(self, sql, error_details):
        correction_prompt = self._build_error_correction_prompt(sql, error_details)
        correction_response = self._generate_sql(correction_prompt)
        corrected_sql = self._extract_sql_query(correction_response)
        if corrected_sql:
            return corrected_sql
        return sql

    def _auto_correct_sql(self, sql, error_msg):
        for pattern, fix_function in self.error_patterns.items():
            if re.search(pattern, error_msg, re.IGNORECASE):
                corrected_sql = fix_function(sql, error_msg)
                if corrected_sql != sql:
                    return corrected_sql
        correction_prompt = self._build_error_correction_prompt(sql, error_msg)
        correction_response = self._generate_sql(correction_prompt)
        corrected_sql = self._extract_sql_query(correction_response)
        if corrected_sql:
            return corrected_sql
        return sql
        
    def _classify_error(self, error_message):
        """Classifies database errors for better user feedback"""
        error_message = str(error_message).lower()
        
        if any(term in error_message for term in ["relation", "table", "doesn't exist", "does not exist"]):
            return "missing_table"
        elif any(term in error_message for term in ["syntax error", "parse error"]):
            return "syntax_error"
        elif any(term in error_message for term in ["permission", "privilege", "not allowed"]):
            return "permission_error"
        elif any(term in error_message for term in ["timeout", "timed out"]):
            return "timeout_error"
        else:
            return "unknown_error"
    
    def generate_enhanced_pdf(self, data, analysis_text, chart_fig=None):
        """
        Generate an enhanced PDF report with data tables and visualizations
        
        Args:
            data: DataFrame containing the data
            analysis_text: Text analysis to include in the PDF
            chart_fig: Optional matplotlib figure to include
            
        Returns:
            PDF file as bytes
        """
        output = BytesIO()
        
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            
            # Create the PDF document
            doc = SimpleDocTemplate(output, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Create custom styles
            title_style = ParagraphStyle(
                'Title',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=12
            )
            
            heading_style = ParagraphStyle(
                'Heading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=10
            )
            
            # Start building the PDF content
            content = []
            
            # Add title
            content.append(Paragraph("Data Analysis Report", title_style))
            content.append(Spacer(1, 0.25*inch))
            
            # Add date
            content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
            content.append(Spacer(1, 0.25*inch))
            
            # Add analysis summary
            content.append(Paragraph("Analysis", heading_style))
            
            # Split analysis text and format paragraphs
            for para in analysis_text.split('\n\n'):
                if para.strip():
                    if para.strip().lower().startswith(('findings', 'recommendations')):
                        content.append(Paragraph(para, heading_style))
                    else:
                        content.append(Paragraph(para, styles["Normal"]))
                    content.append(Spacer(1, 0.1*inch))
            
            content.append(Spacer(1, 0.25*inch))
            
            # Add chart if provided
            if chart_fig is not None:
                # Save the chart to a temporary file
                chart_data = BytesIO()
                chart_fig.savefig(chart_data, format='png', dpi=300, bbox_inches='tight')
                chart_data.seek(0)
                
                # Add the chart to the PDF
                content.append(Paragraph("Data Visualization", heading_style))
                img = Image(chart_data, width=6*inch, height=4*inch)
                content.append(img)
                content.append(Spacer(1, 0.25*inch))
            
            # Add data table
            content.append(Paragraph("Data Table", heading_style))
            
            # Limit the number of rows and columns for readability
            max_rows = 50
            max_cols = 10
            
            if data is not None and not data.empty:
                # Truncate data if necessary
                display_data = data.head(max_rows)
                
                if len(display_data.columns) > max_cols:
                    display_data = display_data.iloc[:, :max_cols]
                
                # Convert DataFrame to a list of lists
                table_data = [display_data.columns.tolist()]
                for i, row in display_data.iterrows():
                    table_data.append(row.tolist())
                
                # Create the table
                table = Table(table_data, repeatRows=1)
                
                # Add table styles
                table_style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ])
                table.setStyle(table_style)
                
                # Add the table to the content
                content.append(table)
                
                # Add note about data truncation if necessary
                if len(data) > max_rows or len(data.columns) > max_cols:
                    content.append(Spacer(1, 0.1*inch))
                    truncate_note = f"Note: Data has been truncated. Full dataset contains {len(data)} rows and {len(data.columns)} columns."
                    content.append(Paragraph(truncate_note, styles["Italic"]))
            
            # Build the PDF
            doc.build(content)
            pdf_data = output.getvalue()
            return pdf_data
            
        except Exception as e:
            # Fallback to a simpler PDF if ReportLab fails
            try:
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_pdf import PdfPages
                
                with PdfPages(output) as pdf:
                    # Create a title page
                    fig_title = plt.figure(figsize=(8.5, 11))
                    fig_title.text(0.5, 0.9, "Data Analysis Report", fontsize=16, ha='center')
                    fig_title.text(0.5, 0.85, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=12, ha='center')
                    fig_title.text(0.1, 0.7, analysis_text, fontsize=10, va='top', wrap=True)
                    pdf.savefig(fig_title)
                    plt.close(fig_title)
                    
                    # Add the chart if available
                    if chart_fig is not None:
                        pdf.savefig(chart_fig)
                    
                    # Create a data table page
                    if data is not None and not data.empty:
                        fig_data = plt.figure(figsize=(8.5, 11))
                        fig_data.text(0.5, 0.95, "Data Table", fontsize=14, ha='center')
                        
                        # Truncate data for display
                        display_data = data.head(40)
                        
                        ax = fig_data.add_subplot(111)
                        ax.axis('off')
                        table = ax.table(
                            cellText=display_data.values,
                            colLabels=display_data.columns,
                            loc='center',
                            cellLoc='center'
                        )
                        table.auto_set_font_size(False)
                        table.set_fontsize(8)
                        table.scale(1, 1.5)
                        pdf.savefig(fig_data)
                        plt.close(fig_data)
                
                return output.getvalue()
                
            except Exception as sub_error:
                # If all else fails, return None or a very basic PDF
                return None
        
    def _ensure_case_insensitive_matching(self, sql):
        """Modify SQL to ensure case-insensitive text matching"""
        # Replace common equality text comparisons with case-insensitive versions
        sql = re.sub(
            r'(\w+\.\w+)\s*=\s*[\'"]([^\'"]+)[\'"]',
            r'LOWER(\1) = LOWER(\'\2\')',
            sql
        )
        
        # Replace LIKE with ILIKE for pattern matching
        sql = re.sub(r'\bLIKE\b', 'ILIKE', sql, flags=re.IGNORECASE)
        
        return sql

    def _create_relaxed_query(self, sql, empty_result=False):
        """Create a less restrictive version of a query that's returning no results"""
        # If there are multiple conditions in the WHERE clause, try relaxing them
        where_match = re.search(r'(WHERE\s+.*?)(?:GROUP BY|ORDER BY|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            conditions = re.split(r'\s+AND\s+', where_clause[6:], flags=re.IGNORECASE)
            
            # If we have multiple conditions, try with fewer
            if len(conditions) > 1:
                # Only use conditions that are likely to be the main filters (not joins)
                main_conditions = [c for c in conditions if re.search(r'(LOWER|ILIKE|name|date)', c, re.IGNORECASE)]
                if main_conditions:
                    # Use only the first condition initially for maximum results
                    relaxed_where = "WHERE " + main_conditions[0]
                    relaxed_sql = sql.replace(where_clause, relaxed_where)
                    return relaxed_sql
                    
        # If relaxing WHERE didn't work or wasn't applicable, try converting INNER JOINs to LEFT JOINs
        if empty_result:
            relaxed_sql = re.sub(
                r'INNER JOIN',
                'LEFT JOIN',
                sql,
                flags=re.IGNORECASE
            )
            if relaxed_sql != sql:
                return relaxed_sql
                
        return sql

    def _execute_query(self, sql, max_retries=6):
        if not self.db_conn:
            return {"success": False, "error": "No database connection available"}
        retries = 0
        current_sql = sql
        last_error = None
        
        # Pre-process SQL to ensure case-insensitive matching
        current_sql = self._ensure_case_insensitive_matching(current_sql)
        
        while retries <= max_retries:
            cursor = None
            try:
                cursor = self.db_conn.cursor()
                cursor.execute(current_sql)
                columns = [desc[0] for desc in cursor.description]
                results = cursor.fetchall()
                
                # If query returned empty result set but we have retries left,
                # try to relax the conditions
                if len(results) == 0 and retries < max_retries:
                    if cursor:
                        cursor.close()
                    relaxed_sql = self._create_relaxed_query(current_sql, empty_result=True)
                    if relaxed_sql != current_sql:
                        current_sql = relaxed_sql
                        retries += 1
                        continue
                
                df = pd.DataFrame(results, columns=columns)
                col_name_counts = defaultdict(int)
                new_columns = []
                for col in df.columns:
                    if col in new_columns:
                        col_name_counts[col] += 1
                        new_col = f"{col}_{col_name_counts[col]}"
                    else:
                        new_col = col
                    new_columns.append(new_col)
                df.columns = new_columns
                cursor.close()
                return {"success": True, "data": df}
            except Exception as e:
                last_error = str(e)
                if cursor:
                    cursor.close()
                if retries < max_retries:
                    # More aggressive error correction
                    corrected_sql = self._auto_correct_sql(current_sql, last_error)
                    if corrected_sql != current_sql:
                        current_sql = corrected_sql
                        retries += 1
                        continue
                break
        return {"success": False, "error": last_error}
    
    def _create_enhanced_response(self, query, llm_response, sql_query, data):
        if data is not None:
            # Prepare data sample for the prompt
            data_sample = data.head(50)
            
            # Handle different data types for serialization
            for col in data_sample.columns:
                if pd.api.types.is_datetime64_any_dtype(data_sample[col]):
                    data_sample[col] = data_sample[col].astype(str)
                elif pd.api.types.is_complex_dtype(data_sample[col]):
                    data_sample[col] = data_sample[col].astype(str)
            
            # Convert to JSON for the prompt
            try:
                data_str = data_sample.to_json(orient="records")
            except:
                # Fallback for complex datatypes that can't be easily serialized
                data_str = str(data_sample.values.tolist())
            
            # Include data statistics in the prompt
            stats_info = ""
            try:
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    stats = data[numeric_cols].describe().to_dict()
                    stats_info = f"\nData Statistics: {json.dumps(stats)}"
            except:
                pass
            
            # Include original query for better context
            analysis_prompt = f"""Here are the results of the SQL query for the question: "{query}"

SQL Query:
{sql_query}

Query Results (first 50 rows): {data_str}
{stats_info}

Data Shape: {data.shape[0]} rows, {data.shape[1]} columns
Columns: {', '.join(data.columns.tolist())}

Based on this data, please provide:

1. A clear explanation of what the data shows - focus specifically on answering the user's original question
2. Always give the findings and recommendations in point form (eg. 1., 2., 3.)
2. Identification of any patterns, trends, or anomalies in the data
3. 2-3 specific, actionable business recommendations based on these findings

The user asked specifically about: "{query}"
Make sure your response directly addresses this question with insights from the data.

IMPORTANT FORMATTING INSTRUCTIONS:
- Do NOT include additional SQL queries
- Do NOT mention "follow-up questions" at all
- Keep your response concise (max 250-300 words)
- Focus only on the immediate question and insights
- Format your response in a clear, professional manner
- Divide your response into "Findings" and "Recommendations" sections only
""" 
            enhanced_response = query_groq_api(analysis_prompt, self.api_key, self.model) 
            if enhanced_response: 
                cleaned_response = self._clean_response(enhanced_response) 
                return cleaned_response 
        return llm_response
        
    def process_abc_xyz_analysis(self, df, item_column, value_column, date_column):
        """Process ABC-XYZ analysis based on query results"""
        if df is None or df.empty:
            return None
        
        try:
            # Check if required columns exist and use smart column detection
            if item_column not in df.columns:
                item_column = next((col for col in df.columns if any(x in col.lower() for x in ['product', 'item', 'name', 'sku', 'description'])), None)
            
            if value_column not in df.columns:
                value_column = next((col for col in df.columns if any(x in col.lower() for x in ['sales', 'revenue', 'qty', 'quantity', 'amount', 'value', 'total', 'sum', 'count'])), None)
                
            if date_column not in df.columns:
                date_column = next((col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'period', 'month', 'year', 'day', 'week'])), None)
            
            # If we still don't have our required columns, return None
            if not all([item_column, value_column, date_column]):
                return None
                
            # Perform the analysis
            return perform_abc_xyz_analysis(df, item_column, value_column, date_column)
        except Exception as e:
            st.warning(f"Could not perform ABC-XYZ analysis: {str(e)}")
            return None
    
    def process_forecasting(self, df, date_column=None, value_column=None, periods=3, method='arima'):
        """Process time series forecasting based on query results with improved column detection"""
        if df is None or df.empty:
            return None
        
        try:
            # Auto-detect columns if not provided with expanded patterns
            if date_column is None or date_column not in df.columns:
                date_column = next((col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'period', 'month', 'year', 'day', 'timestamp'])), None)
            
            if value_column is None or value_column not in df.columns:
                value_column = next((col for col in df.columns if any(x in col.lower() for x in ['sales', 'revenue', 'qty', 'quantity', 'amount', 'value', 'total', 'price', 'income', 'profit'])), None)
            
            # If we still don't have our required columns, try more general approach - any datetime column and any numeric column
            if date_column is None:
                date_cols = df.select_dtypes(include=['datetime64']).columns
                if len(date_cols) > 0:
                    date_column = date_cols[0]
            
            if value_column is None:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    value_column = numeric_cols[0]
            
            # If we still don't have our required columns, return None
            if not all([date_column, value_column]):
                return None
                
            # Convert date column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                try:
                    df[date_column] = pd.to_datetime(df[date_column])
                except:
                    return None  # Can't convert to datetime
                
            # Perform the forecasting
            return perform_time_series_forecasting(df, date_column, value_column, periods, method)
        except Exception as e:
            st.warning(f"Could not perform forecasting: {str(e)}")
            return None
    
    def process_query(self, query, history):
        """
        Process user query through the entire pipeline with improved error handling
        """
        processing_status = st.empty()
        
        # Step 1: Understand the intent behind the query
        intent = self.understand_intent(query)
        processing_status.info("Building query...")
        
        # Step 2: Build a prompt using the intent and history
        prompt = self._build_prompt(query, history)
        processing_status.info("Generating SQL query...")
        
        # Step 3: Generate SQL using LLM
        llm_response = self._generate_sql(prompt)
        processing_status.empty()
        
        if not llm_response:
            return {
                "response": "I'm sorry, I couldn't generate a response. Please try again later.",
                "sql": None,
                "data": None,
                "analysis_type": None,
                "analysis_data": None
            }
            
        # Step 4: Extract SQL from the response
        sql_query = self._extract_sql_query(llm_response)
        if not sql_query:
            return {
                "response": "I'm sorry, I couldn't generate a valid SQL query. Could you rephrase your question?",
                "sql": None,
                "data": None,
                "analysis_type": None,
                "analysis_data": None
            }
            
        # Step 5: Execute the query with retry logic
        if self.db_conn:
            processing_status.info("Executing query...")
            result = self._execute_query(sql_query)
            processing_status.empty()
            
            # Check for special analysis types
            analysis_type = None
            analysis_data = None
            
            if result["success"]:
                # Store the data for potential download
                st.session_state.current_data = result["data"]
                
                # Step 6: Determine if special analysis is needed
                if intent["query_type"] == "abc_xyz_analysis":
                    processing_status.info("Performing ABC-XYZ analysis...")
                    analysis_type = "abc_xyz"
                    analysis_data = self.process_abc_xyz_analysis(
                        result["data"], 
                        next((col for col in result["data"].columns if "product" in col.lower()), None),
                        next((col for col in result["data"].columns if any(x in col.lower() for x in ["sales", "total", "revenue", "amount"])), None),
                        next((col for col in result["data"].columns if "date" in col.lower()), None)
                    )
                elif intent["query_type"] == "abc_analysis":
                    processing_status.info("Performing ABC analysis...")
                    analysis_type = "abc"
                    abc_xyz_result = self.process_abc_xyz_analysis(
                        result["data"], 
                        next((col for col in result["data"].columns if "product" in col.lower()), None),
                        next((col for col in result["data"].columns if any(x in col.lower() for x in ["sales", "total", "revenue", "amount"])), None),
                        next((col for col in result["data"].columns if "date" in col.lower()), None)
                    )
                    if abc_xyz_result:
                        analysis_data = {
                            'detailed': abc_xyz_result['detailed'],
                            'summary': abc_xyz_result['abc_summary']
                        }
                elif intent["query_type"] == "xyz_analysis":
                    processing_status.info("Performing XYZ analysis...")
                    analysis_type = "xyz"
                    abc_xyz_result = self.process_abc_xyz_analysis(
                        result["data"], 
                        next((col for col in result["data"].columns if "product" in col.lower()), None),
                        next((col for col in result["data"].columns if any(x in col.lower() for x in ["sales", "total", "revenue", "amount"])), None),
                        next((col for col in result["data"].columns if "date" in col.lower()), None)
                    )
                    if abc_xyz_result:
                        analysis_data = {
                            'detailed': abc_xyz_result['detailed'],
                            'summary': abc_xyz_result['xyz_summary']
                        }
                # Check for forecasting requests
                elif intent["query_type"] == "forecasting":
                    processing_status.info("Performing forecasting analysis...")
                    analysis_type = "forecast"
                    analysis_data = self.process_forecasting(
                        result["data"],
                        next((col for col in result["data"].columns if "date" in col.lower()), None),
                        next((col for col in result["data"].columns if any(x in col.lower() for x in ["sales", "total", "revenue", "amount"])), None),
                        3,  # Default forecast periods
                        'arima'  # Default method
                    )
                
                # Step 7: Generate enhanced response with results
                processing_status.info("Analyzing results...")
                enhanced_response = self._create_enhanced_response(query, llm_response, sql_query, result["data"])
                processing_status.empty()
                
                return {
                    "response": enhanced_response,
                    "sql": sql_query,
                    "data": result["data"],
                    "analysis_type": analysis_type,
                    "analysis_data": analysis_data
                }
            else:
                # Step 8: Handle query errors with more useful feedback
                error_type = self._classify_error(result['error'])
                if error_type == "missing_table":
                    error_msg = f"I couldn't access some information in the database. There might be issues with the table structure. Database error: {result['error']}"
                elif error_type == "syntax_error":
                    error_msg = f"I created a query with incorrect syntax. Let me try to rephrase your question or ask in a different way. Technical error: {result['error']}"
                elif error_type == "permission_error":
                    error_msg = "I don't have permission to access some of this data. Please contact your database administrator."
                else:
                    error_msg = f"I'm sorry, I couldn't retrieve the data. The database reported: {result['error']}"
                
                return {
                    "response": error_msg,
                    "sql": sql_query,
                    "data": None,
                    "analysis_type": None,
                    "analysis_data": None
                }
        else:
            error_msg = "Database connection is not available. Please check your database configuration."
            return {
                "response": error_msg,
                "sql": sql_query,
                "data": None,
                "analysis_type": None,
                "analysis_data": None
            }
    
    def _clean_response(self, response):
        # Remove sections that shouldn't be included
        patterns = [
            r"(?i)Follow-up Questions.*?(?=\n\n|\Z)",
            r"(?i)Additional Queries.*?(?=\n\n|\Z)",
            r"(?i)Further Analysis.*?(?=\n\n|\Z)",
            r"(?i)Next Steps.*?(?=\n\n|\Z)",
            r"(?i)Additional Considerations.*?(?=\n\n|\Z)"
        ]
        for pattern in patterns:
            response = re.sub(pattern, "", response, flags=re.DOTALL)
        
        # Remove individual lines with unwanted content
        lines = response.split('\n')
        filtered_lines = []
        for line in lines:
            if not re.search(r"(?i)follow.?up|additional.?quer|further.?analy", line):
                filtered_lines.append(line)
        
        # Reassemble and clean up extra whitespace
        response = '\n'.join(filtered_lines)
        response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
        return response.strip()
    
    def understand_intent(self, query):
        """
        Enhanced function to detect query intent using semantic understanding
        rather than keyword-based matching.
        """
        # Initialize with default values to prevent None issues
        intent = {
            "original_query": query,
            "query_type": "unknown",
            "entities": {
                "time_period": None,
                "products": [],
                "categories": [],
                "locations": [],
                "metrics": [],
                "payment_methods": [],
                "vendors": [],
                "limit": None,
                "filters": {},
                "grouping": []
            },
            "analysis_type": "summary",
            "comparison": None
        }

        # Use LLM for intent and entity recognition if API key is available
        if self.api_key:
            try:
                intent_prompt = f"""Analyze this retail/inventory database query and identify:
    1. The main query type from these options:
    - sales_report (general sales questions)
    - inventory_analysis (stock levels, availability)
    - sales_by_warehouse (sales by location)
    - sales_by_payment_method (payment analysis)
    - vendor_sales (vendor/supplier analysis)
    - abc_analysis (Pareto/ABC classification)
    - xyz_analysis (demand variability)
    - abc_xyz_analysis (combined ABC-XYZ)
    - forecasting (predictions and projections)
    - customer_insights (customer behavior)
    - product_performance (specific product analysis)
    - ranking (top/worst performers)
    - unknown (if none of the above)

    2. Entities mentioned:
    - Specific products by name
    - Product categories
    - Locations or warehouses
    - Time periods (dates, months, quarters)
    - Metrics requested (sales, revenue, profit, margin, etc.)
    - Payment methods
    - Limit values (top 10, etc.)
    - Comparison requests

    Return a JSON object with these fields. For arrays, return empty arrays if none found.
    Be thorough in identifying entities - search for specific product names, categories, etc.

    Query: {query}
    """
                
                intent_response = query_groq_api(intent_prompt, self.api_key, self.model)
                if intent_response:
                    try:
                        # Try to extract JSON from the response
                        json_match = re.search(r"```json\s*(.*?)\s*```", intent_response, re.DOTALL)
                        if json_match:
                            intent_response = json_match.group(1)
                        
                        # Try alternate formats if the LLM didn't wrap in code blocks
                        if not json_match:
                            # Try to extract just a JSON object
                            json_match = re.search(r"\{.*\}", intent_response, re.DOTALL)
                            if json_match:
                                intent_response = json_match.group(0)
                        
                        # Parse the JSON response
                        semantic_intent = json.loads(intent_response)
                        
                        # Update intent with LLM's understanding
                        if "query_type" in semantic_intent:
                            intent["query_type"] = semantic_intent["query_type"]
                        
                        # Extract entities
                        if "entities" in semantic_intent:
                            entities = semantic_intent["entities"]
                            for key in entities:
                                if key in intent["entities"]:
                                    intent["entities"][key] = entities[key]
                        else:
                            # Look for individual entity fields
                            for entity_key in ["products", "categories", "locations", "time_period", 
                                            "metrics", "payment_methods", "limit", "vendors"]:
                                if entity_key in semantic_intent:
                                    intent["entities"][entity_key] = semantic_intent[entity_key]
                        
                        # Handle special fields
                        if "comparison" in semantic_intent:
                            intent["comparison"] = semantic_intent["comparison"]
                        
                        if "analysis_type" in semantic_intent:
                            intent["analysis_type"] = semantic_intent["analysis_type"]
                    except Exception as e:
                        # If JSON parsing fails, continue with the basic intent
                        pass
            except Exception as e:
                # If overall LLM processing fails, continue with the basic intent
                pass
        
        # Enhance entity extraction with more specific LLM prompting
        intent = self._enhance_entity_extraction(query, intent)
        
        # Fallback to rule-based approach if LLM didn't provide a type
        if intent["query_type"] == "unknown":
            query_lower = query.lower()
            # ABC/XYZ Analysis patterns
            if (("abc" in query_lower and "xyz" in query_lower) or 
                "abc-xyz" in query_lower or "abc xyz" in query_lower):
                intent["query_type"] = "abc_xyz_analysis"
                intent["analysis_type"] = "abc_xyz"
            elif ("abc" in query_lower and 
                ("analysis" in query_lower or "classification" in query_lower or "categorize" in query_lower)):
                intent["query_type"] = "abc_analysis" 
                intent["analysis_type"] = "abc"
            elif ("xyz" in query_lower and 
                ("analysis" in query_lower or "classification" in query_lower or "categorize" in query_lower)):
                intent["query_type"] = "xyz_analysis"
                intent["analysis_type"] = "xyz"
            # Forecasting patterns
            elif any(x in query_lower for x in [
                "forecast", "predict", "prediction", "forecasting", "future", 
                "next month", "next quarter", "next year", "upcoming", "projected"
            ]):
                intent["query_type"] = "forecasting"
                intent["analysis_type"] = "predictive"
            # Warehouse/Location patterns
            elif any(term in query_lower for term in ["warehouse", "store", "location", "branch", "site"]):
                intent["query_type"] = "sales_by_warehouse"
            # Vendor Sales
            elif any(term in query_lower for term in ["vendor", "supplier", "manufacturer", "distributor"]):
                intent["query_type"] = "vendor_sales"
            # Default to sales_report if nothing else matched and it has "sales" in it
            elif "sales" in query_lower:
                intent["query_type"] = "sales_report"
            # Default to inventory_analysis if it has "inventory" or "stock" in it
            elif any(term in query_lower for term in ["inventory", "stock", "on hand"]):
                intent["query_type"] = "inventory_analysis"
            # Default to general sales_report if we still don't have a type
            else:
                intent["query_type"] = "sales_report"

        # Ensure a reasonable limit is set for any ranking queries
        if intent["query_type"] == "ranking" and not intent["entities"]["limit"]:
            intent["entities"]["limit"] = 10  # Default limit

        return intent

    def _enhance_entity_extraction(self, query, intent):
        """Enhanced entity extraction with better text processing"""
        if not intent:  # Guard against None value
            intent = {
                "original_query": query,
                "query_type": "unknown",
                "entities": {
                    "time_period": None,
                    "products": [],
                    "categories": [],
                    "locations": [],
                    "metrics": [],
                    "payment_methods": [],
                    "vendors": [],
                    "limit": None,
                    "filters": {},
                    "grouping": []
                },
                "analysis_type": "summary",
                "comparison": None
            }
            
        # Add additional pattern matching for common entity formats
        # Look for location/warehouse names
        warehouse_patterns = [
            r'(?:in|at|for|from)\s+(?:the\s+)?(?:warehouse|store|location)\s+(?:named|called)?\s+[\'"]?([A-Za-z\s\-]+)[\'"]?',
            r'(?:warehouse|store|location)\s+[\'"]?([A-Za-z\s\-]+)[\'"]?'
        ]
        
        for pattern in warehouse_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                location = match.group(1).strip()
                if location and location not in intent["entities"]["locations"]:
                    intent["entities"]["locations"].append(location)
    
        # Look for category names
        category_patterns = [
            r'(?:in|for|from)\s+(?:the\s+)?(?:category|categories)\s+[\'"]?([A-Za-z\s\-]+)[\'"]?',
            r'(?:category|categories)\s+[\'"]?([A-Za-z\s\-]+)[\'"]?'
        ]
        
        for pattern in category_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                category = match.group(1).strip()
                if category and category not in intent["entities"]["categories"]:
                    intent["entities"]["categories"].append(category)
        
        # Look for vendor names
        vendor_patterns = [
            r'(?:by|from|for)\s+(?:the\s+)?(?:vendor|supplier|manufacturer)\s+[\'"]?([A-Za-z\s\-]+)[\'"]?',
            r'(?:vendor|supplier|manufacturer)\s+[\'"]?([A-Za-z\s\-]+)[\'"]?'
        ]
        
        for pattern in vendor_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                vendor = match.group(1).strip()
                if vendor and vendor not in intent["entities"]["vendors"]:
                    intent["entities"]["vendors"].append(vendor)
        
        # Look for time periods
        time_patterns = [
            r'(?:in|for|during|from)\s+(?:the\s+)?(?:year|month)\s+(\d{4})',
            r'(?:in|for|during|from)\s+(\d{4})',
            r'(\d{4})\s+(?:to|through|until|-)\s+(\d{4})'
        ]
        
        for pattern in time_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 1:  # Range of years
                    time_period = f"{match.group(1)}-{match.group(2)}"
                else:  # Single year
                    time_period = match.group(1)
                
                if time_period:
                    intent["entities"]["time_period"] = time_period
                    
        entity_prompt = f"""
    Examine this retail database query and extract the following specific entities:
    1. Location/store names (exact names like "RedHill", etc.)
    2. Product categories (like "AC", etc.)
    3. Time periods with exact details (specific years, months, date ranges)
    4. Product names or descriptors
    5. Vendor/supplier names

    For each entity type, return ALL possible mentions, including partial words that could be names.
    Format your response as JSON with arrays for each entity type.

    Query: "{query}"
    """
        
        entity_response = query_groq_api(entity_prompt, self.api_key, self.model)
        
        if entity_response:
            try:
                # Try to extract JSON from the response
                json_match = re.search(r"```json\s*(.*?)\s*```", entity_response, re.DOTALL)
                if json_match:
                    entity_response = json_match.group(1)
                
                # Try alternate formats if the LLM didn't wrap in code blocks
                if not json_match:
                    # Try to extract just a JSON object
                    json_match = re.search(r"\{.*\}", entity_response, re.DOTALL)
                    if json_match:
                        entity_response = json_match.group(0)
                
                # Parse the JSON response
                entities = json.loads(entity_response)
                
                # Update the intent with more detailed entities
                if "locations" in entities and entities["locations"]:
                    for location in entities["locations"]:
                        if location not in intent["entities"]["locations"]:
                            intent["entities"]["locations"].append(location)
                    
                if "categories" in entities and entities["categories"]:
                    for category in entities["categories"]:
                        if category not in intent["entities"]["categories"]:
                            intent["entities"]["categories"].append(category)
                    
                if "time_periods" in entities and entities["time_periods"]:
                    # Use the first time period if multiple are found
                    if isinstance(entities["time_periods"], list) and entities["time_periods"]:
                        intent["entities"]["time_period"] = entities["time_periods"][0]
                    else:
                        intent["entities"]["time_period"] = entities["time_periods"]
                    
                if "products" in entities and entities["products"]:
                    for product in entities["products"]:
                        if product not in intent["entities"]["products"]:
                            intent["entities"]["products"].append(product)
                
                if "vendors" in entities and entities["vendors"]:
                    for vendor in entities["vendors"]:
                        if vendor not in intent["entities"]["vendors"]:
                            intent["entities"]["vendors"].append(vendor)
                    
            except Exception as e:
                # If JSON parsing fails, continue with the extracted entities from regex
                pass
                
        return intent
        
def main():
    """Main application function"""
    # Main interface
    st.title("🛍️ DNS Chatbot")
    
    # Initialize session state variables if they don't exist
    if "chart_type" not in st.session_state:
        st.session_state.chart_type = "bar"
        
    if "current_data" not in st.session_state:
        st.session_state.current_data = None
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Initialize custom widget keys that may be used later
    for key in ["custom_x_axis", "custom_y_axis", "custom_hue"]:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Visualization Settings")
        
        # Chart type selector
        chart_types = [
            "bar", "line", "pie", "scatter", "histogram", 
            "heatmap"
        ]
        selected_chart = st.selectbox("Select Chart Type", 
                                    options=chart_types, 
                                    index=chart_types.index(st.session_state.chart_type))
        st.session_state.chart_type = selected_chart
        
        # Export options
        st.header("Export Options")
        
        if st.session_state.current_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export to Excel"):
                    excel_data = to_excel(st.session_state.current_data)
                    st.download_button(
                        label="Download Excel",
                        data=excel_data,
                        file_name="dns_data_export.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                if st.button("Export to PDF"):
                    # Get the latest assistant message for insights
                    latest_assistant_msg = next((msg for msg in reversed(st.session_state.messages) 
                                             if msg["role"] == "assistant"), None)
                    
                    analysis_text = latest_assistant_msg.get("content", "") if latest_assistant_msg else ""
                    
                    # Create figure for chart if available
                    chart_fig = None
                    if st.session_state.current_data is not None:
                        try:
                            import matplotlib.pyplot as plt
                            df = st.session_state.current_data
                            
                            # Try to determine best columns for visualization
                            x_col = next((col for col in df.columns if any(x in col.lower() for x in ['name', 'product', 'category', 'date', 'month'])), df.columns[0])
                            y_col = next((col for col in df.columns if any(x in col.lower() for x in ['sales', 'revenue', 'quantity', 'qty', 'total'])), 
                                         df.columns[1] if len(df.columns) > 1 else df.columns[0])
                            
                            # Create figure
                            chart_fig, ax = plt.subplots(figsize=(10, 6))
                            if st.session_state.chart_type == "bar":
                                if pd.api.types.is_numeric_dtype(df[y_col]):
                                    df.plot(kind='bar', x=x_col, y=y_col, ax=ax)
                            elif st.session_state.chart_type == "line":
                                if pd.api.types.is_numeric_dtype(df[y_col]):
                                    df.plot(kind='line', x=x_col, y=y_col, ax=ax)
                            elif st.session_state.chart_type == "pie" and len(df) <= 10:
                                if pd.api.types.is_numeric_dtype(df[y_col]):
                                    df.plot(kind='pie', y=y_col, labels=df[x_col], ax=ax)
                            elif st.session_state.chart_type == "scatter":
                                if pd.api.types.is_numeric_dtype(df[y_col]):
                                    df.plot(kind='scatter', x=x_col, y=y_col, ax=ax)
                            else:
                                # Default to bar chart
                                if pd.api.types.is_numeric_dtype(df[y_col]):
                                    df.plot(kind='bar', x=x_col, y=y_col, ax=ax)
                            plt.title(f"{y_col} by {x_col}")
                            plt.tight_layout()
                        except Exception as e:
                            st.warning(f"Could not create chart for PDF: {str(e)}")
                            chart_fig = None
                    
                    # Use the enhanced PDF export from DatabaseAgent
                    # If you've integrated DatabaseAgent, use its method
                    db_conn = get_db_connection()
                    agent = DatabaseAgent(db_conn)
                    pdf_data = agent.generate_enhanced_pdf(
                        st.session_state.current_data, 
                        analysis_text,
                        chart_fig
                    )
                    
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name="dns_data_export.pdf",
                        mime="application/pdf"
                    )
        
        # Clear chat option
        st.header("Chat Options")
        if st.button("Clear Chat History", type="primary"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.current_data = None
            st.rerun()

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
        # Display data visualization if available
        if "data" in message and message["data"] is not None:
            data_container = st.container()
            with data_container:
                # Always show the data table first
                st.subheader("Data Results")
                st.dataframe(message["data"], use_container_width=True)
                
                # Add interactive elements using Streamlit
                st.subheader("Interactive Chart Settings")
                cols = message["data"].columns
                
                # For numeric columns, allow user to select which to visualize
                numeric_cols = message["data"].select_dtypes(include=['number']).columns.tolist()
                categorical_cols = message["data"].select_dtypes(exclude=['number']).columns.tolist()
                
                if numeric_cols and categorical_cols:
                    col1, col2, col3 = st.columns(3)
                    
                    # Initialize default values
                    default_x = categorical_cols[0] if categorical_cols else None
                    default_y = numeric_cols[0] if numeric_cols else None
                    
                    # Initialize session state for these widgets with unique keys based on message ID
                    widget_id = f"widget_{id(message)}"
                    x_key = f"x_axis_{widget_id}"
                    y_key = f"y_axis_{widget_id}"
                    hue_key = f"hue_{widget_id}"
                    
                    # Make sure the keys exist in session state
                    if x_key not in st.session_state:
                        st.session_state[x_key] = default_x
                    if y_key not in st.session_state:
                        st.session_state[y_key] = default_y
                    if hue_key not in st.session_state:
                        st.session_state[hue_key] = "None"
                    
                    with col1:
                        selected_x = st.selectbox("X-axis", 
                                                options=categorical_cols, 
                                                index=categorical_cols.index(default_x) if default_x in categorical_cols else 0,
                                                key=x_key)
                    
                    with col2:
                        selected_y = st.selectbox("Y-axis", 
                                                options=numeric_cols, 
                                                index=numeric_cols.index(default_y) if default_y in numeric_cols else 0,
                                                key=y_key)
                    
                    with col3:
                        selected_hue = st.selectbox("Color by", 
                                                options=['None'] + categorical_cols, 
                                                index=0,
                                                key=hue_key)
                    
                    if selected_x and selected_y:
                        hue_col = None if selected_hue == 'None' else selected_hue
                        
                        # Create interactive plotly visualization
                        st.plotly_chart(
                            create_plotly_visualization(
                                message["data"], 
                                st.session_state.chart_type,
                                x_column=selected_x,
                                y_column=selected_y,
                                category_column=hue_col,
                                title="Custom Data Visualization"
                            ),
                            use_container_width=True
                        )
                else:
                    # Fallback to default visualization
                    if len(cols) >= 2:
                        # Try to determine best columns for visualization
                        x_col = next((col for col in cols if any(x in col.lower() for x in ["name", "product", "category", "date", "month", "period", "label"])), cols[0])
                        y_col = next((col for col in cols if any(x in col.lower() for x in ["sales", "revenue", "quantity", "qty", "total", "amount", "value", "count"])), cols[1])
                        
                        # Optional category column
                        cat_col = None
                        if len(cols) >= 3:
                            cat_col = next((col for col in cols if col not in [x_col, y_col] and any(x in col.lower() for x in ["category", "type", "class", "group", "status"])), None)
                        
                        # Create interactive plotly visualization instead of matplotlib
                        st.plotly_chart(
                            create_plotly_visualization(
                                message["data"], 
                                st.session_state.chart_type,
                                x_column=x_col,
                                y_column=y_col,
                                category_column=cat_col,
                                title="Data Visualization"
                            ),
                            use_container_width=True
                        )
                                
                # Different visualizations based on analysis type
                if "analysis_type" in message and message["analysis_type"] is not None:
                    if message["analysis_type"] == "abc_xyz":
                        if message["analysis_data"]:
                            st.subheader("ABC-XYZ Analysis")
                            
                            # Show ABC summary
                            st.write("ABC Classification Summary:")
                            st.dataframe(message["analysis_data"]["abc_summary"])
                            
                            # Show XYZ summary
                            st.write("XYZ Classification Summary:")
                            st.dataframe(message["analysis_data"]["xyz_summary"])
                            
                            # Show cross table
                            st.write("ABC-XYZ Cross Analysis:")
                            st.dataframe(message["analysis_data"]["cross_table"])
                            
                            # Visualization
                            if message["analysis_data"]["detailed"] is not None:
                                st.plotly_chart(
                                    create_plotly_visualization(
                                        message["analysis_data"]["detailed"], 
                                        "abc", 
                                        x_column=message["analysis_data"]["detailed"].columns[0],
                                        y_column=next((col for col in message["analysis_data"]["detailed"].columns if col not in ["abc_class", "xyz_class", "abc_xyz_class", "cumulative_percentage", "cumulative_value"]), None),
                                        title="ABC-XYZ Analysis"
                                    ),
                                    use_container_width=True
                                )
                                
                    elif message["analysis_type"] == "abc":
                        if message["analysis_data"]:
                            st.subheader("ABC Analysis")
                            
                            # Show ABC summary
                            st.write("ABC Classification Summary:")
                            st.dataframe(message["analysis_data"]["summary"])
                            
                            # Visualization
                            if message["analysis_data"]["detailed"] is not None:
                                st.plotly_chart(
                                    create_plotly_visualization(
                                        message["analysis_data"]["detailed"], 
                                        "abc", 
                                        x_column=message["analysis_data"]["detailed"].columns[0],
                                        y_column=next((col for col in message["analysis_data"]["detailed"].columns if col not in ["abc_class", "xyz_class", "abc_xyz_class", "cumulative_percentage", "cumulative_value"]), None),
                                        title="ABC Analysis"
                                    ),
                                    use_container_width=True
                                )
                                
                    elif message["analysis_type"] == "xyz":
                        if message["analysis_data"]:
                            st.subheader("XYZ Analysis")
                            
                            # Show XYZ summary
                            st.write("XYZ Classification Summary:")
                            st.dataframe(message["analysis_data"]["summary"])
                            
                            # Visualization
                            if message["analysis_data"]["detailed"] is not None:
                                st.plotly_chart(
                                    create_plotly_visualization(
                                        message["analysis_data"]["detailed"], 
                                        "xyz", 
                                        x_column="mean",
                                        y_column="cv",
                                        title="XYZ Analysis - Coefficient of Variation"
                                    ),
                                    use_container_width=True
                                )
                    
                    elif message["analysis_type"] == "forecast":
                        if message["analysis_data"]:
                            st.subheader("Forecasting Analysis")
                            
                            # Show forecasting results
                            st.write("Historical Data:")
                            st.dataframe(message["analysis_data"]["historical"])
                            
                            st.write("Forecast:")
                            st.dataframe(message["analysis_data"]["forecast"])
                            
                            # Visualization
                            forecast_data = {
                                'historical': message["analysis_data"]["historical"],
                                'forecast': message["analysis_data"]["forecast"]
                            }
                            st.plotly_chart(
                                create_plotly_visualization(
                                    forecast_data,
                                    "forecast",
                                    title="Time Series Forecast"
                                ),
                                use_container_width=True
                            )


        # Chat input
    prompt = st.chat_input("Ask a question about your database...")
    
    if prompt:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Process and display assistant response    
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with timeout_spinner(25, "Analyzing your data..."):
                # Get database connection
                db_conn = get_db_connection()
                agent = DatabaseAgent(db_conn)
                
                # Process the query
                result = agent.process_query(prompt, st.session_state.chat_history)
                
                # Display response
                message_placeholder.markdown(result["response"])
                
                # Only show data if available
                if result["data"] is not None:
                    # Display the data in a table first
                    # Display the data in a table first
                    st.subheader("Data Results")
                    st.dataframe(result["data"], use_container_width=True)
                    
                    # Different visualizations based on analysis type
                    if result["analysis_type"] is not None:
                        if result["analysis_type"] == "abc_xyz":
                            if result["analysis_data"]:
                                st.subheader("ABC-XYZ Analysis")
                                
                                # Show ABC summary
                                st.write("ABC Classification Summary:")
                                st.dataframe(result["analysis_data"]["abc_summary"])
                                
                                # Show XYZ summary
                                st.write("XYZ Classification Summary:")
                                st.dataframe(result["analysis_data"]["xyz_summary"])
                                
                                # Show cross table
                                st.write("ABC-XYZ Cross Analysis:")
                                st.dataframe(result["analysis_data"]["cross_table"])
                                
                                # Visualization
                                if result["analysis_data"]["detailed"] is not None:
                                    st.plotly_chart(
                                        create_plotly_visualization(
                                            result["analysis_data"]["detailed"], 
                                            "abc", 
                                            x_column=result["analysis_data"]["detailed"].columns[0],
                                            y_column=next((col for col in result["analysis_data"]["detailed"].columns if col not in ["abc_class", "xyz_class", "abc_xyz_class", "cumulative_percentage", "cumulative_value"]), None),
                                            title="ABC-XYZ Analysis"
                                        ),
                                        use_container_width=True
                                    )
                                    
                        elif result["analysis_type"] == "abc":
                            if result["analysis_data"]:
                                st.subheader("ABC Analysis")
                                
                                # Show ABC summary
                                st.write("ABC Classification Summary:")
                                st.dataframe(result["analysis_data"]["summary"])
                                
                                # Visualization
                                if result["analysis_data"]["detailed"] is not None:
                                    st.plotly_chart(
                                        create_plotly_visualization(
                                            result["analysis_data"]["detailed"], 
                                            "abc", 
                                            x_column=result["analysis_data"]["detailed"].columns[0],
                                            y_column=next((col for col in result["analysis_data"]["detailed"].columns if col not in ["abc_class", "xyz_class", "abc_xyz_class", "cumulative_percentage", "cumulative_value"]), None),
                                            title="ABC Analysis"
                                        ),
                                        use_container_width=True
                                    )
                                    
                        elif result["analysis_type"] == "xyz":
                            if result["analysis_data"]:
                                st.subheader("XYZ Analysis")
                                
                                # Show XYZ summary
                                st.write("XYZ Classification Summary:")
                                st.dataframe(result["analysis_data"]["summary"])
                                
                                # Visualization
                                if result["analysis_data"]["detailed"] is not None:
                                    st.plotly_chart(
                                        create_plotly_visualization(
                                            result["analysis_data"]["detailed"], 
                                            "xyz", 
                                            x_column="mean",
                                            y_column="cv",
                                            title="XYZ Analysis - Coefficient of Variation"
                                        ),
                                        use_container_width=True
                                    )
                        
                        elif result["analysis_type"] == "forecast":
                            if result["analysis_data"]:
                                st.subheader("Forecasting Analysis")
                                
                                # Show forecasting results
                                st.write("Historical Data:")
                                st.dataframe(result["analysis_data"]["historical"])
                                
                                st.write("Forecast:")
                                st.dataframe(result["analysis_data"]["forecast"])
                                
                                # Visualization
                                forecast_data = {
                                    'historical': result["analysis_data"]["historical"],
                                    'forecast': result["analysis_data"]["forecast"]
                                }
                                st.plotly_chart(
                                    create_plotly_visualization(
                                        forecast_data,
                                        "forecast",
                                        title="Time Series Forecast"
                                    ),
                                    use_container_width=True
                                )
                    else:
                        # Regular data visualization
                        cols = result["data"].columns
                        
                        if len(cols) >= 2:
                            # Try to determine best columns for visualization
                            x_col = next((col for col in cols if any(x in col.lower() for x in ["name", "product", "category", "date", "month", "period", "label"])), cols[0])
                            y_col = next((col for col in cols if any(x in col.lower() for x in ["sales", "revenue", "quantity", "qty", "total", "amount", "value", "count"])), cols[1])
                            
                            # Optional category column
                            cat_col = None
                            if len(cols) >= 3:
                                cat_col = next((col for col in cols if col not in [x_col, y_col] and any(x in col.lower() for x in ["category", "type", "class", "group", "status"])), None)
                            
                            # Create visualization
                            st.plotly_chart(
                                create_plotly_visualization(
                                    result["data"], 
                                    st.session_state.chart_type,
                                    x_column=x_col,
                                    y_column=y_col,
                                    category_column=cat_col,
                                    title=f"Data Visualization"
                                ),
                                use_container_width=True
                            )
                            
                            # Add interactive elements to adjust visualization
                            st.subheader("Customize Visualization")
                            
                            # For numeric columns, allow user to select which to visualize
                            numeric_cols = result["data"].select_dtypes(include=['number']).columns.tolist()
                            categorical_cols = result["data"].select_dtypes(exclude=['number']).columns.tolist()
                            
                            if numeric_cols and categorical_cols:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    custom_x = st.selectbox("X-axis", 
                                                        options=categorical_cols, 
                                                        index=categorical_cols.index(x_col) if x_col in categorical_cols else 0,
                                                        key="custom_x_axis")
                                
                                with col2:
                                    custom_y = st.selectbox("Y-axis", 
                                                        options=numeric_cols, 
                                                        index=numeric_cols.index(y_col) if y_col in numeric_cols else 0,
                                                        key="custom_y_axis")
                                
                                with col3:
                                    custom_hue = st.selectbox("Color by", 
                                                          options=['None'] + categorical_cols, 
                                                          index=0 if cat_col is None else categorical_cols.index(cat_col)+1,
                                                          key="custom_hue")
                                
                                if st.button("Update Visualization"):
                                    hue_col = None if custom_hue == 'None' else custom_hue
                                    
                                    st.plotly_chart(
                                        create_plotly_visualization(
                                            result["data"], 
                                            st.session_state.chart_type,
                                            x_column=custom_x,
                                            y_column=custom_y,
                                            category_column=hue_col,
                                            title="Custom Data Visualization"
                                        ),
                                        use_container_width=True
                                    )
        
        # Store assistant message in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "sql": result["sql"],
            "data": result["data"],
            "analysis_type": result["analysis_type"],
            "analysis_data": result["analysis_data"]
        })
        
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["response"]
        })

# Run the application
if __name__ == "__main__":
    main()