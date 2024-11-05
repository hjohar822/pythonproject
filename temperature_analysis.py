import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
from scipy.stats import f_oneway

def load_and_clean_temperature_data():
    """Load and clean the temperature impact dataset"""
    
    # Load data
    df = pd.read_csv('ev_charging_patterns.csv')
    
    # Convert time columns to datetime
    df['charging_start_time'] = pd.to_datetime(df['Charging Start Time'])
    df['charging_end_time'] = pd.to_datetime(df['Charging End Time'])
    
    # Extract temperature and efficiency data
    df['ambient_temp'] = df['Temperature (°C)']
    df['energy_efficiency'] = df['Energy Consumed (kWh)'] / df['Distance Driven (since last charge) (km)']
    
    # Remove temperature outliers above 40°C
    print(f"\nTemperature Outlier Removal Statistics:")
    original_count = len(df)
    df = df[df['ambient_temp'] <= 40]
    removed_temp = original_count - len(df)
    print(f"Removed {removed_temp} records with temperature above 40°C")
    
    # Remove energy efficiency outliers using IQR method
    Q1 = df['energy_efficiency'].quantile(0.25)
    Q3 = df['energy_efficiency'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    original_count = len(df)
    df = df[(df['energy_efficiency'] >= lower_bound) & (df['energy_efficiency'] <= upper_bound)]
    removed_efficiency = original_count - len(df)
    print(f"\nEnergy Efficiency Outlier Removal Statistics:")
    print(f"Removed {removed_efficiency} records outside the range: {lower_bound:.3f} to {upper_bound:.3f} kWh/km")
    
    # Create temperature buckets
    df['temp_range'] = pd.cut(
        df['ambient_temp'],
        bins=[-float('inf'), 0, 10, 20, 30, 40],
        labels=['Below 0°C', '0-10°C', '10-20°C', '20-30°C', '30-40°C']
    )
    
    return df

def create_temperature_visualizations(df):
    """Create visualizations for temperature impact analysis"""
    
    # Scatter plot of temperature vs energy efficiency
    fig_scatter = px.scatter(
        df,
        x='ambient_temp',
        y='energy_efficiency',
        title='Temperature vs Energy Efficiency',
        labels={
            'ambient_temp': 'Temperature (°C)',
            'energy_efficiency': 'Energy Efficiency (kWh/km)'
        }
    )
    
    # Box plot of efficiency by temperature range
    fig_box = px.box(
        df,
        x='temp_range',
        y='energy_efficiency',
        title='Energy Efficiency by Temperature Range',
        labels={
            'temp_range': 'Temperature Range',
            'energy_efficiency': 'Energy Efficiency (kWh/km)'
        }
    )
    
    # Line plot of average efficiency by temperature range
    avg_efficiency = df.groupby('temp_range', observed=True)['energy_efficiency'].mean().reset_index()
    
    fig_line = px.line(
        avg_efficiency,
        x='temp_range',
        y='energy_efficiency',
        title='Average Energy Efficiency by Temperature Range',
        labels={
            'temp_range': 'Temperature Range',
            'energy_efficiency': 'Average Energy Efficiency (kWh/km)'
        }
    )
    
    # Update line plot to use markers
    fig_line.update_traces(mode='lines+markers')
    
    return {
        'scatter': fig_scatter,
        'box': fig_box,
        'line': fig_line
    }

def analyze_temperature_impact(df):
    """Analyze the impact of temperature on charging efficiency"""
    
    def get_stats(data):
        """Helper function to calculate all statistics at once"""
        return pd.Series({
            'Count': len(data),
            'Mean': data.mean(),
            'Std Dev': data.std(),
            'Min': data.min(),
            'Q1': data.quantile(0.25),
            'Median': data.median(),
            'Q3': data.quantile(0.75),
            'Max': data.max()
        }).round(3)
    
    # Calculate correlations
    correlation = df['ambient_temp'].corr(df['energy_efficiency'])
    
    # Calculate average efficiency by temperature range
    temp_efficiency = df.groupby('temp_range', observed=True)['energy_efficiency'].mean()
    
    # Calculate descriptive statistics using the helper function
    temp_stats = get_stats(df['ambient_temp'])
    efficiency_stats = get_stats(df['energy_efficiency'])
    
    # Calculate temperature ranges
    temp_ranges = {
        'min_temp': df['ambient_temp'].min(),
        'max_temp': df['ambient_temp'].max(),
        'avg_temp': df['ambient_temp'].mean()
    }
    
    # Perform ANOVA test between temperature ranges
    ranges = df['temp_range'].unique()
    if len(ranges) >= 2:
        groups = [group['energy_efficiency'].values for name, group in df.groupby('temp_range', observed=True)]
        f_stat, p_value = f_oneway(*groups)
    else:
        f_stat = float('nan')
        p_value = float('nan')
        print(f"\nWarning: Insufficient temperature ranges for comparison")
    
    return {
        'correlation': correlation,
        'temp_efficiency': temp_efficiency,
        'temperature_ranges': temp_ranges,
        'f_statistic': f_stat,
        'p_value': p_value,
        'descriptive_stats': {
            'temperature': temp_stats,
            'efficiency': efficiency_stats
        }
    }

def generate_insights(analysis_results):
    """Generate text insights from the analysis results"""
    
    # Find the temperature range with best efficiency
    best_temp_range = analysis_results['temp_efficiency'].idxmin()
    
    insights = {
        'correlation': f"Temperature correlation with energy consumption: {analysis_results['correlation']:.2f}",
        'temp_range_impact': f"Most efficient temperature range: {best_temp_range}",
        'temperature_range': (
            f"Temperature range: {analysis_results['temperature_ranges']['min_temp']:.1f}°C to "
            f"{analysis_results['temperature_ranges']['max_temp']:.1f}°C"
        ),
        'statistical_significance': (
            "Temperature impact is statistically significant" 
            if analysis_results['p_value'] < 0.05 
            else "Temperature impact is not statistically significant"
        )
    }
    
    return insights

if __name__ == "__main__":
    # Load and prepare data
    df = load_and_clean_temperature_data()
    
    # Perform analysis
    analysis_results = analyze_temperature_impact(df)
    
    # Create visualizations
    visualizations = create_temperature_visualizations(df)
    
    # Generate insights
    insights = generate_insights(analysis_results)
    
    print("Analysis complete. Results stored in analysis_results and visualizations dictionaries.") 