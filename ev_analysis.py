import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import os

def load_and_prepare_data():
    """Load and prepare the dataset for analysis"""
    # Read the data
    df = pd.read_csv('ev_charging_patterns.csv')
    
    # Convert "Charging Start Time" to datetime
    df['Charging Start Time'] = pd.to_datetime(df['Charging Start Time'])
    df['Charging End Time'] = pd.to_datetime(df['Charging End Time'])
    
    # Calculate cost efficiency (USD per kWh)
    df['cost_efficiency'] = df['Charging Cost (USD)'] / df['Energy Consumed (kWh)']
    
    # Remove outliers for cost efficiency using IQR method
    Q1_cost = df['cost_efficiency'].quantile(0.25)
    Q3_cost = df['cost_efficiency'].quantile(0.75)
    IQR_cost = Q3_cost - Q1_cost
    lower_bound_cost = Q1_cost - 1.5 * IQR_cost
    upper_bound_cost = Q3_cost + 1.5 * IQR_cost
    
    # Remove outliers for battery capacity using IQR method
    Q1_battery = df['Battery Capacity (kWh)'].quantile(0.25)
    Q3_battery = df['Battery Capacity (kWh)'].quantile(0.75)
    IQR_battery = Q3_battery - Q1_battery
    lower_bound_battery = Q1_battery - 1.5 * IQR_battery
    upper_bound_battery = Q3_battery + 1.5 * IQR_battery
    
    # Filter out outliers for both variables
    df = df[
        (df['cost_efficiency'] >= lower_bound_cost) & 
        (df['cost_efficiency'] <= upper_bound_cost) &
        (df['Battery Capacity (kWh)'] >= lower_bound_battery) &
        (df['Battery Capacity (kWh)'] <= upper_bound_battery)
    ]
    
    # Print outlier removal statistics
    print("\nOutlier Removal Statistics:")
    print(f"Cost Efficiency - Removed {len(df[df['cost_efficiency'] > upper_bound_cost])} upper outliers")
    print(f"Cost Efficiency - Removed {len(df[df['cost_efficiency'] < lower_bound_cost])} lower outliers")
    print(f"Battery Capacity - Removed {len(df[df['Battery Capacity (kWh)'] > upper_bound_battery])} upper outliers")
    print(f"Battery Capacity - Removed {len(df[df['Battery Capacity (kWh)'] < lower_bound_battery])} lower outliers")
    
    # Handle any missing values
    numeric_cols = ['Vehicle Age (years)', 'Battery Capacity (kWh)', 
                   'Charging Cost (USD)', 'Energy Consumed (kWh)', 
                   'cost_efficiency']
    df = df.dropna(subset=numeric_cols)
    
    return df

def analyze_age_efficiency_relationship(df):
    """Analyze the relationship between vehicle age and charging cost efficiency"""
    
    # 1. Basic correlation analysis
    correlation = df['Vehicle Age (years)'].corr(df['cost_efficiency'])
    
    # 2. Model-specific analysis
    model_correlations = df.groupby('Vehicle Model', observed=True).apply(
        lambda x: x['Vehicle Age (years)'].corr(x['cost_efficiency'])
    ).reset_index()
    model_correlations.columns = ['Vehicle Model', 'age_efficiency_correlation']
    
    # 3. Control for battery capacity
    df['battery_capacity_group'] = pd.qcut(df['Battery Capacity (kWh)'], q=4, 
                                         labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    # Calculate statistics for each group
    grouped_stats = df.groupby(['Vehicle Model', 'battery_capacity_group'], observed=True).agg({
        'Vehicle Age (years)': 'mean',
        'cost_efficiency': ['mean', 'std', 'count'],
        'Battery Capacity (kWh)': 'mean'
    }).round(3)
    
    return correlation, model_correlations, grouped_stats

def perform_statistical_tests(df):
    """Perform statistical tests to validate relationships"""
    
    # 1. Linear regression
    X = df[['Vehicle Age (years)', 'Battery Capacity (kWh)']]
    y = df['cost_efficiency']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    # 2. ANOVA test for vehicle model differences
    model_groups = [group for _, group in df.groupby('Vehicle Model')['cost_efficiency']]
    f_statistic, p_value = stats.f_oneway(*model_groups)
    
    return model.summary(), f_statistic, p_value

def calculate_descriptive_stats(df):
    """Calculate descriptive statistics for vehicle data"""
    
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
    
    # Calculate overall statistics for each column
    overall_stats = pd.DataFrame({
        'Vehicle Age (years)': get_stats(df['Vehicle Age (years)']),
        'Battery Capacity (kWh)': get_stats(df['Battery Capacity (kWh)']),
        'Cost Efficiency (USD/kWh)': get_stats(df['cost_efficiency'])
    })
    
    # Calculate statistics by vehicle model
    model_stats = pd.DataFrame({
        'Vehicle Model': [],
        'Vehicle Age (years)': [],
        'Battery Capacity (kWh)': [],
        'Cost Efficiency (USD/kWh)': []
    })
    
    for model in df['Vehicle Model'].unique():
        model_data = df[df['Vehicle Model'] == model]
        model_row = {
            'Vehicle Model': model,
            'Vehicle Age (years)': get_stats(model_data['Vehicle Age (years)']).mean(),
            'Battery Capacity (kWh)': get_stats(model_data['Battery Capacity (kWh)']).mean(),
            'Cost Efficiency (USD/kWh)': get_stats(model_data['cost_efficiency']).mean()
        }
        model_stats = pd.concat([model_stats, pd.DataFrame([model_row])], ignore_index=True)
    
    return {
        'overall': overall_stats,
        'by_model': model_stats
    }

def create_visualizations(df):
    """Create visualizations for the analysis"""
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # 1. Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Vehicle Age (years)', y='cost_efficiency', 
                    hue='Vehicle Model', alpha=0.6)
    sns.regplot(data=df, x='Vehicle Age (years)', y='cost_efficiency', 
                scatter=False, color='black', line_kws={'linestyle': '--'})
    plt.title('Vehicle Age vs Charging Cost Efficiency')
    plt.xlabel('Vehicle Age (years)')
    plt.ylabel('Cost Efficiency (USD/kWh)')
    plt.savefig('static/age_efficiency_scatter.png')
    plt.close()
    
    # 2. Box plot by vehicle model
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Vehicle Model', y='cost_efficiency')
    plt.title('Cost Efficiency Distribution by Vehicle Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/model_efficiency_box.png')
    plt.close()
    
    # 3. Heat map for battery capacity influence
    pivot_data = df.pivot_table(
        values='cost_efficiency',
        index=pd.qcut(df['Vehicle Age (years)'], q=5),
        columns=pd.qcut(df['Battery Capacity (kWh)'], q=5),
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Cost Efficiency by Age and Battery Capacity')
    plt.xlabel('Battery Capacity Quintiles')
    plt.ylabel('Vehicle Age Quintiles')
    plt.tight_layout()
    plt.savefig('static/efficiency_heatmap.png')
    plt.close()

if __name__ == "__main__":
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Perform analyses
    correlation, model_correlations, grouped_stats = analyze_age_efficiency_relationship(df)
    desc_stats = calculate_descriptive_stats(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Perform statistical tests
    regression_summary, f_statistic, p_value = perform_statistical_tests(df)
    
    # Print results
    print("\nDescriptive Statistics:")
    print(desc_stats)
    print("\nOverall Correlation between Vehicle Age and Cost Efficiency:", correlation)
    print("\nCorrelation by Vehicle Model:")
    print(model_correlations)
    print("\nGrouped Statistics:")
    print(grouped_stats)
    print("\nANOVA Test Results:")
    print(f"F-statistic: {f_statistic}")
    print(f"p-value: {p_value}")
    print("\nRegression Summary:")
    print(regression_summary) 