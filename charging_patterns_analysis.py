import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def analyze_charging_patterns():
    """Analyze charging patterns for different user types"""
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Perform analyses
    pattern_analysis = analyze_charging_behavior(df)
    statistical_tests = perform_statistical_tests(df)
    visualizations = create_pattern_visualizations(df)
    descriptive_stats = calculate_descriptive_stats(df)  # Add this line
    
    return {
        'pattern_analysis': pattern_analysis,
        'statistical_tests': statistical_tests,
        'visualizations': visualizations,
        'descriptive_stats': descriptive_stats,  # Add this line
        'raw_data': df
    }

def load_and_prepare_data():
    """Load and prepare the dataset for analysis"""
    # Load data
    df = pd.read_csv('ev_charging_patterns.csv')
    
    # Convert time columns to datetime
    df['Charging Start Time'] = pd.to_datetime(df['Charging Start Time'])
    df['Charging End Time'] = pd.to_datetime(df['Charging End Time'])
    
    # Calculate Percentage Charged
    df['Percentage Charged'] = df['State of Charge (End %)'] - df['State of Charge (Start %)']
    
    # Remove invalid percentage charged values (below 0 or above 100)
    df = df[(df['Percentage Charged'] >= 0) & (df['Percentage Charged'] <= 100)]
    
    # Print outlier removal statistics
    print("\nOutlier Removal Statistics:")
    print(f"Removed {len(df[df['Percentage Charged'] < 0])} negative percentage values")
    print(f"Removed {len(df[df['Percentage Charged'] > 100])} percentage values above 100")
    
    # Extract time-related features
    df['Hour'] = df['Charging Start Time'].dt.hour
    df['Day of Week'] = df['Charging Start Time'].dt.day_name()
    
    # Create Time of Day categories
    df['Time of Day'] = pd.cut(df['Hour'], 
                              bins=[-np.inf, 6, 12, 18, np.inf],
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    return df

def analyze_charging_behavior(df):
    """Analyze charging patterns by user type, day, and time"""
    
    # Basic patterns by user type
    user_patterns = df.groupby('User Type').agg({
        'Percentage Charged': ['mean', 'median', 'std'],
        'Charging Duration (hours)': 'mean',
        'Energy Consumed (kWh)': 'mean'
    }).round(2)
    
    # Patterns by day of week
    day_patterns = df.groupby(['User Type', 'Day of Week'], observed=True)['Percentage Charged'].agg([
        'mean', 'median', 'count'
    ]).round(2)
    
    # Patterns by time of day
    time_patterns = df.groupby(['User Type', 'Time of Day'], observed=True)['Percentage Charged'].agg([
        'mean', 'median', 'count'
    ]).round(2)
    
    return {
        'user_patterns': user_patterns,
        'day_patterns': day_patterns,
        'time_patterns': time_patterns
    }

def perform_statistical_tests(df):
    """Perform statistical tests on charging patterns"""
    
    # Create Python-friendly column names
    df = df.copy()  # Create a copy to avoid modifying the original
    df['percentage_charged'] = df['Percentage Charged']
    df['day_of_week'] = df['Day of Week']
    df['time_of_day'] = df['Time of Day']
    df['user_type'] = df['User Type']
    
    # Two-way ANOVA for User Type and Day of Week
    formula = 'percentage_charged ~ C(user_type) + C(day_of_week) + C(user_type):C(day_of_week)'
    model = sm.OLS.from_formula(formula, data=df).fit()
    day_anova = sm.stats.anova_lm(model, typ=2)
    
    # Two-way ANOVA for User Type and Time of Day
    formula_time = 'percentage_charged ~ C(user_type) + C(time_of_day) + C(user_type):C(time_of_day)'
    model_time = sm.OLS.from_formula(formula_time, data=df).fit()
    time_anova = sm.stats.anova_lm(model_time, typ=2)
    
    # Chi-square test for Time of Day preferences
    contingency = pd.crosstab(df['User Type'], df['Time of Day'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    # T-tests for Percentage Charged between user types
    user_types = df['User Type'].unique()
    t_stat, t_p_value = stats.ttest_ind(
        df[df['User Type'] == user_types[0]]['percentage_charged'],
        df[df['User Type'] == user_types[1]]['percentage_charged']
    )
    
    return {
        'day_anova': day_anova,
        'time_anova': time_anova,
        'chi_square': {'statistic': chi2, 'p_value': p_value},
        't_test': {'statistic': t_stat, 'p_value': t_p_value}
    }

def create_pattern_visualizations(df):
    """Create visualizations for charging patterns"""
    
    # Box plot of Percentage Charged by User Type and Day of Week
    day_box = px.box(df, 
                     x='Day of Week',
                     y='Percentage Charged',
                     color='User Type',
                     title='Charging Patterns by Day of Week')
    
    # Box plot of Percentage Charged by User Type and Time of Day
    time_box = px.box(df,
                      x='Time of Day',
                      y='Percentage Charged',
                      color='User Type',
                      title='Charging Patterns by Time of Day')
    
    # Heat map of average Percentage Charged
    pivot_data = df.pivot_table(
        values='Percentage Charged',
        index='Time of Day',
        columns='Day of Week',
        aggfunc='mean'
    )
    
    heatmap = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdBu_r'
    ))
    heatmap.update_layout(
        title='Average Percentage Charged by Day and Time',
        xaxis_title='Day of Week',
        yaxis_title='Time of Day'
    )
    
    # State of charge patterns
    soc_plot = go.Figure()
    for user_type in df['User Type'].unique():
        user_data = df[df['User Type'] == user_type]
        soc_plot.add_trace(go.Box(
            y=user_data['Percentage Charged'],
            name=user_type,
            boxmean=True
        ))
    soc_plot.update_layout(title='Charging Percentage by User Type')
    
    return {
        'day_box': day_box,
        'time_box': time_box,
        'heatmap': heatmap,
        'soc_plot': soc_plot
    }

def calculate_descriptive_stats(df):
    """Calculate descriptive statistics for charging percentage"""
    
    def get_stats(data):
        """Helper function to calculate all statistics at once"""
        return pd.Series({
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'Q1': data.quantile(0.25),
            'median': data.median(),
            'Q3': data.quantile(0.75),
            'max': data.max()
        }).round(2)
    
    # Overall descriptive statistics
    overall_stats = get_stats(df['Percentage Charged'])
    
    # Descriptive statistics by user type
    user_type_stats = df.groupby('User Type')['Percentage Charged'].apply(get_stats).unstack()
    
    return {
        'overall': overall_stats,
        'by_user_type': user_type_stats
    }

if __name__ == "__main__":
    results = analyze_charging_patterns()
    print("\nAnalysis completed successfully")
    
    print("\nCharging Pattern Analysis:")
    print("\nUser Type Patterns:")
    print(results['pattern_analysis']['user_patterns'])
    
    print("\nStatistical Test Results:")
    print("\nTwo-way ANOVA (Day of Week):")
    print(results['statistical_tests']['day_anova'])
    print("\nTwo-way ANOVA (Time of Day):")
    print(results['statistical_tests']['time_anova'])
    print("\nChi-square test p-value:", results['statistical_tests']['chi_square']['p_value'])
    print("\nT-test p-value:", results['statistical_tests']['t_test']['p_value'])