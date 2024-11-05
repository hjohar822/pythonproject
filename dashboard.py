import dash
from dash import dcc, html, dash_table
import plotly.express as px
import pandas as pd
from ev_analysis import (
    load_and_prepare_data, 
    analyze_age_efficiency_relationship, 
    perform_statistical_tests,
    calculate_descriptive_stats
)
from charging_patterns_analysis import analyze_charging_patterns
from temperature_analysis import (
    load_and_clean_temperature_data,
    analyze_temperature_impact,
    create_temperature_visualizations,
    generate_insights
)

# Initialize the Dash app
app = dash.Dash(__name__)

# Load and prepare the data for both analyses
df_efficiency = load_and_prepare_data()
desc_stats = calculate_descriptive_stats(df_efficiency)
correlation, model_correlations, grouped_stats = analyze_age_efficiency_relationship(df_efficiency)
regression_summary, f_statistic, p_value = perform_statistical_tests(df_efficiency)

# Load charging patterns analysis results
charging_results = analyze_charging_patterns()

# Load temperature analysis results
df_temperature = load_and_clean_temperature_data()
temp_analysis = analyze_temperature_impact(df_temperature)
temp_visualizations = create_temperature_visualizations(df_temperature)
temp_insights = generate_insights(temp_analysis)

# Layout with research questions in separate tabs
app.layout = html.Div([
    # Title
    html.H1("EV Charging Analysis Dashboard",
            style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2c3e50'}),
    
    # Main Research Questions Tabs
    dcc.Tabs([
        # Research Question 1: Cost Efficiency Analysis
        dcc.Tab(label='Cost Efficiency Analysis', children=[
            # Key Findings Section
            html.Div([
                html.H3("Key Findings", 
                        style={'marginBottom': 20, 'color': '#34495e'}),
                html.Div([
                    html.P([
                        "Overall correlation between vehicle age and cost efficiency: ",
                        html.Strong(f"{correlation:.3f}")
                    ]),
                    html.P([
                        "Average Cost Efficiency: ",
                        html.Strong(f"{df_efficiency['cost_efficiency'].mean():.3f} USD/kWh")
                    ]),
                    html.P([
                        "Most Efficient Model: ",
                        html.Strong(f"{df_efficiency.groupby('Vehicle Model')['cost_efficiency'].mean().idxmin()}")
                    ])
                ], style={'marginBottom': 30, 'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px'})
            ]),
            
            # Subtabs for Cost Efficiency Analysis
            dcc.Tabs([
                # Tab 1: Age vs Efficiency
                dcc.Tab(label='Age vs Efficiency', children=[
                    dcc.Graph(
                        figure=px.scatter(df_efficiency, 
                                        x='Vehicle Age (years)',
                                        y='cost_efficiency',
                                        color='Vehicle Model',
                                        trendline='ols',
                                        title='Vehicle Age vs Cost Efficiency')
                    )
                ]),
                
                # Tab 2: Model Comparison
                dcc.Tab(label='Model Comparison', children=[
                    dcc.Graph(
                        figure=px.box(df_efficiency,
                                    x='Vehicle Model',
                                    y='cost_efficiency',
                                    title='Cost Efficiency by Vehicle Model')
                    )
                ]),
                
                # Tab 3: Battery Impact
                dcc.Tab(label='Battery Impact', children=[
                    dcc.Graph(
                        figure=px.scatter(df_efficiency,
                                        x='Battery Capacity (kWh)',
                                        y='cost_efficiency',
                                        color='Vehicle Model',
                                        size='Vehicle Age (years)',
                                        title='Battery Capacity vs Cost Efficiency')
                    )
                ]),
                
                # Tab 4: Statistical Summary
                dcc.Tab(label='Statistical Summary', children=[
                    html.Div([
                        # Overall Descriptive Statistics
                        html.H4("Overall Descriptive Statistics", style={'marginTop': 20}),
                        dash_table.DataTable(
                            data=desc_stats['overall'].reset_index().to_dict('records'),
                            columns=[
                                {'name': 'Statistic', 'id': 'index'},
                                {'name': 'Vehicle Age (years)', 'id': 'Vehicle Age (years)'},
                                {'name': 'Battery Capacity (kWh)', 'id': 'Battery Capacity (kWh)'},
                                {'name': 'Cost Efficiency (USD/kWh)', 'id': 'Cost Efficiency (USD/kWh)'}
                            ],
                            style_cell={
                                'textAlign': 'left',
                                'padding': '10px',
                                'backgroundColor': 'white'
                            },
                            style_header={
                                'backgroundColor': '#f8f9fa',
                                'fontWeight': 'bold',
                                'color': '#2c3e50'
                            },
                            style_table={
                                'borderRadius': '5px',
                                'overflow': 'hidden',
                                'border': '1px solid #e0e0e0'
                            }
                        ),
                        
                        # Statistics by Vehicle Model
                        html.H4("Statistics by Vehicle Model", style={'marginTop': 40}),
                        dash_table.DataTable(
                            data=desc_stats['by_model'].reset_index().to_dict('records'),
                            columns=[
                                {'name': 'Vehicle Model', 'id': 'Vehicle Model'},
                                {'name': 'Vehicle Age (years)', 'id': 'Vehicle Age (years)'},
                                {'name': 'Battery Capacity (kWh)', 'id': 'Battery Capacity (kWh)'},
                                {'name': 'Cost Efficiency (USD/kWh)', 'id': 'Cost Efficiency (USD/kWh)'}
                            ],
                            style_cell={
                                'textAlign': 'left',
                                'padding': '10px',
                                'backgroundColor': 'white'
                            },
                            style_header={
                                'backgroundColor': '#f8f9fa',
                                'fontWeight': 'bold',
                                'color': '#2c3e50'
                            },
                            style_table={
                                'borderRadius': '5px',
                                'overflow': 'hidden',
                                'border': '1px solid #e0e0e0'
                            }
                        ),
                        
                        # Statistical Tests Results
                        html.H4("Statistical Tests", style={'marginTop': 40}),
                        html.Div([
                            html.P([
                                "F-statistic: ",
                                html.Strong(f"{f_statistic:.3f}"),
                                ", p-value: ",
                                html.Strong(f"{p_value:.3f}")
                            ])
                        ], style={
                            'backgroundColor': '#f8f9fa',
                            'padding': '15px',
                            'borderRadius': '5px',
                            'marginTop': '10px'
                        })
                    ], style={'padding': '20px'})
                ])
            ])
        ]),
        
        # Research Question 2: Charging Patterns Analysis
        dcc.Tab(label='Charging Patterns Analysis', children=[
            # Key Findings for Charging Patterns
            html.Div([
                html.H3("Charging Pattern Insights", 
                        style={'marginBottom': 20, 'color': '#34495e'}),
                html.Div([
                    html.P([
                        "Average Percentage Charged: ",
                        html.Strong(f"{charging_results['descriptive_stats']['overall']['mean']:.1f}%")
                    ]),
                ], style={'marginBottom': 30, 'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px'}),
                
                # Subtabs for Charging Patterns
                dcc.Tabs([
                    # Tab 1: Daily Patterns
                    dcc.Tab(label='Daily Patterns', children=[
                        dcc.Graph(figure=charging_results['visualizations']['day_box']),
                        dcc.Graph(figure=charging_results['visualizations']['heatmap'])
                    ]),
                    
                    # Tab 2: Time of Day Analysis
                    dcc.Tab(label='Time of Day Analysis', children=[
                        dcc.Graph(figure=charging_results['visualizations']['time_box']),
                        dcc.Graph(figure=charging_results['visualizations']['soc_plot'])
                    ]),
                    
                    # Tab 3: Statistical Analysis
                    dcc.Tab(label='Statistical Analysis', children=[
                        html.Div([
                            # Descriptive Statistics Section
                            html.H4("Descriptive Statistics", style={'marginTop': 20}),
                            html.Div([
                                html.H5("Overall Charging Statistics"),
                                dash_table.DataTable(
                                    data=[charging_results['descriptive_stats']['overall'].to_dict()],
                                    columns=[{'name': i, 'id': i} for i in charging_results['descriptive_stats']['overall'].index],
                                    style_cell={'textAlign': 'center'},
                                    style_header={'fontWeight': 'bold'}
                                ),
                                
                                html.H5("Statistics by User Type", style={'marginTop': 30}),
                                dash_table.DataTable(
                                    data=charging_results['descriptive_stats']['by_user_type'].reset_index().to_dict('records'),
                                    columns=[
                                        {'name': i, 'id': i} for i in 
                                        ['User Type'] + list(charging_results['descriptive_stats']['by_user_type'].columns)
                                    ],
                                    style_cell={'textAlign': 'center'},
                                    style_header={'fontWeight': 'bold'}
                                )
                            ], style={'marginBottom': 30, 'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px'}),
                            
                            # Two-way ANOVA Results
                            html.H4("Two-way ANOVA Results", style={'marginTop': 20}),
                            html.Div([
                                html.H5("Day of Week Analysis"),
                                dash_table.DataTable(
                                    data=charging_results['statistical_tests']['day_anova'].reset_index().round(4).to_dict('records'),
                                    columns=[{'name': i, 'id': i} for i in charging_results['statistical_tests']['day_anova'].reset_index().columns]
                                ),
                                
                                html.H5("Time of Day Analysis", style={'marginTop': 20}),
                                dash_table.DataTable(
                                    data=charging_results['statistical_tests']['time_anova'].reset_index().round(4).to_dict('records'),
                                    columns=[{'name': i, 'id': i} for i in charging_results['statistical_tests']['time_anova'].reset_index().columns]
                                )
                            ]),
                            
                            # Chi-square and T-test Results
                            html.H4("Additional Statistical Tests", style={'marginTop': 40}),
                            html.P([
                                "Chi-square test (Time of Day Preferences) p-value: ",
                                html.Strong(f"{charging_results['statistical_tests']['chi_square']['p_value']:.4f}")
                            ]),
                            html.P([
                                "T-test (User Type Comparison) p-value: ",
                                html.Strong(f"{charging_results['statistical_tests']['t_test']['p_value']:.4f}")
                            ])
                        ], style={'padding': '20px'})
                    ])
                ])
            ]),
        ]),
        
        # Research Question 3: Temperature Impact Analysis
        dcc.Tab(label='Temperature Impact Analysis', children=[
            html.Div([
                # Key Insights Section
                html.H3("Temperature Impact Analysis", 
                        style={'marginBottom': 20, 'color': '#34495e'}),
                
                dcc.Tabs([
                    # Visualizations Tab
                    dcc.Tab(label='Visualizations', children=[
                        # Temperature Insights
                        html.Div([
                            html.P([
                                "Temperature Correlation: ",
                                html.Strong(temp_insights['correlation'])
                            ]),
                            html.P([
                                "Temperature Range Impact: ",
                                html.Strong(temp_insights['temp_range_impact'])
                            ]),
                            html.P([
                                "Temperature Range: ",
                                html.Strong(temp_insights['temperature_range'])
                            ]),
                            html.P([
                                "Statistical Significance: ",
                                html.Strong(temp_insights['statistical_significance'])
                            ])
                        ], style={'marginBottom': 30, 'backgroundColor': '#f8f9fa', 
                                'padding': '20px', 'borderRadius': '5px'}),
                        
                        # Graphs
                        dcc.Graph(figure=temp_visualizations['scatter']),
                        dcc.Graph(figure=temp_visualizations['box']),
                        dcc.Graph(figure=temp_visualizations['line'])
                    ]),
                    
                    # Statistics Tab
                    dcc.Tab(label='Statistics', children=[
                        html.Div([
                            # Combined Statistics Table
                            html.H4("Descriptive Statistics"),
                            dash_table.DataTable(
                                data=pd.DataFrame({
                                    'Statistic': temp_analysis['descriptive_stats']['temperature'].index,
                                    'Temperature (°C)': temp_analysis['descriptive_stats']['temperature'].values.round(3),
                                    'Energy Efficiency (kWh/km)': temp_analysis['descriptive_stats']['efficiency'].values.round(3)
                                }).to_dict('records'),
                                columns=[
                                    {'name': 'Statistic', 'id': 'Statistic'},
                                    {'name': 'Temperature (°C)', 'id': 'Temperature (°C)'},
                                    {'name': 'Energy Efficiency (kWh/km)', 'id': 'Energy Efficiency (kWh/km)'}
                                ],
                                style_cell={
                                    'textAlign': 'left',
                                    'padding': '10px',
                                    'backgroundColor': 'white'
                                },
                                style_header={
                                    'backgroundColor': '#f8f9fa',
                                    'fontWeight': 'bold',
                                    'color': '#2c3e50'
                                },
                                style_table={
                                    'borderRadius': '5px',
                                    'overflow': 'hidden',
                                    'border': '1px solid #e0e0e0'
                                }
                            ),
                            
                            # Statistical Tests
                            html.H4("Statistical Tests", style={'marginTop': 30}),
                            html.Div([
                                html.P([
                                    "Correlation coefficient: ",
                                    html.Strong(f"{temp_analysis['correlation']:.3f}")
                                ]),
                                html.P([
                                    "F-statistic: ",
                                    html.Strong(f"{temp_analysis['f_statistic']:.3f}"),
                                    ", p-value: ",
                                    html.Strong(f"{temp_analysis['p_value']:.3f}")
                                ])
                            ], style={
                                'backgroundColor': '#f8f9fa',
                                'padding': '15px',
                                'borderRadius': '5px',
                                'marginTop': '10px'
                            })
                        ], style={'padding': '20px'})
                    ])
                ])
            ])
        ]),
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)