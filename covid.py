from dash import Dash, html, dash_table, dcc, Input, Output, State
from components.clustering import COVIDClustering1
from typing import Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os

app = Dash("Covid study dashboard")


cases_type = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'reproduction_rate', 'icu_patients', 'hosp_patients', 'total_tests', 'new_tests', 'positive_rate', 'tests_per_case', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'new_vaccinations', 'vaccinations_per_hundred']


hard_coded_dates = ['2020-05-31', '2020-09-06', '2020-12-06',
                    '2021-03-07', '2021-06-06', '2021-09-05', '2021-12-05',
                    '2022-03-06', '2022-06-05', '2022-09-04', '2022-12-25',
                    '2023-03-05', '2023-06-04', '2023-09-03', '2023-12-03',
                    '2024-03-03', '2024-06-02']

# Load dataset
path = os.path.join('data', 'covid_data.csv')
covid_df = pd.read_csv(path, delimiter=",", dtype=str)
print(covid_df.head())

# INITIALIZATION Of the CLUSTERING MANAGER
clustering_manager = COVIDClustering1(covid_df)
base_features = [
    'total_cases_per_million',
    'total_deaths_per_million', 
    'gdp_per_capita',
    'human_development_index',
    'aged_65_older',
    'population_density'
]

def get_all_iso_code() -> dict:
    # get subset of unique iso_code with locations
    location = covid_df.set_index('iso_code')['location'].to_dict()
    return location

def get_all_dates() -> list:
    dates = covid_df['date'].dropna().unique().tolist()
    dates.sort()
    corrected_dates = []
    for date in dates:
        try:
            pd.to_datetime(date)
            values = covid_df[covid_df['date'] == date]["new_cases"].astype(float).dropna()
            if len(values) == 0 or values.sum() == 0:
                continue
            corrected_dates.append(date)
        except Exception:
            continue
    return dates

def evolution_over_time(iso_code: str) -> go.Figure:
    df_filtered = covid_df[covid_df['iso_code'] == iso_code]
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered = df_filtered.sort_values('date')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['total_cases'].astype(float),
        mode='lines+markers',
        name='Total Cases'
    ))
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['total_deaths'].astype(float),
        mode='lines+markers',
        name='Total Deaths'
    ))
    fig.update_layout(
        title=f'Covid-19 Evolution Over Time for {iso_code}',
        xaxis_title='Date',
        yaxis_title='Count'
    )
    return fig

def total_case_total_death_date_geo(date: str, absolute:bool=True) -> go.Figure:
    df_filtered = covid_df[covid_df['date'] == date]
    df_filtered['total_cases'] = df_filtered['total_cases'].astype(float).fillna(0)
    df_filtered['total_deaths'] = df_filtered['total_deaths'].astype(float).fillna(0)
    if not absolute:
        df_filtered['population'] = df_filtered['population'].astype(float).replace(0, np.nan).fillna(1)
        df_filtered['total_cases'] = df_filtered['total_cases'] / df_filtered['population'] * 100000
        df_filtered['total_deaths'] = df_filtered['total_deaths'] / df_filtered['population'] * 100000

    df_filtered['log_total_cases'] = np.log1p(df_filtered['total_cases'])
    df_filtered['log_total_deaths'] = np.log1p(df_filtered['total_deaths'])

    fig = px.scatter_geo(
        df_filtered,
        locations="iso_code",
        color="log_total_deaths",
        size="log_total_cases",
        hover_name="location",
        projection="natural earth",
        title=f'Global Covid-19 Cases and Deaths on {date}'
    )
    return fig

def total_case_total_death_date(date: str, absolute:bool=True) -> go.Figure:
    df_filtered = covid_df[covid_df['date'] == date]
    df_filtered['total_cases'] = df_filtered['total_cases'].astype(float).fillna(0)
    df_filtered['total_deaths'] = df_filtered['total_deaths'].astype(float).fillna(0)
    
    if not absolute:
        df_filtered['population'] = df_filtered['population'].astype(float).replace(0, np.nan).fillna(1)
        df_filtered['total_cases'] = df_filtered['total_cases'] / df_filtered['population'] * 100000
        df_filtered['total_deaths'] = df_filtered['total_deaths'] / df_filtered['population'] * 100000

    df_filtered['log_total_cases'] = np.log1p(df_filtered['total_cases'])
    df_filtered['log_total_deaths'] = np.log1p(df_filtered['total_deaths'])

    fig = px.scatter(
        df_filtered,
        x='total_cases',
        y='total_deaths',
        hover_name='location',
        title=f'Total Cases vs Total Deaths on {date}',
        labels={'total_cases': 'Total Cases', 'total_deaths': 'Total Deaths'}
    )
    return fig

def deaths_ratio_per_country(date: str) -> go.Figure:
    df_filtered = covid_df[covid_df['date'] == date]
    df_filtered['total_cases'] = df_filtered['total_cases'].astype(float).fillna(0)
    df_filtered['total_deaths'] = df_filtered['total_deaths'].astype(float).fillna(0)
    
    df_filtered['death_ratio'] = df_filtered['total_deaths'] / df_filtered['total_cases'].replace(0, np.nan)

    fig = px.histogram(
        df_filtered,
        x='death_ratio',
        nbins=50,
        title=f'Death Ratio Distribution on {date}',
        labels={'death_ratio': 'Death Ratio (Total Deaths / Total Cases)'}
    )
    fig = px.scatter(
        df_filtered,
        x='location',
        y='death_ratio',
        hover_name='location',
        title=f'Death Ratio per Country on {date}',
        labels={'death_ratio': 'Death Ratio (Total Deaths / Total Cases)'}
    )
    return fig

def absolute_deviation_calcul(series: pd.Series) -> float:
    mean = series.mean()
    return (series - mean).abs().mean()

def correlation_coefficient(series1: pd.Series, series2: pd.Series) -> float:
    return series1.corr(series2)


def get_all_distribution(case_type: str, date: str) -> Tuple[go.Figure, float, float, float, float, float, float]:
    
    df_filtered = covid_df[covid_df['date'] == date].copy()

    df_filtered[case_type] = df_filtered[case_type].astype(float).fillna(0)
    
    mean = df_filtered[case_type].mean()
    median = df_filtered[case_type].median()
    max = df_filtered[case_type].max()
    min = df_filtered[case_type].min()
    std = df_filtered[case_type].std()
    absolute_deviation = absolute_deviation_calcul(df_filtered[case_type])
    
    normalized = False
    if case_type not in ["reproduction_rate", "positive_rate", "tests_per_case"]:
        normalized = True
        pop = pd.to_numeric(df_filtered.get('population', pd.Series(np.nan, index=df_filtered.index)), errors='coerce')
        pop = pop.replace(0, np.nan).fillna(1)
        df_filtered[case_type] = df_filtered[case_type] / pop * 1000000

    
    fig = px.histogram(
        df_filtered,
        x=case_type,
        nbins=50,
        title=f'Distribution of {case_type} on {date} {"(normalized per 1 000 000 people)" if normalized else ""}',
        labels={case_type: case_type.replace('_', ' ').title()}
    )
    return fig, mean, median, max, min, std, absolute_deviation

def get_all_boxplots(date: str) -> go.Figure:
    df_filtered = covid_df[covid_df['date'] == date].copy()
    fig = go.Figure()
    for case in cases_type:
        df_filtered[case] = df_filtered[case].astype(float).fillna(0)
        fig.add_trace(go.Box(y=df_filtered[case], name=case.replace('_', ' ').title()))
    fig.update_layout(
        title=f'Boxplots of Covid-19 Cases/Deaths on {date}',
        yaxis_title='Value'
    )
    return fig

def boxplot_per_case_type(case_type: str, date: str, delay:int=60) -> go.Figure:
    """Generate boxplot for a specific case type on a given date."""
    fig = go.Figure()



    df_filtered = covid_df[covid_df['date'] == date].copy()
    df_filtered[case_type] = df_filtered[case_type].astype(float).fillna(0)


    fig.add_trace(go.Box(y=df_filtered[case_type], name=case_type.replace('_', ' ').title()))
    fig.update_layout(
        title=f'Boxplot of {case_type.replace("_", " ").title()} on {date}',
        yaxis_title='Value'
    )
    return fig


countries = get_all_iso_code()
first_country = list(countries.keys())[0]
dates = get_all_dates()

# Requires Dash 2.17.0 or later
app.layout = [
    html.Main(children=[
        html.H1('Covid Study Dashboard', style={'textAlign': 'center'}),
        dcc.Dropdown(
            id='iso-code-dropdown',
            options=[{'label': loc, 'value': code} for code, loc in countries.items()],
            value=first_country
        ),
        dcc.Graph(
            id='evolution-graph',
            figure=evolution_over_time(first_country)
        ),
        html.Button(
            children="Absolute/relatif",
            id="refresh-button",
            className="button",
        ),
        dcc.Dropdown(
            id='date-dropdown',
            options=[date for date in dates],
            value=dates[0]
        ),
        dcc.Graph(
            id='total-case-death-graph',
            figure=total_case_total_death_date_geo(dates[0])
        ),
        dcc.Graph(
            id='total-case-vs-death-graph',
            figure=total_case_total_death_date(dates[0])
        ),
        dcc.Graph(
            id='death-ratio-graph',
            figure=deaths_ratio_per_country(dates[0])
        ),
        html.Div(id='output-container',
            children=[
                html.H2(id='distribution', children='Distribution of Cases/Deaths'),
                dcc.Dropdown(
                    id='case-type-dropdown',
                    options=[{'label': case.replace("_", " "), 'value': case} for case in cases_type],
                    value=cases_type[0]
                ),
                dcc.Dropdown(
                    id='date-distribution-dropdown',
                    options=[date for date in hard_coded_dates],
                    value=hard_coded_dates[0]
                ),
                html.Ul(id='statistics-list',
                    children=[
                        html.Li(id='mean-item', children='Mean: '),
                        html.Li(id='median-item', children='Median: '),
                        html.Li(id='max-item', children='Max: '),
                        html.Li(id='min-item', children='Min: '),
                        html.Li(id='std-item', children='Standard Deviation: '),
                        html.Li(id='absolute-deviation-item', children='Absolute Deviation: '),
                    ]
                ),
                dcc.Graph(
                    id='distribution-graph',
                    figure=get_all_distribution(cases_type[0], dates[0])[0]
                )
            ]
        ),

        html.H2("Clustering des Pays"),
        html.Div([
            html.Div([
                html.Label("Méthode de clustering:"),
                dcc.Dropdown(
                    id='course-method',
                    options=[
                        {'label': 'K-means (Auto K)', 'value': 'kmeans_auto'},
                        {'label': 'K-means (K fixe)', 'value': 'kmeans_fixed'},
                        {'label': 'Gaussian Mixture', 'value': 'gmm'},
                        {'label': 'DBSCAN', 'value': 'dbscan'}
                    ],
                    value='kmeans_auto'
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                html.Label("Nombre de clusters (si fixe):"),
                dcc.Input(
                    id='course-n-clusters',
                    type='number',
                    min=2, max=10,
                    value=4,
                    disabled=False
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        ]),
        html.Div([
            html.Button("Méthode du Coude", id="course-elbow-btn"),
            html.Button("Comparer Méthodes", id="course-compare-btn"),
            html.Button("Lancer Clustering", id="course-cluster-btn"),
        ], style={'margin': '20px 0'}),
        
        html.Div(id="course-elbow-plot"),
        html.Div(id="course-methods-comparison"),
        html.Div(id="course-clustering-results")
    ]
)]


@app.callback(
    Output('evolution-graph', 'figure'),
    Input('iso-code-dropdown', 'value')
)
def update_evolution_graph(iso_code):
    if not iso_code:
        return go.Figure()
    try:
        return evolution_over_time(iso_code)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error building figure: {e}", showarrow=False)
        return fig

@app.callback(
    Output('total-case-death-graph', 'figure'),
    Output('total-case-vs-death-graph', 'figure'),
    Output('death-ratio-graph', 'figure'),
    Input('refresh-button', 'n_clicks'),
    Input('date-dropdown', 'value')
)
def update_total_graphs(n_clicks, date):
    if not date:
        return go.Figure(), go.Figure()
    try:
        absolute = (n_clicks % 2 == 0) if n_clicks is not None else True
        fig_geo = total_case_total_death_date_geo(date, absolute)
        fig_scatter = total_case_total_death_date(date, absolute)
        fig_death_ratio = deaths_ratio_per_country(date)
        return fig_geo, fig_scatter, fig_death_ratio
    except Exception as e:
        err = go.Figure()
        err.add_annotation(text=f"Error building figure: {e}", showarrow=False)
        return err, err, err


@app.callback(
    Output('distribution', 'children'),
    Output('distribution-graph', 'figure'),
    Output('mean-item', 'children'),
    Output('median-item', 'children'),
    Output('max-item', 'children'),
    Output('min-item', 'children'),
    Output('std-item', 'children'),
    Output('absolute-deviation-item', 'children'),
    Input('case-type-dropdown', 'value'),
    Input('date-distribution-dropdown', 'value')
)
def update_distribution_graph(case_type:str, date:str):
    if not case_type or not date:
        return go.Figure()
    try:
        title = f'Distribution of {case_type.replace("_", " ").title()}'
        fig, mean, median, max, min, std, absolute_deviation = get_all_distribution(case_type, date)
        return title, \
            fig, \
            f'Mean: {mean:.2f}', \
            f'Median: {median:.2f}', \
            f'Max: {max:.2f}', \
            f'Min: {min:.2f}', \
            f'Standard Deviation: {std:.2f}', \
            f'Absolute Deviation: {absolute_deviation:.2f}'
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error building figure: {e}", showarrow=False)
        return fig


@app.callback(
    Output('course-elbow-plot', 'children'),
    Input('course-elbow-btn', 'n_clicks')
)
def show_elbow_plot(n_clicks):
    if not n_clicks:
        return ""
    
    try:
        elbow_fig, inertias = clustering_manager.create_elbow_plot(base_features)
        return dcc.Graph(figure=elbow_fig)
    except Exception as e:
        return html.Div(f"Erreur: {str(e)}")

@app.callback(
    Output('course-methods-comparison', 'children'),
    Input('course-compare-btn', 'n_clicks')
)
def compare_methods(n_clicks):
    if not n_clicks:
        return ""
    
    try:
        comparison = clustering_manager.compare_methods(base_features)
        
        results = [html.H4("Comparaison des Méthodes")]
        for method, result in comparison.items():
            silhouette_text = f", Silhouette: {result.get('silhouette', 'N/A')}" if 'silhouette' in result else ""
            results.append(html.P(f"{method.upper()}: {result['n_clusters']} clusters{silhouette_text}"))
        
        return html.Div(results)
    except Exception as e:
        return html.Div(f"Erreur: {str(e)}")

@app.callback(
    Output('course-clustering-results', 'children'),
    Input('course-cluster-btn', 'n_clicks'),
    [State('course-method', 'value'),
     State('course-n-clusters', 'value')]
)
def run_course_clustering(n_clicks, method, n_clusters):
    if not n_clicks:
        return ""
    
    try:
        if method == 'kmeans_auto':
            clustered_data, model, features, silhouette = clustering_manager.perform_kmeans(
                base_features, auto_select=True
            )
        elif method == 'kmeans_fixed':
            clustered_data, model, features, silhouette = clustering_manager.perform_kmeans(
                base_features, n_clusters=n_clusters, auto_select=False
            )
        elif method == 'gmm':
            clustered_data, model, features = clustering_manager.perform_gaussian_mixture(
                base_features, n_components=n_clusters
            )
        else:  # dbscan
            clustered_data, model, features = clustering_manager.perform_dbscan(base_features)
        
        # Interprétation
        interpretations = clustering_manager.interpret_clusters(clustered_data, features)
        
        # Visualisations
        visualizations = clustering_manager.create_clustering_visualizations(
            clustered_data, features, method.upper()
        )
        
        # Construction des résultats
        results = [
            html.H3("Résultats du Clustering"),
            html.H4(f"Méthode: {method.upper()}"),
        ]
        
        # Score silhouette si disponible
        if 'silhouette' in locals() and silhouette != -1:
            results.append(html.P(f"Score Silhouette: {silhouette:.3f}"))
        
        # Interprétation textuelle
        results.append(html.H4("Interprétation des Clusters"))
        for interp in interpretations:
            results.extend([
                html.H5(f"{interp['cluster']}: {interp['label']}"),
                html.P(f"Pays représentatifs: {', '.join(interp['countries'][:3])}"),
                html.Hr()
            ])
        
        # Graphiques
        results.extend([
            dcc.Graph(figure=visualizations['pca']),
            dcc.Graph(figure=visualizations['map']),
            dcc.Graph(figure=visualizations['heatmap'])
        ])
        
        return html.Div(results)
        
    except Exception as e:
        return html.Div(f"Erreur: {str(e)}")


if __name__ == '__main__':
    app.run(debug=True)