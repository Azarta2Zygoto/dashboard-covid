from dash import Dash, html, dash_table, dcc, Input, Output
from typing import Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os

app = Dash("Covid study dashboard")
cases_type = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'reproduction_rate', 'icu_patients', 'hosp_patients', 'total_tests', 'new_tests', 'positive_rate', 'tests_per_case', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'new_vaccinations']


hard_coded_dates = ['2020-06-01', '2020-09-01', '2020-12-01',
                    '2021-03-01', '2021-06-01', '2021-09-01', '2021-12-01',
                    '2022-03-01', '2022-06-01', '2022-09-01', '2022-12-25',
                    '2023-03-01', '2023-06-01', '2023-09-01', '2023-12-01']

# Load dataset
path = os.path.join('data', 'covid_data.csv')
covid_df = pd.read_csv(path, delimiter=",", dtype=str)
print(covid_df.head())

def get_all_iso_code() -> dict:
    # get subset of unique iso_code with locations
    location = covid_df.set_index('iso_code')['location'].to_dict()
    return location

def get_all_dates() -> list:
    dates = covid_df['date'].dropna().unique().tolist()
    dates.sort()
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



countries = get_all_iso_code()
first_country = list(countries.keys())[0]
dates = get_all_dates()

# Requires Dash 2.17.0 or later
app.layout = [
    html.Main(children=[
        html.H1('Covid Study Dashboard'),
        dcc.Dropdown(
            id='iso-code-dropdown',
            options=[{'label': loc, 'value': code} for code, loc in countries.items()],
            value=first_country
        ),
        dcc.Graph(
            id='evolution-graph',
            figure=evolution_over_time(first_country)
        ),
        html.Button("Absolute/relatif", id="refresh-button"),
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
                ),
                dcc.Graph(
                    id='boxplot-graph',
                    figure=get_all_boxplots(hard_coded_dates[0])
                )
            ])
    ]),
]

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
if __name__ == '__main__':
    app.run(debug=True)