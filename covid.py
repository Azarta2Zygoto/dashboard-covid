from dash import Dash, html, dash_table, dcc, Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os

app = Dash("Covid study dashboard")

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
        )
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
    
    
if __name__ == '__main__':
    app.run(debug=True)