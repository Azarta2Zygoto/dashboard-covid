from dash import Dash, html, dcc, Input, Output, State
from components.clustering import COVIDClustering1
from components.adjacency import hdi_adjacency_networks
from typing import Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os

BASE = "/dashboard-covid"
app = Dash(
    "Covid study dashboard",
    requests_pathname_prefix=BASE + "/",
    routes_pathname_prefix=BASE + "/")


cases_type = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'reproduction_rate', 'icu_patients', 'hosp_patients', 'total_tests', 'new_tests', 'positive_rate', 'tests_per_case', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'new_vaccinations', 'vaccinations_per_hundred']


hard_coded_dates = ['2020-05-31', '2020-09-06', '2020-12-06',
                    '2021-03-07', '2021-06-06', '2021-09-05', '2021-12-05',
                    '2022-03-06', '2022-06-05', '2022-09-04', '2022-12-25',
                    '2023-03-05', '2023-06-04', '2023-09-03', '2023-12-03',
                    '2024-03-03', '2024-06-02']
base_features = [
    'total_cases_per_million',
    'total_deaths_per_million', 
    'gdp_per_capita',
    'human_development_index',
    'aged_65_older',
    'population_density'
]


# Load dataset
path = os.path.join('data', 'covid_data.csv')
covid_df = pd.read_csv(path, delimiter=",", dtype=str)

# Data preprocessing
print("Preprocessing data...")
covid_df['date'] = pd.to_datetime(covid_df['date'], errors='coerce')
covid_df.sort_values(['iso_code', 'date'], inplace=True)
covid_df.reset_index(drop=True, inplace=True)

# INITIALIZATION Of the CLUSTERING MANAGER
clustering_manager = COVIDClustering1(covid_df)


def get_all_iso_code() -> dict:
    """Get all ISO codes with their corresponding location names."""
    location = covid_df.set_index('iso_code')['location'].to_dict()
    return location


def get_all_dates() -> list:
    """Get all valid dates from the dataset."""
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


# Precompute countries and dates for dropdowns
countries = get_all_iso_code()
first_country = list(countries.keys())[0]
dates = get_all_dates()



def evolution_over_time(iso_code: str, absolute:bool=True) -> go.Figure:
    df_filtered = covid_df[covid_df['iso_code'] == iso_code]
    if df_filtered.empty:
        return go.Figure()
    
    total_case = df_filtered['total_cases'].astype(float).fillna(0)
    total_death = df_filtered['total_deaths'].astype(float).fillna(0)
    if not absolute:
        population = df_filtered['population'].astype(float).replace(0, np.nan).fillna(1)
        total_case = total_case / population * 1000000
        total_death = total_death / population * 1000000

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=total_case.astype(float),
        mode='lines+markers',
        name='Total Cases'
    ))
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=total_death.astype(float),
        mode='lines+markers',
        name='Total Deaths'
    ))
    fig.update_layout(
        title=f"Evolution du nombre d'infectés et de mort du Covid 19 dans le temps dans le pays {countries.get(iso_code, 'Unknown Country')}{'' if absolute else ' par million d habitants'}",
        xaxis_title='Date',
        yaxis_title='Count'
    )
    return fig

def total_case_total_death_date_geo(date: str, absolute:bool=True) -> go.Figure:
    correct_date = pd.to_datetime(date)
    df_filtered = covid_df[covid_df['date'] == correct_date].copy()
    if df_filtered.empty:
        return go.Figure()
    
    total_case = df_filtered['total_cases'].astype(float).fillna(0)
    total_death = df_filtered['total_deaths'].astype(float).fillna(0)
    if not absolute:
        population = df_filtered['population'].astype(float).replace(0, np.nan).fillna(1)
        total_case = total_case / population * 1000000
        total_death = total_death / population * 1000000

    df_filtered['log_total_cases'] = np.log1p(total_case)
    df_filtered['log_total_deaths'] = np.log1p(total_death)

    fig = px.scatter_geo(
        df_filtered,
        locations="iso_code",
        color="log_total_deaths",
        size="log_total_cases",
        hover_name="location",
        projection="natural earth",
        title=f'Nombre de cas et de décès dû au Covid-19 le {date} {"" if absolute else "(par million d habitants)"}',
    )
    return fig

def absolute_deviation_calcul(series: pd.Series) -> float:
    mean = series.mean()
    return (series - mean).abs().mean()

def correlation_coefficient(series1: pd.Series, series2: pd.Series) -> float:
    return series1.corr(series2)


def get_all_distribution(case_type: str, date: str) -> Tuple[go.Figure, float, float, float, float, float, float]:
    correct_date = pd.to_datetime(date)
    df_filtered = covid_df[covid_df['date'] == correct_date].copy()

    df_filtered[case_type] = df_filtered[case_type].astype(float).fillna(0)
    df_filtered = df_filtered[df_filtered[case_type] > 0]
    
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


def total_case_box_plot(
    df: pd.DataFrame,
    date: str,
    delta:int=1,
    is_absolute: bool=True
) -> Tuple[go.Figure, go.Figure]:
    """
    Create a box plot for total cases on a given date.

    Args:
        df (pd.DataFrame): The dataframe containing covid data.
        date (str): The date for which to create the box plot.

    Returns:
        go.Figure: A Plotly box plot figure.
    """
    fig = go.Figure()

    describes = []

    for i in range(5):
        date_value = pd.to_datetime(date) - pd.DateOffset(days=(i-2)*delta*7)
        date_str = date_value.strftime('%Y-%m-%d')
        temp_df = df[df['date'] == date_str].copy()
        temp_df = temp_df[pd.to_numeric(temp_df['total_cases'], errors='coerce').notnull()]
        temp_df['total_cases'] = pd.to_numeric(temp_df['total_cases'], errors='coerce')
        temp_df = temp_df[temp_df['total_cases'] > 0]
        temp_df = temp_df[temp_df['continent'].notna()]
        values = temp_df['total_cases']
        if not is_absolute:
            values = temp_df['total_cases'] / pd.to_numeric(temp_df['population'], errors='coerce') * 1000000
        
        median = values.median()
        absolute_deviation = absolute_deviation_calcul(values)

        describes.append(values.describe())
        describes[-1]['absolute_deviation'] = absolute_deviation
        describes[-1]['median'] = median
        fig.add_trace(go.Box(
            y=values,
            name=date_str,
            boxmean='sd',
            boxpoints='all',                # afficher tous les points
            jitter=0.6,                     # dispersion des points pour lisibilité
            pointpos=0,                     # position des points relatif à la boîte
            marker=dict(size=5, opacity=0.8),
            hovertext=temp_df['location'],  # texte affiché pour chaque point
            hovertemplate='%{hovertext}<br>Total Cases: %{y}<extra></extra>'
        ))

    title = 'Total Cases Box Plot on ' + date
    if not is_absolute:
        title += ' (per million inhabitants)'
    fig.update_layout(title=title)

    fig2 = go.Figure()
    for i, desc in enumerate(describes):
        fig2.add_trace(go.Bar(
            x=desc.index,
            y=desc.values,
            name=(pd.to_datetime(date) - pd.DateOffset(days=(i-2)*delta*7)).strftime('%Y-%m-%d')
        ))
    title2 = 'Descriptive Statistics for Total Cases on ' + date
    if not is_absolute:
        title2 += ' (per million inhabitants)'
    fig2.update_layout(title=title2)

    return fig, fig2

def total_case_evolution(particular_quantile:float, is_absolute: bool=True) -> go.Figure:
    """
    Create a line plot for total cases evolution over time.
    There are also lines for median, mean and std deviation and percentiles.
    """
    fig = go.Figure()

    df_clean = covid_df.copy()
    df_clean['total_cases'] = pd.to_numeric(df_clean['total_cases'], errors='coerce')
    df_clean = df_clean[df_clean['total_cases'] > 0]
    df_clean = df_clean[df_clean['continent'].notna()]

    if df_clean.empty:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available for the selected options.", showarrow=False)
        return empty_fig

    if is_absolute:
        df_clean['value'] = df_clean['total_cases']
    else:
        df_clean = df_clean[df_clean['population'].notna()]
        if df_clean.empty:
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="No population data to compute per-million values.", showarrow=False)
            return empty_fig
        df_clean['value'] = df_clean['total_cases'] / df_clean['population'] * 1e6

    grouped = df_clean.groupby('date')['value']
    if grouped.ngroups == 0:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No grouped data available.", showarrow=False)
        return empty_fig
    
    medians = grouped.median().sort_index()
    means = grouped.mean().sort_index()
    stds = grouped.std().sort_index().fillna(0)
    p25 = grouped.quantile(0.25).sort_index()
    p75 = grouped.quantile(0.75).sort_index()

    particular_series = None
    if particular_quantile is not None:
        if not (0 <= particular_quantile <= 1):
            particular_quantile = 0
        else:
            particular_series = grouped.quantile(particular_quantile).sort_index()

    # Ensure x-axis are datetimes sorted
    x = pd.to_datetime(medians.index)

    # Mean +/- std band
    upper = (means + stds).values
    lower = (means - stds).clip(lower=0).values

    fig.add_trace(go.Scatter(
        x=x, y=upper,
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=lower,
        fill='tonexty',
        fillcolor='rgba(173,216,230,0.2)',  # light blue band
        line=dict(width=0),
        name='Mean ± Std',
        hoverinfo='skip'
    ))

    # Median and Mean lines
    fig.add_trace(go.Scatter(x=x, y=medians.values, mode='lines+markers', name='Median'))
    fig.add_trace(go.Scatter(x=x, y=means.values, mode='lines+markers', name='Mean'))
    # Percentile ribbons (25/75)
    fig.add_trace(go.Scatter(x=x, y=p25.values, mode='lines', name='25th Percentile', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=x, y=p75.values, mode='lines', name='75th Percentile', line=dict(dash='dash')))

    if particular_series is not None:
        fig.add_trace(go.Scatter(x=x, y=particular_series.values, mode='lines', name=f'{int(particular_quantile*100)}th Percentile', line=dict(dash='dot')))

    title = 'Total Cases Evolution Over Time'
    if not is_absolute:
        title += ' (per million inhabitants)'
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Total Cases')

    return fig



def general_case_box_plot(
    df: pd.DataFrame,
    case:str,
    date: str,
    delta:int=1,
    is_absolute: bool=True
) -> Tuple[go.Figure, go.Figure]:
    """
    Create a box plot for new cases on a given date.

    Args:
        df (pd.DataFrame): The dataframe containing covid data.
        date (str): The date for which to create the box plot.

    Returns:
        go.Figure: A Plotly box plot figure.
    """
    fig = go.Figure()

    describes = []

    for i in range(5):
        date_value = pd.to_datetime(date) - pd.DateOffset(days=(i-2)*delta*7)
        date_str = date_value.strftime('%Y-%m-%d')
        temp_df = df[df['date'] == date_str].copy()
        temp_df = temp_df[pd.to_numeric(temp_df[case], errors='coerce').notnull()]
        temp_df[case] = pd.to_numeric(temp_df[case], errors='coerce')
        temp_df = temp_df[temp_df[case] > 0]
        temp_df = temp_df[temp_df[case].notna()]
        values = temp_df[case]
        if not is_absolute:
            values = temp_df[case] / pd.to_numeric(temp_df['population'], errors='coerce') * 1000000
        
        median = values.median()
        absolute_deviation = absolute_deviation_calcul(values)

        describes.append(values.describe())
        describes[-1]['absolute_deviation'] = absolute_deviation
        describes[-1]['median'] = median
        fig.add_trace(go.Box(
            y=values,
            name=date_str,
            boxmean='sd',
            boxpoints='all',                # afficher tous les points
            jitter=0.6,                     # dispersion des points pour lisibilité
            pointpos=0,                     # position des points relatif à la boîte
            marker=dict(size=5, opacity=0.8),
            hovertext=temp_df['location'],  # texte affiché pour chaque point
            hovertemplate='%{hovertext}<br>New Cases: %{y}<extra></extra>'
        ))

    title = f'{case} Box Plot on ' + date
    if not is_absolute:
        title += ' (per million inhabitants)'
    fig.update_layout(title=title)

    fig2 = go.Figure()
    for i, desc in enumerate(describes):
        fig2.add_trace(go.Bar(
            x=desc.index,
            y=desc.values,
            name=(pd.to_datetime(date) - pd.DateOffset(days=(i-2)*delta*7)).strftime('%Y-%m-%d')
        ))
    title2 = f'Descriptive Statistics for {case} on ' + date
    if not is_absolute:
        title2 += ' (per million inhabitants)'
    fig2.update_layout(title=title2)
    return fig, fig2


def new_case_box_plot(
    df: pd.DataFrame,
    date: str,
    delta:int=1,
    is_absolute: bool=True
) -> Tuple[go.Figure, go.Figure]:
    """
    Create a box plot for new cases on a given date.

    Args:
        df (pd.DataFrame): The dataframe containing covid data.
        date (str): The date for which to create the box plot.

    Returns:
        go.Figure: A Plotly box plot figure.
    """
    fig = go.Figure()

    describes = []

    for i in range(5):
        date_value = pd.to_datetime(date) - pd.DateOffset(days=(i-2)*delta*7)
        date_str = date_value.strftime('%Y-%m-%d')
        temp_df = df[df['date'] == date_str].copy()
        temp_df = temp_df[pd.to_numeric(temp_df['new_cases'], errors='coerce').notnull()]
        temp_df['new_cases'] = pd.to_numeric(temp_df['new_cases'], errors='coerce')
        temp_df = temp_df[temp_df['new_cases'] > 0]
        temp_df = temp_df[temp_df['continent'].notna()]
        values = temp_df['new_cases']
        if not is_absolute:
            values = temp_df['new_cases'] / pd.to_numeric(temp_df['population'], errors='coerce') * 1000000
        
        median = values.median()
        absolute_deviation = absolute_deviation_calcul(values)

        describes.append(values.describe())
        describes[-1]['absolute_deviation'] = absolute_deviation
        describes[-1]['median'] = median
        fig.add_trace(go.Box(
            y=values,
            name=date_str,
            boxmean='sd',
            boxpoints='all',                # afficher tous les points
            jitter=0.6,                     # dispersion des points pour lisibilité
            pointpos=0,                     # position des points relatif à la boîte
            marker=dict(size=5, opacity=0.8),
            hovertext=temp_df['location'],  # texte affiché pour chaque point
            hovertemplate='%{hovertext}<br>New Cases: %{y}<extra></extra>'
        ))

    title = 'New Cases Box Plot on ' + date
    if not is_absolute:
        title += ' (per million inhabitants)'
    fig.update_layout(title=title)

    fig2 = go.Figure()
    for i, desc in enumerate(describes):
        fig2.add_trace(go.Bar(
            x=desc.index,
            y=desc.values,
            name=(pd.to_datetime(date) - pd.DateOffset(days=(i-2)*delta*7)).strftime('%Y-%m-%d')
        ))
    title2 = 'Descriptive Statistics for New Cases on ' + date
    if not is_absolute:
        title2 += ' (per million inhabitants)'
    fig2.update_layout(title=title2)
    return fig, fig2


colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

def hdi_adjacency_network_figure(G: nx.Graph, clusters=None) -> go.Figure:
    """
    Create a Plotly figure for the HDI adjacency network.
    """
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    inner_colors = []
    print("cluster", clusters)
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>HDI: {G.nodes[node]['human_development_index']}")
        if clusters is not None:
            line = clusters[clusters['location'] == node]
            print(line)
            cluster_number = int(line['cluster'].values[0] if not line.empty else -1)
            inner_colors.append(colors[cluster_number % len(colors)] if cluster_number >= 0 else '#CCCCCC')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(color=inner_colors)
        )
    

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>HDI Adjacency Network',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="HDI Adjacency Network based on similarity scores",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    return fig


matrix = hdi_adjacency_networks(covid_df)
# Requires Dash 2.17.0 or later
app.layout = [
    html.Main(children=[
        html.H1('Covid Study Dashboard', style={'textAlign': 'center'}),
        html.Section(
            className="section",
            children=[
                html.H2("Introduction : Etude d'une base de données sur le Covid-19"),
                html.P(
                    "Dans cette étude, nous analysons une base de données mondiale sur le Covid-19, "
                    "contenant des informations sur les cas, les décès, les tests, la vaccination, "
                    "ainsi que des indicateurs socio-économiques pour différents pays."
                ),
                html.Hr(),
                html.Div(
                    className="row",
                    children= [
                        html.Label("Sélectionnez un pays :"),
                        dcc.Dropdown(
                            id='iso-code-dropdown',
                            options=[{'label': loc, 'value': code} for code, loc in countries.items()],
                            value=first_country
                        ),
                    ]
                ),
                html.Div(
                    className="row",
                    children= [
                        html.Label("Avoir les données en valeurs absolues ou par millions d'habitants :"),
                        html.Button(
                            children="Avoir les résultats relatifs",
                            id="absolute-button",
                            className="button",
                        ),
                    ]
                ),
                dcc.Graph(
                    id='evolution-graph',
                    figure=evolution_over_time(first_country)
                ),
            ]
        ),
        html.Section(
            className="section",
            children=[
                html.H2("Section 1 : Etude géographique et temporelle des cas et décès"),
                html.P(
                    "Cette section explore la répartition géographique et l'évolution temporelle des cas et décès liés au Covid-19."
                    " Le logarithme des données a été pris afin de mettre en évidence les différennces entre les différents pays."
                    " La couleur représente le nombre de décès tandis que la taille des points représente le nombre de cas."
                ),
                html.Hr(),
                html.Div(
                    className="row",
                    children= [
                        html.Label("Sélectionnez une date :"),
                        dcc.Dropdown(
                            id='date-dropdown-geo',
                            options=[date for date in hard_coded_dates],
                            value=hard_coded_dates[0]
                        ),
                    ]
                ),
                html.Div(
                    className="row",
                    children= [
                        html.Label("Avoir les données en valeurs absolues ou par millions d'habitants :"),
                        html.Button(
                            children="Avoir les résultats relatifs",
                            id="absolute-button-geo",
                            className="button",
                        ),
                    ]
                ),
                dcc.Graph(
                    id='total-case-death-geo',
                    figure=total_case_total_death_date_geo(dates[0])
                ),
            ]
        ),

        html.Div(
            className='section',
            children=[
                html.H2('Section 2 : Distribution sur différentes variables'),
                html.P(
                    "Cette section présente la distribution des différents types de cas et décès liés au Covid-19 à une date donnée." \
                    " Vous pouvez Sélectionnez le type de cas et la date pour visualiser la distribution correspondante."
                    " Toutes les valeurs sont normalisées par million d'habitants, sauf pour les taux (reproduction_rate, positive_rate, tests_per_case)."
                ),
                html.Hr(),
                
                html.Div(
                    className="row",
                    children= [
                        html.Label("Sélectionnez une date :"),
                        dcc.Dropdown(
                            id='date-dropdown-distribution',
                            options=[date for date in hard_coded_dates],
                            value=hard_coded_dates[0]
                        ),
                    ]
                ),
                html.Div(
                    className="row",
                    children= [
                        html.Label("Sélectionnez une variable que vous souhaitez étudier :"),
                        dcc.Dropdown(
                            id='case-type-dropdown',
                            options=[{'label': case.replace("_", " "), 'value': case} for case in cases_type],
                            value=cases_type[0]
                        ),
                    ]
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
                    figure=get_all_distribution(cases_type[0], hard_coded_dates[0])[0]
                )
            ]
        ),

        
        html.Section(
            className="section",
            children=[
                html.H2("Section 3.1 : Etude du nombre de cas totaux de Covid-19 entre pays"),
                html.P(
                    "Cette section présente une analyse du nombre total de cas de Covid-19 entre les pays."
                    " Vous pouvez sélectionner une date spécifique pour visualiser la distribution des cas totaux à travers différents pays à cette date."
                    " Les graphiques affichent les boxplots des cas totaux des pays du monde à la date donnée et 14, 7 jours avant et 7, 14 jours après pour comparison."
                    " Nettoyage des données, les pays avec 0 cas au moment de la date ne sont pas compris dans l'étude afin d'avoir les données seulement sur les pays infectés."
                ),
                html.Div(
                    className="row",
                    children= [
                        html.Label("Sélectionner une date d'étude :"),
                        dcc.Dropdown(
                            id='date-dropdown',
                            options=[date for date in hard_coded_dates],
                            value=hard_coded_dates[0]
                        ),
                    ]
                ),
                html.P(
                    "Afin de ne pas avoir de grands écarts entre les pays fortement peuplés et ceux faiblement, il est possible de normaliser les données par million d'habitants."
                    " Par défaut, les résultats sont en absolu."
                ),
                html.Button(
                    'Avoir les résultats relatifs',
                    id='absolute-button-total-cases',
                    n_clicks=0,
                    className="button",
                ),
                dcc.Graph(
                    id='box-plot',
                    figure=total_case_box_plot(covid_df, hard_coded_dates[0])[0]
                ),
                dcc.Graph(
                    id='descriptive-stats-plot',
                    figure=total_case_box_plot(covid_df, hard_coded_dates[0])[1]
                ),
            ]
        ),
        
        html.Section(
            className="section",
            children=[
                html.H2("Section 3.2 : Etude du nombre de nouveaux cas de Covid-19 entre pays"),
                html.P(
                    "Cette section présente une analyse du nombre de nouveaux cas de Covid-19 entre pays sur plusieurs dates."
                    " Vous pouvez sélectionner une date spécifique pour visualiser la distribution du nombre de nouveau cas à travers différents pays."
                    " Les graphiques affichent les boxplots à la date donnée et 14, 7 jours avant et 7, 14 jours après pour comparison."
                    " Il y a un nettoyage des données, les pays avec 0 cas au moment de la date ne sont pas compris dans l'étude afin d'avoir les données seulement sur les pays infectés."
                ),
                html.Div(
                    className="row",
                    children= [
                        html.Label("Sélectionner une date d'étude :"),
                        dcc.Dropdown(
                            id='date-dropdown-new-cases',
                            options=[date for date in hard_coded_dates],
                            value=hard_coded_dates[0]
                        ),
                    ]
                ),
                html.P(
                    "Afin de ne pas avoir de grands écarts entre les pays fortement peuplés et ceux faiblement, il est possible de normaliser les données par million d'habitants."
                    " Par défaut, les résultats sont en absolu."
                ),
                html.Button(
                    'Avoir les résultats relatifs',
                    id='absolute-button-new-cases',
                    n_clicks=0,
                    className="button",
                ),
                dcc.Graph(
                    id='box-plot-new-cases',
                    figure=new_case_box_plot(covid_df, hard_coded_dates[0])[0]
                ),
                dcc.Graph(
                    id='descriptive-stats-plot-new-cases',
                    figure=new_case_box_plot(covid_df, hard_coded_dates[0])[1]
                ),
            ]
        ),


        html.Section(
            className="section",
            children=[
                html.H2("Section 3.3 : Etude globale des autres variables entre pays"),
                html.P(
                    "Dans la base de données, il nous est donné un grand nombre de données différentes, il est possible de faire une rapide étude similaire sur chacune de ces données."
                    " Vous pouvez sélectionner une date spécifique pour visualiser la distribution de la variable choisie à travers différents pays."
                ),
                html.Div(
                    className="row",
                    children= [
                        html.Label("Sélectionner une date d'étude :"),
                        dcc.Dropdown(
                            id='date-dropdown-general-case',
                            options=[date for date in hard_coded_dates],
                            value=hard_coded_dates[0]
                        ),
                    ]
                ),
                html.Div(
                    className="row",
                    children= [
                        html.Label("Sélectionner une variable d'étude :"),
                        dcc.Dropdown(
                            id='case-dropdown-general-case',
                            options=[date for date in cases_type],
                            value=cases_type[0]
                        ),
                    ]
                ),
                html.P(
                    "Afin de ne pas avoir de grands écarts entre les pays fortement peuplés et ceux faiblement, il est possible de normaliser les données par million d'habitants."
                    " Par défaut, les résultats sont en absolu."
                ),
                html.Button(
                    'Avoir les résultats relatifs',
                    id='absolute-button-general-case',
                    n_clicks=0,
                    className="button",
                ),
                dcc.Graph(
                    id='box-plot-general-case',
                    figure=general_case_box_plot(covid_df, cases_type[0], hard_coded_dates[0])[0]
                ),
                dcc.Graph(
                    id='descriptive-stats-plot-general-case',
                    figure=general_case_box_plot(covid_df, cases_type[0], hard_coded_dates[0])[1]
                ),
            ]
        ),

        
        html.Section(
            className="section",
            children=[
                html.H2("Section 3.4 : Etude des variations suivant les dates du nombre de cas totaux."),
                html.P(
                    "Cette fois-ci, nous étudions directement l'évolution des différentes valeurs (moyenne, médianne, pourcentiles) sur le nombre de cas totaux."
                    " Attention : les calculs peuvent prendre du temps, il faut 15 secondes pour passer des valeurs absolues à releatives et inversement ou lors d'un changement du quartile."
                ),
                html.P(
                    "Afin de ne pas avoir de grands écarts entre les pays fortement peuplés et ceux faiblement, il est possible de normaliser les données par million d'habitants."
                    " Par défaut, les résultats sont en absolu."
                ),
                html.Button(
                    'Avoir les résultats relatifs',
                    id='absolute-button-total-evolution',
                    n_clicks=0,
                    className="button",
                ),
                html.P("Sélectionner le quartile à afficher :"),
                dcc.Slider(
                    id='quartile-range-slider',
                    min=1,
                    max=100,
                    value=90,
                ),
                dcc.Graph(
                    id='box-plot-total-evolution',
                    figure=total_case_evolution(particular_quantile=0.9, is_absolute=True)
                ),
            ]
        ),

        html.Section(
            className="section",
            children=[
            html.H2("Section 4 : Clustering des Pays"),
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
                html.Button("Méthode du Coude", id="course-elbow-btn", className="button"),
                html.Button("Comparer Méthodes", id="course-compare-btn", className="button"),
                html.Button("Lancer Clustering", id="course-cluster-btn", className="button"),
            ], style={'margin': '20px 0'}),
            
            html.Div(id="course-elbow-plot"),
            html.Div(id="course-methods-comparison"),
            html.Div(id="course-clustering-results"),
            ]
        ),

        html.Section(
            className="section",
            children=[
                html.H2("Section 5.1 : Matrice d'adjacence basée sur l'indice de développement humain (IDH)"),
                html.P(
                    "Cette section présente une matrice d'adjacence basée sur les différences d'indice de développement humain (IDH) entre les pays."
                    " Les pays sont connectés par des arêtes pondérées en fonction de la similarité de leur IDH."
                    " Cette matrice peut être utilisée pour analyser les relations entre les pays en fonction de leur niveau de développement humain."
                    " De plus, avec le travail effectué sur les clusters, cela permet de faire des visualisation sur des potentielles corrélations entre hdi et gestion de la pandémie."
                ),
                dcc.Graph(
                    id='hdi-adjacency-network',
                    figure=hdi_adjacency_network_figure(matrix)
                ),
            ]
        ),

        html.Section(
            className="section",
            children=[
                html.H2("Section 5.2 : Résultat du clustering et analyse par rapport à l'IDH"),
                html.P(
                    "Il est nécessaire d'avoir lancé un clustering en amont pour que cette section fonctionne."
                    " Cette section présente une visualisation des résultats du clustering des pays en fonction de leurs valeurs IDH."
                ),
                dcc.Graph(
                    id='hdi-adjacency-network-clustered',
                    figure=hdi_adjacency_network_figure(matrix)
                ),
            ]
        ),

        html.Footer(
            className="footer",
            children="Covid Study Dashboard © 2025 - Coumba Bocar KANE & Quentin POTIRON"
        )
    ]
)]


@app.callback(
    Output('evolution-graph', 'figure'),
    Output('absolute-button', 'value'),
    Input('iso-code-dropdown', 'value'),
    Input('absolute-button', 'n_clicks'),
)
def update_evolution_graph(iso_code, is_absolute:int=0):
    if not iso_code:
        return go.Figure(), 'Avoir les résultats relatifs'
    
    if is_absolute is None:
        absolute = True
    else:
        absolute = (is_absolute % 2 == 0)
    button_text = 'Avoir les résultats relatifs' if absolute else 'Avoir les résultats absolus'

    try:
        fig = evolution_over_time(iso_code, absolute=absolute)
        return fig, button_text
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error building figure: {e}", showarrow=False)
        return fig, button_text

@app.callback(
    Output('total-case-death-geo', 'figure'),
    Output('absolute-button-geo', 'value'),
    Input('date-dropdown-geo', 'value'),
    Input('absolute-button-geo', 'n_clicks'),
)
def update_total_graphs(date, is_absolute:int=0):
    if not date:
        return go.Figure(), 'Avoir les résultats relatifs'
    if is_absolute is None:
        absolute = True
    else:
        absolute = (is_absolute % 2 == 0)
    button_text = 'Avoir les résultats relatifs' if absolute else 'Avoir les résultats absolus'
        
    try:
        fig = total_case_total_death_date_geo(date, absolute=absolute)
        return fig, button_text
    
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error building figure: {e}", showarrow=False)
        return fig, button_text


@app.callback(
    Output('distribution-graph', 'figure'),
    Output('mean-item', 'children'),
    Output('median-item', 'children'),
    Output('max-item', 'children'),
    Output('min-item', 'children'),
    Output('std-item', 'children'),
    Output('absolute-deviation-item', 'children'),
    Input('case-type-dropdown', 'value'),
    Input('date-dropdown-distribution', 'value')
)
def update_distribution_graph(case_type:str, date:str):
    if not case_type or not date:
        return go.Figure()
    try:
        fig, mean, median, max, min, std, absolute_deviation = get_all_distribution(case_type, date)
        return fig, \
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
    Output('box-plot', 'figure'),
    Output('descriptive-stats-plot', 'figure'),
    Output('absolute-button-total-cases', 'children'),
    Input('date-dropdown', 'value'),
    Input('absolute-button-total-cases', 'n_clicks'),
)
def update_box_plot(selected_date: str, is_absolute: int) -> Tuple[go.Figure, go.Figure, str]:
    """
    Update the box plot based on the selected date from the dropdown.

    Args:
        selected_date (str): The date selected from the dropdown.

    Returns:
        go.Figure: The updated box plot figure.
    """
    if is_absolute is None:
        is_absolute = 0
    fig1, fig2 = total_case_box_plot(covid_df, selected_date, is_absolute=(is_absolute % 2 == 0))
    button = 'Avoir les résultats relatifs' if (is_absolute % 2 == 0) else 'Avoir les résultats absolus'
    return fig1, fig2, button


@app.callback(
    Output('box-plot-new-cases', 'figure'),
    Output('descriptive-stats-plot-new-cases', 'figure'),
    Output('absolute-button-new-cases', 'children'),
    Input('date-dropdown-new-cases', 'value'),
    Input('absolute-button-new-cases', 'n_clicks'),
)
def update_box_plot_new_cases(selected_date: str, is_absolute: int) -> Tuple[go.Figure, go.Figure, str]:
    """
    Update the box plot based on the selected date from the dropdown.

    Args:
        selected_date (str): The date selected from the dropdown.

    Returns:
        go.Figure: The updated box plot figure.
    """
    fig1, fig2 = new_case_box_plot(covid_df, selected_date, is_absolute=(is_absolute % 2 == 0))
    button = 'Avoir les résultats relatifs' if (is_absolute % 2 == 0) else 'Avoir les résultats absolus'
    return fig1, fig2, button

@app.callback(
    Output('box-plot-general-case', 'figure'),
    Output('descriptive-stats-plot-general-case', 'figure'),
    Output('absolute-button-general-case', 'children'),
    Input('date-dropdown-general-case', 'value'),
    Input('case-dropdown-general-case', 'value'),
    Input('absolute-button-general-case', 'n_clicks'),
)
def update_box_plot_general_case(selected_date: str, selected_case: str, is_absolute:int) -> Tuple[go.Figure, go.Figure, str]:
    """
    Update the box plot based on the selected date from the dropdown.

    Args:
        selected_date (str): The date selected from the dropdown.
        selected_case (str): The case type selected from the dropdown.

    Returns:
        go.Figure: The updated box plot figure.
    """
    fig1, fig2 = general_case_box_plot(covid_df, selected_case, selected_date, is_absolute=(is_absolute % 2 == 0))
    button = 'Avoir les résultats relatifs' if (is_absolute % 2 == 0) else 'Avoir les résultats absolus'
    return fig1, fig2, button




@app.callback(
    Output('box-plot-total-evolution', 'figure'),
    Output('absolute-button-total-evolution', 'children'),
    Output('quartile-range-slider', 'value'),
    Input('quartile-range-slider', 'value'),
    Input('absolute-button-total-evolution', 'n_clicks'),
)
def update_total_case_evolution(quartile:int, is_absolute: int) -> Tuple[go.Figure, str, int]:
    """
    Update the total case evolution plot based on the absolute/relative button.

    Args:
        is_absolute (int): The number of clicks on the absolute/relative button.

    Returns:
        go.Figure: The updated total case evolution figure.
    """
    print(is_absolute, "value", (is_absolute % 2 == 0))
    fig = total_case_evolution(particular_quantile=quartile/100, is_absolute=(is_absolute % 2 == 0))
    button = 'Avoir les résultats relatifs' if (is_absolute % 2 == 0) else 'Avoir les résultats absolus'
    return fig, button, quartile


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
    Output('hdi-adjacency-network-clustered', 'figure'),
    Input('course-cluster-btn', 'n_clicks'),
    [State('course-method', 'value'),
     State('course-n-clusters', 'value')]
)
def run_course_clustering(n_clicks, method, n_clusters):
    if not n_clicks:
        return "", go.Figure()
    
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
        
        print(clustered_data)
        hdi_figure = hdi_adjacency_network_figure(matrix, clusters=clustered_data)
        return html.Div(results), hdi_figure
        
    except Exception as e:
        return html.Div(f"Erreur: {str(e)}"), go.Figure()


if __name__ == '__main__':
    app.run(debug=True)