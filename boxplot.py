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

cases_types_dict = {
    'total_cases': False,
    'new_cases': True,
    'total_deaths': False,
    'new_deaths': True,
    'reproduction_rate': True,
    'icu_patients': True,
    'hosp_patients': True,
    'total_tests': False,
    'new_tests': True,
    'positive_rate': True,
    'tests_per_case': True,
    'total_vaccinations': False,
    'people_vaccinated': False,
    'people_fully_vaccinated': False,
    'new_vaccinations': True
}

# Load dataset
path = os.path.join('data', 'covid_data.csv')
covid_df = pd.read_csv(path, delimiter=",", dtype=str)


hard_coded_dates = ['2020-05-31', '2020-09-06', '2020-12-06',
                    '2021-03-07', '2021-06-06', '2021-09-05', '2021-12-05',
                    '2022-03-06', '2022-06-05', '2022-09-04', '2022-12-25',
                    '2023-03-05', '2023-06-04', '2023-09-03', '2023-12-03',
                    '2024-03-03', '2024-06-02']

def absolute_deviation_calcul(series: pd.Series) -> float:
    mean = series.mean()
    return (series - mean).abs().mean()


def adjacency_matrix_human_development_index(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series, list]:
    """Compute adjacency matrix based on human development index differences.
    5 if difference == 0
    4 if difference == 0.01
    3 if difference == 0.02
    2 if difference == 0.03
    1 if difference == 0.04
    0 otherwise
    Args:
        df (pd.DataFrame): DataFrame containing 'location', 'human_development
        _index', and 'continent' columns.
    Returns:
        np.ndarray: Adjacency matrix as per the defined scoring system.
    """

    tmp = df[['location', 'human_development_index', 'continent']].copy()
    tmp['human_development_index'] = pd.to_numeric(tmp['human_development_index'], errors='coerce')
    tmp = tmp[tmp['human_development_index'].notna() & tmp['continent'].notna()]

    hdi_series = tmp.groupby('location', sort=False)['human_development_index'].first()
    locations = hdi_series.index.to_list()
    hdi_values = hdi_series.to_numpy(dtype=float)

    # Pairwise absolute differences using broadcasting (very fast in numpy)
    diffs = np.abs(hdi_values[:, None] - hdi_values[None, :])
    # Round differences to 2 decimals to match original binning logic
    diffs = np.round(diffs, 2)

    # Vectorized mapping: 0->5, 0.01->4, 0.02->3, 0.03->2, 0.04->1, else 0
    targets = np.array([0.00, 0.01, 0.02, 0.03, 0.04])
    scores = np.array([5, 4, 3, 2, 1], dtype=np.int8)

    # Use broadcasting with isclose for robustness against float errors
    matrix = np.zeros_like(diffs, dtype=np.int8)
    for tval, score in zip(targets, scores):
        mask = np.isclose(diffs, tval, atol=1e-6)
        matrix[mask] = score
    
    return matrix, hdi_series, locations


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

    df_clean = covid_df[pd.to_numeric(covid_df['total_cases'], errors='coerce').notnull()].copy()
    df_clean['total_cases'] = pd.to_numeric(df_clean['total_cases'], errors='coerce')
    df_clean = df_clean[df_clean['total_cases'] > 0]
    df_clean = df_clean[df_clean['continent'].notna()]

    if is_absolute:
        df_grouped = df_clean.groupby('date')['total_cases']
    else:
        df_clean['population'] = pd.to_numeric(df_clean['population'], errors='coerce')
        df_grouped = (df_clean['total_cases'] / df_clean['population'] * 1000000).groupby(df_clean['date'])

    dates = []
    medians = []
    means = []
    std_devs = []
    p25s = []
    p75s = []
    particular_quantiles = []

    for date, group in df_grouped:
        dates.append(date)
        medians.append(group.median())
        means.append(group.mean())
        std_devs.append(group.std())
        p25s.append(group.quantile(0.25))
        p75s.append(group.quantile(0.75))
        if particular_quantile is not None:
            particular_quantiles.append(group.quantile(particular_quantile))

    fig.add_trace(go.Scatter(x=dates, y=medians, mode='lines+markers', name='Median'))
    fig.add_trace(go.Scatter(x=dates, y=means, mode='lines+markers', name='Mean'))
    fig.add_trace(go.Scatter(x=dates, y=p25s, mode='lines', name='25th Percentile', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=dates, y=p75s, mode='lines', name='75th Percentile', line=dict(dash='dash')))
    if particular_quantile is not None:
        fig.add_trace(go.Scatter(x=dates, y=particular_quantiles, mode='lines', name=f'{particular_quantile*100}th Percentile', line=dict(dash='dot')))

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


def hdi_adjacency_networks(df: pd.DataFrame) -> nx.Graph:
    """
    Create a NetworkX graph based on HDI adjacency matrix.
    Nodes are countries, edges are weighted by HDI similarity scores.
    """
    matrix, hdi_series, locations = adjacency_matrix_human_development_index(df)

    G = nx.Graph()

    for i, loc1 in enumerate(locations):
        G.add_node(loc1, human_development_index=hdi_series[loc1])
        for j, loc2 in enumerate(locations):
            if i < j and matrix[i, j] > 0:
                G.add_edge(loc1, loc2, weight=matrix[i, j])
    return G

def hdi_adjacency_network_figure(G: nx.Graph) -> go.Figure:
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
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>HDI: {G.nodes[node]['human_development_index']}")
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text)
    

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


app.layout = [
    html.Main(children=[
        html.H1('Covid Study Dashboard'),
        html.Section(
            className="section",
            children=[
                html.H2("Section 1 : Etude du nombre de cas totaux de Covid-19 entre pays"),
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
                    id='absolute-button',
                    n_clicks=0,
                    className="button",
                ),
            ]
        ),
        dcc.Graph(
            id='box-plot',
            figure=total_case_box_plot(covid_df, hard_coded_dates[0])[0]
        ),
        dcc.Graph(
            id='descriptive-stats-plot',
            figure=total_case_box_plot(covid_df, hard_coded_dates[0])[1]
        ),
        
        html.Section(
            className="section",
            children=[
                html.H2("Section 2 : Etude du nombre de nouveaux cas de Covid-19 entre pays"),
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
            ]
        ),
        dcc.Graph(
            id='box-plot-new-cases',
            figure=new_case_box_plot(covid_df, hard_coded_dates[0])[0]
        ),
        dcc.Graph(
            id='descriptive-stats-plot-new-cases',
            figure=new_case_box_plot(covid_df, hard_coded_dates[0])[1]
        ),


        html.Section(
            className="section",
            children=[
                html.H2("Section 3 : Etude globale des autres variables entre pays"),
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
            ]
        ),
        dcc.Graph(
            id='box-plot-general-case',
            figure=general_case_box_plot(covid_df, cases_type[0], hard_coded_dates[0])[0]
        ),
        dcc.Graph(
            id='descriptive-stats-plot-general-case',
            figure=general_case_box_plot(covid_df, cases_type[0], hard_coded_dates[0])[1]
        ),

        
        html.Section(
            className="section",
            children=[
                html.H2("Section 4 : Etude des variations suivant les dates du nombre de cas totaux."),
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
            ]
        ),
        dcc.Graph(
            id='box-plot-total-evolution',
            figure=total_case_evolution(particular_quantile=0.9, is_absolute=True)
        ),

        html.Section(
            className="section",
            children=[
                html.H2("Section 5 : Matrice d'adjacence basée sur l'indice de développement humain (IDH)"),
                html.P(
                    "Cette section présente une matrice d'adjacence basée sur les différences d'indice de développement humain (IDH) entre les pays."
                    " Les pays sont connectés par des arêtes pondérées en fonction de la similarité de leur IDH."
                    " Cette matrice peut être utilisée pour analyser les relations entre les pays en fonction de leur niveau de développement humain."
                    " De plus, avec le travail effectué sur les clusters, cela permet de faire des visualisation sur des potentielles corrélations entre hdi et gestion de la pandémie."
                ),
            ]
        ),
        dcc.Graph(
            id='hdi-adjacency-network',
            figure=hdi_adjacency_network_figure(hdi_adjacency_networks(covid_df))
        ),
        html.Footer(
            className="footer",
            children="Covid Study Dashboard © 2025 - Coumba Bocar KANE & Quentin POTIRON"
        )
    ])
]


@app.callback(
    Output('box-plot', 'figure'),
    Output('descriptive-stats-plot', 'figure'),
    Output('absolute-button', 'children'),
    Input('date-dropdown', 'value'),
    Input('absolute-button', 'n_clicks'),
)
def update_box_plot(selected_date: str, is_absolute: int) -> Tuple[go.Figure, go.Figure, str]:
    """
    Update the box plot based on the selected date from the dropdown.

    Args:
        selected_date (str): The date selected from the dropdown.

    Returns:
        go.Figure: The updated box plot figure.
    """
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



if __name__ == '__main__':
    app.run(debug=True)