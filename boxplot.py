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
        print(f"Date: {date_str}, Median: {median}")
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
        print(f"Date: {date_str}, Median: {median}")
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
                    "Afin de ne pas avoir de grands écarts entre les pays fortement peulplés et ceux faiblement, il est possible de normaliser les données par million d'habitants."
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
                    "Afin de ne pas avoir de grands écarts entre les pays fortement peulplés et ceux faiblement, il est possible de normaliser les données par million d'habitants."
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

if __name__ == '__main__':
    app.run(debug=True)