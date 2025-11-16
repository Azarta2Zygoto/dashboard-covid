from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os

app = Dash("Medecine dashboard")

# Load dataset
path = os.path.join('data', 'medicine_dataset.csv')
medecins_df = pd.read_csv(path, delimiter=",", dtype=str)
print(medecins_df.head())

def get_info_element(df:pd.DataFrame ,column_name:str, count:bool=False) -> dict:
    """Get unique sorted list from a DataFrame column."""
    feature_cols = [col for col in df.columns if col.startswith(column_name)]
    # Compte toutes les occurrences
    all_values = pd.Series([val for col in feature_cols for val in df[col].dropna()])
    value_counts = all_values.value_counts()
    unique_values = sorted(value_counts.index.tolist())
    if count:
        return {val: int(value_counts[val]) for val in unique_values}
    return {val: val for val in unique_values}

chemical_classifications = get_info_element(medecins_df, 'Chemical Class', count=True)
chemical_df = pd.DataFrame(list(chemical_classifications.items()), columns=['Chemical Class', 'Count'])
side_effect = get_info_element(medecins_df, 'sideEffect', count=True)
side_effect_df = pd.DataFrame(list(side_effect.items()), columns=['Side Effect', 'Count'])
use = get_info_element(medecins_df, 'use', count=True)
use_df = pd.DataFrame(list(use.items()), columns=['Type of uses', 'Count'])
habit = get_info_element(medecins_df, 'Habit Forming', count=True)
habit_df = pd.DataFrame(list(habit.items()), columns=['Habit Forming', 'Count'])
therapeutic = get_info_element(medecins_df, 'Therapeutic Class', count=True)
therapeutic_df = pd.DataFrame(list(therapeutic.items()), columns=['Therapeutic Class', 'Count'])
print(side_effect_df.head())

def build_medicine_graph(df: pd.DataFrame, max_nodes: int = 200):
    """Build a Plotly network figure linking medicines by shared side effects."""
    # detect sideEffect columns
    side_cols = [c for c in df.columns if c.startswith('sideEffect')]
    # choose a name column (first column that is not a known feature)
    feature_prefixes = ['Chemical Class','sideEffect','use','Habit Forming','Therapeutic Class']
    name_col = next((c for c in df.columns if not any(c.startswith(p) for p in feature_prefixes)), df.columns[0])

    # collect side effects per medicine (as sets)
    med_side = {}
    for _, row in df.iterrows():
        name = str(row[name_col]) if pd.notna(row[name_col]) else f"row_{_}"
        se_vals = []
        for c in side_cols:
            if c in df.columns and pd.notna(row[c]):
                # if a cell contains multiple items separated by commas/semicolons, split them
                parts = [p.strip() for p in str(row[c]).replace(';',',').split(',') if p.strip()]
                se_vals.extend(parts)
        se_set = set(se_vals)
        if se_set:
            med_side[name] = se_set

    if not med_side:
        # empty figure with message
        fig = go.Figure()
        fig.add_annotation(text="No side-effect data found to build graph.", showarrow=False)
        return fig

    # limit number of medicines to avoid huge graphs
    meds_sorted = sorted(med_side.items(), key=lambda kv: len(kv[1]), reverse=True)
    meds_sorted = meds_sorted[:max_nodes]
    meds = [m for m, s in meds_sorted]
    sets = {m: s for m, s in meds_sorted}

    # build graph where edge exists if two medicines share >=1 side effect
    G = nx.Graph()
    for m in meds:
        G.add_node(m, side_effects=", ".join(sorted(sets[m])), nside=len(sets[m]))
    for i in range(len(meds)):
        for j in range(i+1, len(meds)):
            a, b = meds[i], meds[j]
            shared = sets[a].intersection(sets[b])
            if shared:
                G.add_edge(a, b, weight=len(shared), shared=", ".join(sorted(shared)))

    if G.number_of_edges() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No shared side effects between selected medicines.", showarrow=False)
        return fig

    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=100)

    edge_x = []
    edge_y = []
    edge_widths = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_widths.append(data.get('weight', 1))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for n, data in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{n}<br>Side effects ({data.get('nside',0)}): {data.get('side_effects')}")
        node_size.append(10 + 6 * data.get('nside', 0))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[data.get('nside',0) for _, data in G.nodes(data=True)],
            size=node_size,
            colorbar=dict(title='Number of side effects'),
            line_width=1
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Medicine similarity by shared side effects (top {len(meds)} medicines)',
                        title_x=0.5,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig

# build graph figure (limit nodes to 200 by default)
med_graph_fig = build_medicine_graph(medecins_df, max_nodes=200)


# Requires Dash 2.17.0 or later
app.layout = [
    html.Main(children=[
        html.H1('Medecine Chemical Classifications'),
        dash_table.DataTable(
            id='medecins-table',
            columns=[{"name": i, "id": i} for i in chemical_df.columns],
            data=chemical_df.to_dict('records'),
            page_size=10,
            sort_action='native',
        ),
        # Network graph linking medicines by shared side effects
        html.Div(children=[
            html.H2("Medicines linked by shared side effects"),
            dcc.Graph(id='med-sideeffect-graph', figure=med_graph_fig)
        ]),
        html.Div(children=[
            dcc.Graph(figure=px.histogram(chemical_df, x='Chemical Class', y='Count', title='Distribution of Chemical Classifications'))
        ]),
        html.Div(className='six columns', children=[
            dcc.Graph(figure=px.histogram(side_effect_df, x='Side Effect', y='Count', title='Distribution of Side Effects'))
        ]),
        html.Div(className='six columns', children=[
            dcc.Graph(figure=px.histogram(use_df, x='Type of uses', y='Count', title='Distribution of Types of Uses'))
        ]),
        html.Div(className='six columns', children=[
            dcc.Graph(figure=px.histogram(habit_df, x='Habit Forming', y='Count', title='Distribution of Habit Forming'))
        ]),
        html.Div(className='six columns', children=[
            dcc.Graph(figure=px.histogram(therapeutic_df, x='Therapeutic Class', y='Count', title='Distribution of Therapeutic Classes').update_xaxes(tickangle=45, categoryorder='total descending'))
        ])
    ])
]

if __name__ == '__main__':
    app.run(debug=True)