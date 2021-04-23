"""
Goal: being able to generate fake NPS one at a time, using Eric's code.

"""
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np

import pandas as pd
from UltraCold.MTF import make_M2k_Fit

df = pd.read_csv(
    'https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv'
)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#                   A ,  tau,   S0, alpha,  phi, beta, delta_s
paras_bounds = ([-1000, 0.5, -20, -20, -np.pi, -30,
                 -np.pi], [1000, 10, 20, 20, np.pi, 30, np.pi])


def make_slider(name, min, max, value=0):
    marks = {
        0: str(0),
        min: f"{min:.2f}",
        max: f"{max:.2f}",
    }
    for v in np.linspace(min, max, 11):
        marks.setdefault(v, f"{v:.2f}")

    f = (lambda v, name=name: name + f" = {v:.2f}")
    app.callback(
        Output(name + '-output-container', 'children'),
        Input(name + '-slider', 'value'),
    )(f)

    f = (lambda v: v)
    app.callback(
        Output(name + '-slider', 'value'),
        Input(name + '_input', 'value'),
    )(f)

    return (
        html.Div(id=name + '-output-container',
                 children=name,
                 style={'font-size': 'x-large'}),
        html.Div([
            dcc.Slider(
                id=name + '-slider',
                min=float(min),
                max=float(max),
                value=value,
                step=0.01,
                marks=marks,
                updatemode='drag',
            ),
            dcc.Input(
                id=name + '_input',
                type="number",
                value=value,
                step=0.1,
            )
        ]),
        html.Div(children="_" * 50, ),
    )


app.layout = html.Div(
    [
        html.Title('MTF Playground'),
        # html.Div("Playground", style={"font-size": "xx-large"}),
        dcc.Graph(
            id='graph-with-slider',
            style={
                'left': '0px',
                'width': '70%',
                #  'display': 'unset';
            }),
        html.Div([]),
        html.Div(children=[
            *make_slider('A', -10, 10, 1),
            *make_slider('tau', -0.5, 10, 0.8),
            *make_slider('S0', -20, 20, 0),
            *make_slider('alpha', -20, 20, 1),
            *make_slider('phi', -np.pi, np.pi, 1.5),
            *make_slider('beta', -30, 30, 1),
            *make_slider('delta_s', -np.pi, np.pi, 3),
        ],
                 style={
                     'right': '0px',
                     'width': '30%'
                 })
    ],
    style={'display': 'flex'},
)


@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('A-slider', 'drag_value'),
    Input('tau-slider', 'drag_value'),
    Input('S0-slider', 'drag_value'),
    Input('alpha-slider', 'drag_value'),
    Input('phi-slider', 'drag_value'),
    Input('beta-slider', 'drag_value'),
    Input('delta_s-slider', 'drag_value'),
)
def update_figure(*para):
    print(para)
    # para_guess = [1, 1, 1, 1, 1.5, 1, 3]

    fig = px.imshow(
        make_M2k_Fit(para, None),
        -1,
        1,
        color_continuous_scale='RdBu_r',
    )

    fig.update_layout(transition_duration=100)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)