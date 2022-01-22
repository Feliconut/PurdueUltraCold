"""
Goal: being able to generate fake NPS one at a time, using Eric's code.

"""
# from functools import lru_cache
from functools import lru_cache
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from dash_extensions import Monitor
from scipy.sparse import data

from UltraCold.MTF import make_M2k_Fit

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#                   A ,  tau,   S0, alpha,  phi, beta, delta_s
paras_bounds = ([-1000, 0.5, -20, -20, -np.pi, -30,
                 -np.pi], [1000, 10, 20, 20, np.pi, 30, np.pi])


def make_slider(name, min, max, value=0):
    slider_id = name + '-slider'
    input_id = name + '_input'
    monitor_id = name + '_monitor'

    marks = {
        0: str(0),
        min: f"{min:.2f}",
        max: f"{max:.2f}",
    }

    def sync_inputs(data):
        # Get value and trigger id from monitor.
        try:
            probe = data["val"]
            trigger_id, value = probe["trigger"]["id"], float(probe["value"])
        except (TypeError, KeyError):
            print(1)
            raise PreventUpdate
        # Do the appropriate update.
        print(2)
        # return value,value
        if trigger_id == slider_id:
            return dash.no_update, value
        elif trigger_id == input_id:
            return value, dash.no_update

    app.callback(
        Output(slider_id, "value"),
        Output(input_id, "value"),
        # Output(name + '-output-container', 'children')
        Input(monitor_id, "data"))(sync_inputs)

    return (
        html.Div(id=name + '-output-container',
                 children=name,
                 style={'font-size': 'x-large'}),
        Monitor(
            [
                html.Div([
                    dcc.Slider(
                        id=slider_id,
                        min=float(min),
                        max=float(max),
                        value=value,
                        step=0.01,
                        marks=marks,
                        updatemode='drag',
                    ),
                    dcc.Input(
                        id=input_id,
                        type="number",
                        value=value,
                        step=0.01,
                    )
                ])
            ],
            id=monitor_id,
            probes=dict(val=[
                dict(id=slider_id, prop="drag_value"),
                dict(id=input_id, prop="value")
            ]),
        ),
        html.Div(children="_" * 50, ),
    )


ID_GRAPH_FIT = 'fit_graph_id'
ID_GRAPH_EXP = 'exp_graph_id'

app.layout = html.Div(
    [
        html.Title('MTF Playground'),
        # html.Div("Playground", style={"font-size": "xx-large"}),

        html.Div([
            dcc.Graph(
            id=ID_GRAPH_FIT,
            style={
                'left': '0px',
                'width': '70%',
                'height':'50%',
                #  'display': 'unset';
            }),
            dcc.Graph(
            id=ID_GRAPH_EXP,
            style={
                'left': '0px',
                'width': '70%',
                'height':'50%',
                #  'display': 'unset';
            }),
            ]
                             ,style={
                    #  'right': '0px',
                     'width': '100%',
                    #  'height':'50%',

                 }
                 ),
            # html.Div([]),
        html.Div(children=[
            *make_slider('A', 0, 30, 1),\
            *make_slider('tau', 0.2, 2, 0.8),\
            *make_slider('S0', -20, 20, 0),\
            *make_slider('alpha', -20, 20, 0),\
            *make_slider('phi', -np.pi, np.pi, 0),\
            *make_slider('beta', -30, 30, 0),\
            *make_slider('delta_s', -np.pi, np.pi, 0),\
            dcc.Input(
                        id='dataset-id-input',
                        type="text",
                        value="dataset_id",
                    ),\

            html.Div(id='params')
        ],
                 style={
                     'right': '0px',
                     'width': '30%',
                    #  'height':'50%',

                 }
                 )
    ],
    style={'display': 'flex'},
)

# cached_make_M2k_Fit = lru_cache(None)(make_M2k_Fit)


@app.callback(
    Output(ID_GRAPH_FIT, 'figure'),
    Output('params', 'children'),
    Input('A-slider', 'drag_value'),
    Input('tau-slider', 'drag_value'),
    Input('S0-slider', 'drag_value'),
    Input('alpha-slider', 'drag_value'),
    Input('phi-slider', 'drag_value'),
    Input('beta-slider', 'drag_value'),
    Input('delta_s-slider', 'drag_value'),
)
def update_figure(*fit_param):
    print(fit_param)
    # para_guess = [1, 1, 1, 1, 1.5, 1, 3]

    fig = px.imshow(
        make_M2k_Fit(fit_param, ),
        zmin=0,  # vmin
        zmax=fit_param[0],  # vmax = A
        # color_continuous_scale='RdBu_r',
        color_continuous_scale='jet',
    )

    fig.update_layout()
    param_text = repr(fit_param)

    return fig, param_text

@lru_cache(maxsize=256)
def get_mtf(dataset_id):
    from UltraCold import OD, MTF
    return MTF.from_ods(OD.from_dataset(dataset_id))

@app.callback(
    Output(ID_GRAPH_EXP, 'figure'),
    Input('dataset-id-input', 'value'),
    Input('A-slider', 'drag_value'),
)
def update_exp_figure(dataset_id, A):
    try:
        MTF_Exp =  get_mtf(dataset_id)
        fig = px.imshow(
            MTF_Exp,
            zmin=0,  # vmin
            zmax=A,  # vmax = A
            # color_continuous_scale='RdBu_r',
            color_continuous_scale='jet',
        )

        # fig.update_layout()
        return fig

    except FileNotFoundError:
        return dash.no_update
    
        

# TODO Export fitting parameters

# TODO click to fit


if __name__ == '__main__':
    app.run_server(debug=True)
