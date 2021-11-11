from abc import abstractproperty

import dash
import plotly
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output

from components.armRecognizer import RecogArm

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Capstone2: Arm Recognition"
app.layout = html.Div(
    html.Div(
        [
            html.Div(html.H3("CAPSTONE2-HYSJ"), className="container"),
            html.Div(
                [
                    html.Div(
                        dcc.Graph(id="live-update-image"), className="six columns"
                    ),
                    html.Div(
                        dcc.Graph(id="live-update-graph"), className="six columns"
                    ),
                ],
                className="container",
            ),
            # dcc.Graph(id="live-update-image"),
            # dcc.Graph(id="live-update-graph"),
            html.Div(id="live-update-text", className="container"),
            dcc.Interval(
                id="interval-component",
                interval=200,  # in milliseconds
                n_intervals=0,
            ),
        ]
    )
)
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

ARM = RecogArm(isRightArm=True)


@app.callback(
    Output("live-update-image", "figure"),
    Output("live-update-graph", "figure"),
    Output("live-update-text", "children"),
    Input("interval-component", "n_intervals"),
)
def updateChildren(n):
    image, df = ARM.getMyArmsImangeAndDF()
    imFig = px.imshow(image)
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        size="vis",
        color="vis",
        text="label",
        # height=800,
    )
    camera = dict(eye=dict(x=0.0, y=0.0, z=2.5))
    scene = dict(
        xaxis=dict(range=[-0.5, 0.5]),
        yaxis=dict(range=[-0.5, 0.5]),
        zaxis=dict(range=[-0.5, 0.5]),
        aspectmode="cube",
        aspectratio=dict(x=1, y=1, z=1),
    )
    fig.update_layout(
        dict(
            scene=scene,
            scene_camera=camera,
        ),
    )
    isVis = all([i > 0.9 for i in df["vis"]])
    text = [html.Span(f"{isVis=}")]

    return imFig, fig, text


# @app.callback(
#     Output("live-update-text", "children"), Input("interval-component", "n_intervals")
# )
# def update_metrics(n):
#     style = {"padding": "5px", "fontSize": "16px", "text-align": "center"}

#     _, df = ARM.getMyArmsImangeAndDF()
#     isVisible = all([i > 0.8 for i in df["vis"]])

#     return [
#         html.Span(f"{isVisible=}", style=style),
#     ]


# @app.callback(
#     Output("live-update-graph", "figure"), Input("interval-component", "n_intervals")
# )
# def update_graph_live(n):

#     # image, df = ARM.getMyArmsImangeAndDF()
#     image = CURR_IMAGE
#     fig = px.imshow(image)

#     return fig


# @app.callback(
#     Output("live-update-graph2", "figure"), Input("interval-component", "n_intervals")
# )
# def update_graph_live2(n):
#     # image, df = ARM.getMyArmsImangeAndDF()
#     df = CURR_DF

#     fig = px.scatter_3d(
#         df,
#         x="x",
#         y="y",
#         z="z",
#         size="vis",
#         text="label",
#         height=800,
#     )
#     camera = dict(eye=dict(x=0.0, y=0.0, z=2.5))
#     scene = dict(
#         xaxis=dict(range=[-0.5, 0.5]),
#         yaxis=dict(range=[-0.5, 0.5]),
#         zaxis=dict(range=[-0.5, 0.5]),
#         aspectmode="cube",
#         aspectratio=dict(x=1, y=1, z=1),
#     )
#     fig.update_layout(
#         dict(
#             scene=scene,
#             scene_camera=camera,
#         ),
#     )

#     return fig


if __name__ == "__main__":
    app.run_server(debug=True)
