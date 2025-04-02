import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
# CSV 파일 읽기m
try:
    df = pd.read_csv('./MHPH05/Data/Mody/240625/Mody_T2_M_HPH_05.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('./MHPH05/Data/Mody/240625/Mody_T2_M_HPH_05.csv', encoding='cp949')  # 한글 파일의 경우 cp949로 시도

# 빈칸을 바로 위 행의 값으로 채웁니다.
df.fillna(method='ffill', inplace=True)

# 4, 6, 8, 13 열의 데이터를 가져옵니다.
columns_to_plot = [df.columns[1], df.columns[2],df.columns[3], df.columns[7], df.columns[8], df.columns[9], df.columns[10], df.columns[12]]

# Dash 애플리케이션을 초기화합니다.
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Interactive Data Visualization with Dual Axes"),
    html.Div([
        html.Div([
            html.Label("Select Columns for Primary Y-Axis"),
            dcc.Dropdown(
                id='primary-y-dropdown',
                options=[{'label': col, 'value': col} for col in columns_to_plot],
                value=[columns_to_plot[0]],
                multi=True
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.Label("Select Columns for Secondary Y-Axis"),
            dcc.Dropdown(
                id='secondary-y-dropdown',
                options=[{'label': col, 'value': col} for col in columns_to_plot],
                value=[columns_to_plot[1]],
                multi=True
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
    ]),
    dcc.Graph(id='dual-axis-plot')
])


@app.callback(
    Output('dual-axis-plot', 'figure'),
    [Input('primary-y-dropdown', 'value'),
     Input('secondary-y-dropdown', 'value')]
)
def update_graph(primary_y_columns, secondary_y_columns):
    traces = []

    for column in primary_y_columns:
        traces.append(go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=column,
            yaxis='y1'
        ))

    for column in secondary_y_columns:
        traces.append(go.Scatter(
            x=df.index,
            y=df[column],
            mode='lines',
            name=column,
            yaxis='y2'
        ))

    layout = go.Layout(
        title='Data from Selected Columns with Dual Axes',
        xaxis={'title': 'Index'},
        yaxis=dict(
            title='Primary Y-Axis',
            side='left'
        ),
        yaxis2=dict(
            title='Secondary Y-Axis',
            side='right',
            overlaying='y'
        )
    )

    return {'data': traces, 'layout': layout}


# 애플리케이션을 실행합니다.
if __name__ == '__main__':
    app.run_server(debug=True)
