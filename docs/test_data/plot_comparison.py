import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('speed_test_times.csv')
df.index = df['engine']
del df['engine'], df['point'], df['range'], df['shape'], df['count'],
x = ['Point Series', 'Bounding Box Series', 'Polygon Series']
layout = go.Layout(
    title='Time Series Extraction Time By File Engine and Series Type',
    yaxis={'title': 'Time per File in Series (seconds)'},
    xaxis={'title': 'Series Type'},
    barmode='group',
)
bars = [
    go.Bar(name='Point', x=df.index, y=df['point/file'].values),
    go.Bar(name='Bounding Box', x=df.index, y=df['range/file'].values),
    go.Bar(name='Polygon', x=df.index, y=df['shape/file'].values)
]
fig = go.Figure(data=bars, layout=layout)
fig.show()
fig.write_image('total_comparison.svg')
