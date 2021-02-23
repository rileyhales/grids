import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('speed_test_times.csv')
df.index = df['engine']
layout = go.Layout(
    title='Time Series Extraction Time From Web Service',
    yaxis={'title': 'Time per File (Seconds)'},
    xaxis={'title': 'Series Type'},
    barmode='group',
)

x_categories = ["Point", "Bounding Box", "Shape", "Array Stats"]
y_values = [1.145756894, 1.136036864, 1.134677644, 1.087190121]
bars = [
    go.Bar(name='Point', x=x_categories, y=y_values),
]
fig = go.Figure(data=bars, layout=layout)
fig.show()
fig.write_image('remote_speeds.svg')
