import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import random


def plot_data(X, col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X.index, y=X[col], mode='lines', name=col))
    fig.update_layout(height=400, width=800, showlegend=True)
    fig.show()


def plot_anomaly(X_test, y_test, col):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test.index, y=X_test[col], mode='lines', name='y_test'))
    fig.add_trace(go.Scatter(x=y_test[y_test['label'] == 1].index, y=X_test[y_test['label'] == 1][col], mode='markers', name='Anomaly'))
    fig.update_layout(showlegend=True, xaxis_title="Time", yaxis_title="value", height=400, width=800)
    fig.show()