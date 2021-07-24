import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import argparse
import csv

def smoothTriangle(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]
    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed

def smoothExponential(data, weight):
    last = data[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

def plot_training_performance():
    force_only = './images/run-force-only-tag-episode total reward.csv'
    force_vision = './images/run-force-vision-tag-episode total reward.csv'
    vision_single = './images/run-vision-single-tag-episode total reward.csv'
    vision_multiple = './images/run-vision-multiple-tag-episode total reward.csv'
    df0 = pd.read_csv(force_only)
    df1 = pd.read_csv(vision_single)
    df2 = pd.read_csv(vision_multiple)
    df3 = pd.read_csv(force_vision)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df0['Step'], y = smoothExponential(df0['Value'],0.996),name='force-only input'))
    fig.add_trace(go.Scatter(x = df1['Step'], y = smoothExponential(df1['Value'],0.996),name='single-camera input'))
    fig.add_trace(go.Scatter(x = df2['Step'], y = smoothExponential(df2['Value'],0.996),name='multi-camera fusion'))
    fig.add_trace(go.Scatter(x = df3['Step'], y = smoothExponential(df3['Value'],0.996),name='force-vision fusion'))
    fig.update_layout(
        title="Episodic Total Reward",
        xaxis_title="Episodes",
        yaxis_title="Total Reward",
        legend_title="PPO Policies",
        font=dict(
            family="Arial",
            size=20,
            color="RebeccaPurple"
        ),
        plot_bgcolor="rgb(255,255,255)"
    )
    fig.show()

def plot_generalize_capability():
    x = ['env 0', 'env 1', 'env 2', 'env 3', 'env 4', 'env 5', 'env 6', 'env 7', 'env 8']
    y1 = [100, 55, 20, 7, 5, 0, 100, 100, 87]
    y2 = [98, 93, 11, 73, 42, 0, 100, 98, 55]
    y3 = [100, 98, 100, 100, 83, 53, 100, 98, 92]

    fig = go.Figure()
    fig.add_trace(go.Bar(x = x, y = y1, name='single-camera input'))
    fig.add_trace(go.Bar(x = x, y = y2, name='multi-camera fusion'))
    fig.add_trace(go.Bar(x = x, y = y3, name='force-vision fusion'))
    fig.update_layout(
        title="Policy Generalization",
        xaxis_title="Environments",
        yaxis_title="Success Rate (%)",
        legend_title="PPO Policies",
        font=dict(
            family="Arial",
            size=20,
            color="RebeccaPurple"
        ),
        plot_bgcolor="rgb(255,255,255)"
    )
    fig.show()

def plot_force_profile():
    force_sc = './images/env0-sc/force11.csv'
    force_mc = './images/env0-mc/force1.csv'
    force_fv = './images/env0-fv/force25.csv'
    dfsc = pd.read_csv(force_sc)
    dfmc = pd.read_csv(force_mc)
    dffv = pd.read_csv(force_fv)
    #
    fig = make_subplots(rows=3,cols=1)
    fig.add_trace(go.Scatter(x = dfsc['index']/100, y = dfsc['x'],name='single-camera input', legendgroup="single-camera input", marker=dict(color='red')), row=1,col=1)
    fig.add_trace(go.Scatter(x = dfmc['index']/100, y = dfmc['x'],name='multi-camera fusion', legendgroup="multi-camera fusion", marker=dict(color='blue')), row=1,col=1)
    fig.add_trace(go.Scatter(x = dffv['index']/100, y = dffv['x'],name='force-vision fusion', legendgroup="force-vision fusion", marker=dict(color='green')), row=1,col=1)

    fig.add_trace(go.Scatter(x = dfsc['index']/100, y = dfsc['y'],name='single-camera input', legendgroup="single-camera input", marker=dict(color='red'), showlegend=False), row=2,col=1)
    fig.add_trace(go.Scatter(x = dfmc['index']/100, y = dfmc['y'],name='multi-camera fusion', legendgroup="multi-camera fusion", marker=dict(color='blue'), showlegend=False), row=2,col=1)
    fig.add_trace(go.Scatter(x = dffv['index']/100, y = dffv['y'],name='force-vision fusion', legendgroup="force-vision fusion", marker=dict(color='green'), showlegend=False), row=2,col=1)

    fig.add_trace(go.Scatter(x = dfsc['index']/100, y = dfsc['z'],name='single-camera input', legendgroup="single-camera input", marker=dict(color='red'), showlegend=False), row=3,col=1)
    fig.add_trace(go.Scatter(x = dfmc['index']/100, y = dfmc['z'],name='multi-camera fusion', legendgroup="multi-camera fusion", marker=dict(color='blue'), showlegend=False), row=3,col=1)
    fig.add_trace(go.Scatter(x = dffv['index']/100, y = dffv['z'],name='force-vision fusion', legendgroup="force-vision fusion", marker=dict(color='green'), showlegend=False), row=3,col=1)

    fig.update_xaxes(title_text="time (s)", row=1,col=1)
    fig.update_xaxes(title_text="time (s)", row=2,col=1)
    fig.update_xaxes(title_text="time (s)", row=3,col=1)

    fig.update_yaxes(title_text="X-Axis Force (N)", row=1,col=1)
    fig.update_yaxes(title_text="Y-Axis Force (N)", row=2,col=1)
    fig.update_yaxes(title_text="Z-Axis Force (N)", row=3,col=1)

    fig.update_layout(
        title="Force Curve",
        legend_title="PPO Policies",
        font=dict(
            family="Arial",
            size=20,
            color="RebeccaPurple"
        ),
        plot_bgcolor="rgb(255,255,255)"
    )
    fig.show()

def plot_env_force_profile():
    force0 = './images/env0-fv/force25.csv'
    force1 = './images/env1-fv/force0.csv'
    force2 = './images/env2-fv/force0.csv'
    force3 = './images/env3-fv/force0.csv'
    force4 = './images/env4-fv/force0.csv'
    force5 = './images/env5-fv/force0.csv'
    force6 = './images/env6-fv/force0.csv'
    force7 = './images/env8-fv/force0.csv'
    force8 = './images/env10-fv/force46.csv'

    env0 = pd.read_csv(force0)
    env1 = pd.read_csv(force1)
    env2 = pd.read_csv(force2)
    env3 = pd.read_csv(force3)
    env4 = pd.read_csv(force4)
    env5 = pd.read_csv(force5)
    env6 = pd.read_csv(force6)
    env7 = pd.read_csv(force7)
    env8 = pd.read_csv(force8)

    #
    fig = make_subplots(rows=3,cols=1)
    fig.add_trace(go.Scatter(x = env0['index']/100, y = env0['x'],name='env 0', legendgroup="env 0", marker=dict(color='#ff7fc5')), row=1,col=1)
    fig.add_trace(go.Scatter(x = env1['index']/100, y = env1['x'],name='env 1', legendgroup="env 1", marker=dict(color='#074632')), row=1,col=1)
    fig.add_trace(go.Scatter(x = env2['index']/100, y = env2['x'],name='env 2', legendgroup="env 2", marker=dict(color='#6dcff6')), row=1,col=1)
    fig.add_trace(go.Scatter(x = env3['index']/100, y = env3['x'],name='env 3', legendgroup="env 3", marker=dict(color='#d2a828')), row=1,col=1)
    fig.add_trace(go.Scatter(x = env4['index']/100, y = env4['x'],name='env 4', legendgroup="env 4", marker=dict(color='#6495ed')), row=1,col=1)
    fig.add_trace(go.Scatter(x = env5['index']/100, y = env5['x'],name='env 5', legendgroup="env 5", marker=dict(color='#bb6336')), row=1,col=1)
    fig.add_trace(go.Scatter(x = env6['index']/100, y = env6['x'],name='env 6', legendgroup="env 6", marker=dict(color='#2c3644')), row=1,col=1)
    fig.add_trace(go.Scatter(x = env7['index']/100, y = env7['x'],name='env 7', legendgroup="env 7", marker=dict(color='#e53511')), row=1,col=1)
    fig.add_trace(go.Scatter(x = env8['index']/100, y = env8['x'],name='env 8', legendgroup="env 8", marker=dict(color='#757557')), row=1,col=1)

    fig.add_trace(go.Scatter(x = env0['index']/100, y = env0['y'],name='env 0', legendgroup="env 0", marker=dict(color='#ff7fc5'), showlegend=False), row=2,col=1)
    fig.add_trace(go.Scatter(x = env1['index']/100, y = env1['y'],name='env 1', legendgroup="env 1", marker=dict(color='#074632'), showlegend=False), row=2,col=1)
    fig.add_trace(go.Scatter(x = env2['index']/100, y = env2['y'],name='env 2', legendgroup="env 2", marker=dict(color='#6dcff6'), showlegend=False), row=2,col=1)
    fig.add_trace(go.Scatter(x = env3['index']/100, y = env3['y'],name='env 3', legendgroup="env 3", marker=dict(color='#d2a828'), showlegend=False), row=2,col=1)
    fig.add_trace(go.Scatter(x = env4['index']/100, y = env4['y'],name='env 4', legendgroup="env 4", marker=dict(color='#6495ed'), showlegend=False), row=2,col=1)
    fig.add_trace(go.Scatter(x = env5['index']/100, y = env5['y'],name='env 5', legendgroup="env 5", marker=dict(color='#bb6336'), showlegend=False), row=2,col=1)
    fig.add_trace(go.Scatter(x = env6['index']/100, y = env6['y'],name='env 6', legendgroup="env 6", marker=dict(color='#2c3644'), showlegend=False), row=2,col=1)
    fig.add_trace(go.Scatter(x = env7['index']/100, y = env7['y'],name='env 7', legendgroup="env 7", marker=dict(color='#e53511'), showlegend=False), row=2,col=1)
    fig.add_trace(go.Scatter(x = env8['index']/100, y = env8['y'],name='env 8', legendgroup="env 8", marker=dict(color='#757557'), showlegend=False), row=2,col=1)

    fig.add_trace(go.Scatter(x = env0['index']/100, y = env0['z'],name='env 0', legendgroup="env 0", marker=dict(color='#ff7fc5'), showlegend=False), row=3,col=1)
    fig.add_trace(go.Scatter(x = env1['index']/100, y = env1['z'],name='env 1', legendgroup="env 1", marker=dict(color='#074632'), showlegend=False), row=3,col=1)
    fig.add_trace(go.Scatter(x = env2['index']/100, y = env2['z'],name='env 2', legendgroup="env 2", marker=dict(color='#6dcff6'), showlegend=False), row=3,col=1)
    fig.add_trace(go.Scatter(x = env3['index']/100, y = env3['z'],name='env 3', legendgroup="env 3", marker=dict(color='#d2a828'), showlegend=False), row=3,col=1)
    fig.add_trace(go.Scatter(x = env4['index']/100, y = env4['z'],name='env 4', legendgroup="env 4", marker=dict(color='#6495ed'), showlegend=False), row=3,col=1)
    fig.add_trace(go.Scatter(x = env5['index']/100, y = env5['z'],name='env 5', legendgroup="env 5", marker=dict(color='#bb6336'), showlegend=False), row=3,col=1)
    fig.add_trace(go.Scatter(x = env6['index']/100, y = env6['z'],name='env 6', legendgroup="env 6", marker=dict(color='#2c3644'), showlegend=False), row=3,col=1)
    fig.add_trace(go.Scatter(x = env7['index']/100, y = env7['z'],name='env 7', legendgroup="env 7", marker=dict(color='#e53511'), showlegend=False), row=3,col=1)
    fig.add_trace(go.Scatter(x = env8['index']/100, y = env8['z'],name='env 8', legendgroup="env 8", marker=dict(color='#757557'), showlegend=False), row=3,col=1)

    fig.update_xaxes(title_text="time (s)", row=1,col=1)
    fig.update_xaxes(title_text="time (s)", row=2,col=1)
    fig.update_xaxes(title_text="time (s)", row=3,col=1)

    fig.update_yaxes(title_text="X-Axis Force (N)", row=1,col=1)
    fig.update_yaxes(title_text="Y-Axis Force (N)", row=2,col=1)
    fig.update_yaxes(title_text="Z-Axis Force (N)", row=3,col=1)

    fig.update_layout(
        title="Force Curve",
        legend_title="PPO Policies",
        font=dict(
            family="Arial",
            size=20,
            color="RebeccaPurple"
        ),
        plot_bgcolor="rgb(255,255,255)"
    )
    fig.show()



# main loop
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default='train') # 'train', 'generalize'
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.graph == 'train':
        plot_training_performance()
    elif args.graph == 'force':
        plot_force_profile()
    elif args.graph == 'env':
        plot_env_force_profile()
    else:
        plot_generalize_capability()
