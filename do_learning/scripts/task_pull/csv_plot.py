import pandas as pd
import plotly.graph_objects as go
import numpy as np
import argparse

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
            family="Courier New, monospace",
            size=20,
            color="RebeccaPurple"
        ),
        plot_bgcolor="rgb(255,255,255)"
    )
    fig.show()

def plot_generalize_capability():
    x = ['env 0', 'env 1', 'env 2', 'env 3', 'env 4', 'env 5', 'env 6']
    y1 = [100, 55, 20, 7, 5, 0, 100]
    y2 = [98, 93, 11, 73, 42, 0, 100]
    y3 = [100, 98, 100, 100, 83, 53, 100]

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
            family="Courier New, monospace",
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
    else:
        plot_generalize_capability()
