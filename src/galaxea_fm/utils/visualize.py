import plotly.graph_objects as go
import numpy as np

PALETTE = [
    "rgb(0, 255, 0)",
    "rgb(0, 191, 0)",
    "rgb(0, 128, 0)", 
    "rgb(0, 64, 0)", 
    "rgb(0, 0, 0)", 
]


def plot_result(path, gt, pd):
    episode_size, chunk_size, dim = pd.shape
    for d in range(dim):
        fig = go.Figure()
        for t in range(episode_size):
            color_idx = t % len(PALETTE)
            fig.add_trace(go.Scatter(
                x=np.arange(t, t + chunk_size), 
                y=pd[t, :, d], 
                line=dict(color=PALETTE[color_idx]), 
                name=f"pd group {color_idx}", 
                legendgroup=f"pd group {color_idx}", 
                showlegend=t < len(PALETTE), 
            ))
        fig.add_trace(go.Scatter(x=np.arange(episode_size), y=gt[:, d], name="gt", line=dict(color="red")))
        fig.write_html(path / f"{d:02}.html")

    print("Result plot save to", path)