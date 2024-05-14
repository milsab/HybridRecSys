import mlflow
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import utils


def plot_tSNE(embeddings, kmeans_labels):
    """
    Visualizing embeddings using tSNE dimension reduction technique
    :param embeddings:
    :param kmeans_labels:
    :return: 2D tSNE plot of embeddings
    """

    tsne = TSNE(n_components=3, verbose=0, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)

    # Interactive Plot with PLOTLY
    fig = px.scatter(x=tsne_results[:, 0], y=tsne_results[:, 1], color=kmeans_labels.astype(str),
                     labels={'color': 'Cluster'}, title='Interactive t-SNE Clusters')
    fig.update_traces(marker=dict(size=5, opacity=0.6))
    fig.show()
    fig.write_html('plot_images/tSNE_plot.html')
    mlflow.log_artifact('plot_images/tSNE_plot.html')


def plot_PCA(embeddings, kmeans_labels, num_users, num_clusters, run_name):
    """
    Visualizing embeddings using PCA dimension reduction technique
    :param embeddings:
    :param kmeans_labels:
    :return: 2D PCA plot of embeddings
    """

    combined_labels = kmeans_labels.copy()

    # Offset item labels by the number of clusters
    combined_labels[num_users:] += num_clusters

    # Define colors for each combined category
    colors = [
        'rgba(255, 128, 255, 0.8)', 'rgba(255, 128, 128, 0.8)', 'rgba(128, 255, 128, 0.8)', 'rgba(128, 128, 255, 0.8)',
        'rgba(255, 255, 128, 0.8)',
        'rgba(255, 128, 64, 0.8)', 'rgba(64, 128, 255, 0.8)', 'rgba(128, 64, 128, 0.8)', 'rgba(64, 255, 128, 0.8)',
        'rgba(128, 128, 64, 0.8)'
    ]

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    # Create traces
    data = []

    for i, color in enumerate(colors):
        # Select embeddings by combined label
        indices = np.where(combined_labels == i)[0]
        trace = go.Scatter(
            x=pca_result[indices, 0],
            y=pca_result[indices, 1],
            mode='markers',
            marker=dict(color=color),
            name=f'Users in Cluster {i}' if i < 5 else f'Items in Cluster {i - 5}'
        )
        data.append(trace)

    # Create the layout
    layout = go.Layout(
        title=f'{run_name} - PCA visualization of User and Item Embeddings with Cluster Info',
        xaxis=dict(title='Component 1'),
        yaxis=dict(title='Component 2'),
        hovermode='closest'
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()
    fig.write_html(f'plot_images/{run_name}_PCA_plot.html')
    mlflow.log_artifact(f'plot_images/{run_name}_PCA_plot.html')
