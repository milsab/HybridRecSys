from sklearn.cluster import KMeans
import utils


def get_kmeans_clusters(embeddings, K):

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=K, random_state=0).fit(embeddings)

    # Get cluster labels for each node
    labels = kmeans.labels_

    return labels