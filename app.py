from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
import pandas as pd
import plotly.express as px
import json 

app = Flask(__name__)


global_data = None


def generate_data():
    return np.random.rand(300, 2) * 20 - 10


def farthest_first_initialization(X, n_clusters):
    centroids = []
    centroids.append(X[np.random.randint(X.shape[0])])
    for _ in range(1, n_clusters):
        dist = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in X])
        next_centroid = X[np.argmax(dist)]
        centroids.append(next_centroid)
    return np.array(centroids)

# KMeans++ Initialization
def kmeans_plus_plus_initialization(X, n_clusters):
    n_samples, _ = X.shape
    centroids = []

    first_centroid_idx = np.random.randint(n_samples)
    centroids.append(X[first_centroid_idx])

    for _ in range(1, n_clusters):
        dist_sq = np.array([min([np.sum((x - c) ** 2) for c in centroids]) for x in X])
        probabilities = dist_sq / dist_sq.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        next_centroid_idx = np.searchsorted(cumulative_probabilities, r)
        centroids.append(X[next_centroid_idx])

    return np.array(centroids)

def kmeans_with_iterations(data, n_clusters, init_method, run_to_convergence=False):
    if isinstance(init_method, str):
        if init_method == 'random':
            centroids = data[np.random.choice(data.shape[0], n_clusters, replace=False)]
        elif init_method == 'k-means++':
            centroids = kmeans_plus_plus_initialization(data, n_clusters)
    else:
        centroids = init_method

    centroids_list = [centroids.copy()] 
    labels_list = []

    for _ in range(100):
        labels = pairwise_distances_argmin(data, centroids)
        labels_list.append(labels)

        new_centroids = []
        for i in range(n_clusters):
            points_in_cluster = data[labels == i]
            if len(points_in_cluster) == 0:
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(points_in_cluster.mean(axis=0))

        new_centroids = np.array(new_centroids)
        centroids_list.append(new_centroids)

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

        if run_to_convergence:
            centroids_list = [new_centroids]
            labels_list = [labels]
            break

    return centroids_list, labels_list

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/initial_data', methods=['GET'])
def initial_data():
    global global_data
    data = generate_data()
    global_data = data 
    df = pd.DataFrame(data, columns=['x', 'y'])
    fig = px.scatter(df, x='x', y='y', title='Initial Data')
    return jsonify({'graph': fig.to_json()})

@app.route('/kmeans', methods=['POST'])
def kmeans():
    global global_data

    if global_data is None:
        return jsonify({'error': 'No data available. Please generate the dataset first.'})

    num_clusters = int(request.form['num_clusters'])
    init_method = request.form['init_method']
    run_to_convergence = request.form.get('run_to_convergence', 'false').lower() == 'true'

    data = global_data.copy()

    if init_method == 'random':
        init_method_value = 'random'
    elif init_method == 'Farthest':
        init_method_value = farthest_first_initialization(data, num_clusters)
    elif init_method == 'k-means++':
        init_method_value = 'k-means++'
    elif init_method == 'manual':
        manual_centroids = np.array(json.loads(request.form['centroids']))
        init_method_value = manual_centroids

    centroids_list, labels_list = kmeans_with_iterations(
        data, num_clusters, init_method_value, run_to_convergence
    )

    centroids = [c.tolist() for c in centroids_list]
    labels = [l.tolist() for l in labels_list]
    data_list = data.tolist()

    return jsonify({'centroids': centroids, 'labels': labels, 'data': data_list})

if __name__ == '__main__':
    app.run(debug=True)
