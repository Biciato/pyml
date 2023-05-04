from io import BytesIO
from flask import Flask
from matplotlib.figure import Figure
import numpy as np
import base64
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

app = Flask(__name__)

X, y = make_blobs(n_features=2, n_samples=150, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X)

fig = Figure()

axes = fig.subplots()
axes.scatter(X[y_km == 0, 0], X[y_km == 0, 1], c='lightgreen', marker='s', edgecolor='black', s=50)
axes.set_xlabel('CLuster 1')
axes.scatter(X[y_km == 1, 0], X[y_km == 1, 1], c='orange', marker='o', edgecolor='black', s=50)
axes.set_xlabel('CLuster 2')
axes.scatter(X[y_km == 2, 0], X[y_km == 2, 1], c='lightblue', marker='v', edgecolor='black', s=50)
axes.set_xlabel('CLuster 3')
axes.scatter(
            km.cluster_centers_[:, 0], 
            km.cluster_centers_[:, 1], 
            c='red', 
            marker='*', 
            edgecolor='black', 
            s=50)
axes.set_xlabel('Centroids')
axes.legend(scatterpoints=1)
axes.grid()

@app.route('/')
def index():   
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"