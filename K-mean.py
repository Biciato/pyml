from io import BytesIO
from flask import Flask
from matplotlib.figure import Figure
from matplotlib import cm
import numpy as np
import base64
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

app = Flask(__name__)

fig = Figure()

axes = fig.subplots()

X, y = make_blobs(n_features=2, n_samples=150, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0, tol=1e-04)
y_km = km.fit_predict(X)

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km,metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    axes.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)

axes.axvline(silhouette_avg, color='red', linestyle='--')
axes.set_yticks(yticks, cluster_labels + 1)
axes.set_xlabel('Silhouette coefficient')
axes.set_ylabel('Cluster')
axes.set_in_layout(in_layout=True)

@app.route('/')
def index():   
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"