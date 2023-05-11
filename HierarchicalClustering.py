import pandas as pd
import numpy as np
import base64
from io import BytesIO
from flask import Flask
from matplotlib.figure import Figure
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import set_link_color_palette

set_link_color_palette(['black'])

app = Flask(__name__)

fig = Figure()

axes = fig.subplots()
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
fig.set_size_inches((8, 8))
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample((5, 3)) * 10
df = pd.DataFrame(X, columns=variables, index=labels)
row_clusters = linkage(df.values, method='complete', metric='euclidean')
row_dendr = dendrogram(row_clusters, orientation='left')
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')
axd.set_yticks([])
axd.set_xticks([])
for i in axd.spines.values():
    i.set_visible(False)

fig.colorbar(cax)
axm.set_yticklabels([''] + list(df_rowclust.index))
axm.set_xticklabels([''] + list(df_rowclust.columns))

@app.route('/')
def index():   
    buf = BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"