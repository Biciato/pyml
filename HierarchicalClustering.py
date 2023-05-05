import pandas as pd
import numpy as np

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample((5, 3)) * 10
df = pd.DataFrame(X, columns=variables, index=labels)

print(X)