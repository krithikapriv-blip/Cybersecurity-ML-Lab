import numpy as np
from sklearn.cluster import KMeans

# Sample network traffic data
data = np.array([
    [10, 100],
    [12, 110],
    [11, 105],
    [13, 115],
    [100, 900],
    [110, 950],
    [105, 920],
    [15, 120]
])

# Apply K-Means clustering
model = KMeans(n_clusters=2, random_state=42, n_init=10)
model.fit(data)

# Predict network traffic clusters
clusters = model.predict(data)

# Identify the normal traffic cluster
normal_cluster = np.argmin(
    np.sum(model.cluster_centers_, axis=1)
)

print("Network Traffic Anomaly Detection")
print("---------------------------------")

for i, cluster in enumerate(clusters):
    if cluster == normal_cluster:
        status = "Normal Traffic"
    else:
        status = "Anomaly"

    print("Traffic", i + 1, ":", status)
