import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, num_clusters, num_epochs = 20):
        self.num_clusters = num_clusters
        self.num_epochs = num_epochs

    def fit(self, points: np.ndarray):
        N = points.shape[0]
        centroids = np.random.choice(range(N), size=self.num_clusters)
        centroids = points[centroids, :]

        for i in range(self.num_epochs):
            dist_mat = self.distance(points, centroids)
            labels = np.argmin(dist_mat, axis=-1)
            centroids = self.update_centroids(points, labels)

        return centroids, labels
    

    def distance(self, points: np.ndarray, centroids: np.ndarray):
        asq = np.sum(points**2, axis=1)
        bsq = np.sum(centroids**2, axis=1)
        asq = np.expand_dims(asq, axis=1)
        bsq = np.expand_dims(bsq, axis=0)

        ab = points @ centroids.T

        return asq+bsq-2*ab
    
    def update_centroids(self, points: np.ndarray, labels: np.ndarray):
        centroids = np.zeros(shape=(self.num_clusters, points.shape[-1]))
        for i in range(self.num_clusters):
            mask = labels==i
            cluster_pts = points[mask, :]
            centroids[i, :] = np.mean(cluster_pts, axis=0)
        return centroids
    
    def group_pts(points, labels):
        pass


def plot_points(points, labels, colors):
    for i in range(points.shape[0]):
        plt.scatter(points[i, 0], points[i,1], color = colors[labels[i]])
    plt.show()

if __name__ == "__main__":
    points = np.random.rand(100,2)
    model = KMeans(5)
    centroids, labels = model.fit(points)
    print(centroids.shape, labels.shape)
    colors = "red green blue yellow magenta".split()
    plot_points(points, labels, colors)