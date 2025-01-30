
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import diags
from matplotlib.colors import ListedColormap, BoundaryNorm


def preprocess_image(image_path, target_size=(150, 150)):
    """
    Convert image to grayscale and resize.
    """
    image = np.array(Image.open(image_path).convert("L"))  # Convert to grayscale
    image = cv2.resize(image, target_size)  # Resize to target size
    return image


def create_weight_matrix(image, n_neighbors, sigma_pixel, sigma_geom):
    """
    Create a sparse weight matrix for spectral clustering.
    """
    pixel_values = image.flatten().reshape(-1, 1)  # Pixel values as features
    coords = np.indices(image.shape).reshape(2, -1).T  # Coordinates as features
    features = np.hstack((pixel_values, coords))  # Combine pixel values and coordinates

    # Calculate nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='kd_tree', n_jobs=-1).fit(features)
    distances, indices = nbrs.kneighbors(features)

    row_idx, col_idx, weights = [], [], []

    for i in range(len(features)):
        for j in range(1, n_neighbors + 1):  # Start from 1 to exclude self-neighbors
            neighbor_idx = indices[i, j]
            pixel_diff = pixel_values[i] - pixel_values[neighbor_idx]
            geom_dist = np.linalg.norm(coords[i] - coords[neighbor_idx])

            # Compute Gaussian weight
            weight = np.exp(-(pixel_diff**2 / (2 * sigma_pixel**2)) - (geom_dist**2 / (2 * sigma_geom**2)))

            row_idx.append(i)
            col_idx.append(neighbor_idx)
            weights.append(weight)

    # Ensure row_idx, col_idx, and weights are numpy arrays and flattened
    row_idx = np.array(row_idx).flatten()
    col_idx = np.array(col_idx).flatten()
    weights = np.array(weights).flatten()

    # Construct sparse weight matrix
    W = csr_matrix((weights, (row_idx, col_idx)), shape=(len(features), len(features)))

    # Symmetrize the weight matrix
    W = (W + W.T) / 2

    return W

def auto_elbow_with_euclidean_ratio(image, max_clusters=6):
    """
    AutoElbow method to determine the optimal number of clusters.
    """
    inertias = []
    cluster_range = range(1, max_clusters + 1)

    # Calculate inertia for each k (cluster count)
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(image)
        inertias.append(kmeans.inertia_)

    x_norm = (np.array(list(cluster_range)) - np.min(cluster_range)) / (np.max(cluster_range) - np.min(cluster_range))
    y_norm = (np.array(inertias) - np.min(inertias)) / (np.max(inertias) - np.min(inertias))

    a_k = np.sqrt(x_norm ** 2 + y_norm ** 2)  # Distance to the origin
    b_k = np.sqrt((x_norm - 1) ** 2 + (y_norm - 1) ** 2)  # Distance to the max point
    c_k = y_norm  # Distance to the x-axis

    f_k = b_k / (a_k + c_k)

    optimal_idx = np.argmax(f_k)
    optimal_k = cluster_range[optimal_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, inertias, label="Evaluation Metric Curve")
    plt.scatter(optimal_k, inertias[optimal_idx], color="red", label=f"Optimal k: {optimal_k}", zorder=5)
    plt.title("AutoElbow Method with Euclidean Ratio")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.legend()
    plt.grid()
    plt.show()

    return optimal_k

def normalized_spectral_clustering_rw(W, num_clusters):
    """
    Perform spectral clustering using the random walk Laplacian.
    """
    degrees = np.array(W.sum(axis=1)).flatten()
    D_inv = diags(1.0 / degrees)  # Inverse degree matrix

    # Compute the random walk Laplacian
    P = D_inv @ W
    L_rw = csr_matrix(np.eye(P.shape[0])) - P  # Random walk Laplacian: I - P
    eigenvalues, eigenvectors = eigsh(L_rw, k=num_clusters, which='SM')

    # Perform k-means clustering on the eigenvectors
    eigenvectors = eigenvectors[:, :num_clusters]  # Select the first num_clusters eigenvectors
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(eigenvectors)

    return labels


def plot_spectral_clustering(image, labels, image_shape, num_clusters,sigma_pixel, sigma_geom):
    """
    Visualize the original image and the spectral clustering results
    """

    cmap = plt.cm.viridis
    discrete_colors = cmap(np.linspace(0, 1, num_clusters))
    cmap_discrete = ListedColormap(discrete_colors)
    norm = BoundaryNorm(np.arange(-0.5, num_clusters + 0.5), num_clusters, clip=True)

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.figure(figsize=(10, 10))
    clustered_image1 = labels.reshape(image_shape)
    plt.imshow(clustered_image1, cmap=cmap_discrete, norm=norm)
    plt.title(f"Spectral Clustering Results\n"
              f"sigma_pixel = {sigma_pixel}   sigma_geom = {sigma_geom}")
    cbar = plt.colorbar(ticks=range(num_clusters), boundaries=np.arange(-0.5, num_clusters + 0.5))
    cbar.set_label("Cluster Index")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    image_path = "296059.jpg"
    n_neighbors = 8
    sigma_pixel = 3
    sigma_geom = 8

    image = preprocess_image(image_path)
    optimal_k = auto_elbow_with_euclidean_ratio(image, max_clusters=6)
    W = create_weight_matrix(image, n_neighbors, sigma_pixel, sigma_geom)
    labels = normalized_spectral_clustering_rw(W, optimal_k)
    plot_spectral_clustering(image, labels, image.shape, optimal_k, sigma_pixel, sigma_geom)
main()