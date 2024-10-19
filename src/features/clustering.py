import time
import pandas as pd
import torch
from src.utils.variables import RANDOM_SEED
# call kmeans(data: pd.DataFrame, n_clusters: int, max_iters: int, tol: float)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the haversine distance between two points
    :param lat1: (torch.Tensor) Latitude of point 1
	:param lon1: (torch.Tensor) Longitude of point 1
	:param lat2: (torch.Tensor) Latitude of point 2
	:param lon2: (torch.Tensor) Longitude of point 2
	:return: (torch.Tensor) Haversine distance
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(lambda x: x * (torch.pi / 180), (lat1, lon1, lat2, lon2))
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return R * c

def initialize_centroids(data, n_clusters):
    """
    Initialize centroids using k-means++
    :param data: (torch.Tensor) Data
    :param n_clusters: (int) Number of clusters
    :return: (torch.Tensor) Centroids
    """
    n_samples = data.size(0)
    torch.manual_seed(RANDOM_SEED)
    centroids = [data[torch.randint(0, n_samples, (1,)).item()]]
    
    for _ in range(1, n_clusters):
        dist = torch.stack([haversine_distance(data[:, 0], data[:, 1], c[0], c[1]) for c in centroids]).min(dim=0)[0]
        
        probs = dist / dist.sum()
        next_centroid = data[torch.multinomial(probs, 1).item()]
        
        centroids.append(next_centroid)
    
    return torch.stack(centroids)


def kmeans(data: pd.DataFrame, n_clusters: int, max_iters: int, tol: float):
	"""
	K-Means clustering
	:param data: (pd.DataFrame) longitude and latitude data of the train_data
	:param n_clusters: (int) Number of clusters
	:param max_iters: (int) Maximum number of iterations
	:param tol: (float) Tolerance for convergence
	:return: (torch.Tensor) Centroids, (torch.Tensor) Cluster
	"""
	print("------------------------------")
	print("Clustering Start")
	print("------------------------------")
    
	start_time = time.time()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	data = data[['latitude', 'longitude']]
	data = torch.tensor(data.values, dtype=torch.float32).to(device)

	centroids = initialize_centroids(data, n_clusters)
	
	for i in range(max_iters):
		distances = torch.stack([haversine_distance(data[:, 0], data[:, 1], c[0], c[1]) for c in centroids], dim=1)
		cluster_assignments = torch.argmin(distances, dim=1)
		new_centroids = torch.stack([data[cluster_assignments == k].mean(0) for k in range(n_clusters)])
		centroid_shift = torch.norm(centroids - new_centroids, dim=1).max()
		centroids = new_centroids
		
		if centroid_shift < tol:
			print(f"Converged at iteration {i}")
			break
            
	print(f"\nClustering took {time.time() - start_time:.2f} seconds\n")

	print("------------------------------")
	print("Clustering Completed")
	print("------------------------------")
	
	return centroids, cluster_assignments