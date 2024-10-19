import pandas as pd
import torch
from src.features.clustering import haversine_distance

def calculate_cluster_deposit_per_m2(train_data, cluster_assignments, n_clusters):
    """
    Calculate average deposit per m² for each cluster.
    :param train_data: (pd.DataFrame) The train data with 'deposit' and 'area_m2' columns
    :param cluster_assignments: (torch.Tensor) Cluster assignments for each data point
    :param n_clusters: (int) Number of clusters
    :return: (list) Average deposit per m² for each cluster
    """
    cluster_deposit_per_m2 = []
    
    for i in range(n_clusters):
        cluster_indices = (cluster_assignments.cpu().numpy() == i)
        cluster_data = train_data[cluster_indices]
        
        if len(cluster_data) > 0:
            avg_deposit_per_m2 = cluster_data['deposit'].sum() / cluster_data['area_m2'].sum()
            cluster_deposit_per_m2.append(avg_deposit_per_m2)
        else:
            cluster_deposit_per_m2.append(0)
    
    return cluster_deposit_per_m2


def add_cluster_results_to_train_data(train_data, cluster_assignments, cluster_deposit_per_m2, cluster_label):
    """
    Add cluster assignments and deposit per m² results to the train_data DataFrame.
    :param train_data: (pd.DataFrame) The original train data
    :param cluster_assignments: (torch.Tensor) Cluster assignments for each data point
    :param cluster_deposit_per_m2: (list) Average deposit per m² for each cluster
    :param cluster_label: (str) The label to append to the column names (e.g., '100', '80', '2000')
    :return: (pd.DataFrame) Updated train data with cluster and deposit columns
    """
    train_data[f'cluster_{cluster_label}'] = cluster_assignments.cpu().numpy()
    train_data[f'cluster_deposit_per_m2_{cluster_label}'] = train_data[f'cluster_{cluster_label}'].map(lambda cluster: cluster_deposit_per_m2[cluster])
    return train_data


def assign_clusters_to_test_data(test_data, centroids):
    """
    Assign clusters to the test data based on centroids.
    :param test_data: (pd.DataFrame) The test data with 'latitude' and 'longitude' columns
    :param centroids: (torch.Tensor) Centroids of the clusters
    :return: (torch.Tensor) Cluster assignments for the test data
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = torch.tensor(test_data[['latitude', 'longitude']].values, dtype=torch.float32).to(device)
    centroids = centroids.to('device')
    distances = torch.stack([haversine_distance(data_tensor[:, 0], data_tensor[:, 1], c[0], c[1]) for c in centroids], dim=1)
    
    return torch.argmin(distances, dim=1).cpu()  # Return results on CPU


def add_test_data_clusters_and_deposit(test_data, cluster_assignments, cluster_deposit_per_m2, cluster_label):
    """
    Add cluster assignments and deposit per m² to test_data.
    :param test_data: (pd.DataFrame) The test data
    :param cluster_assignments: (torch.Tensor) Cluster assignments for the test data
    :param cluster_deposit_per_m2: (list) Average deposit per m² for each cluster
    :param cluster_label: (str) The label to append to the column names (e.g., '100', '80', '2000')
    :return: (pd.DataFrame) Updated test data with cluster and deposit columns
    """
    test_data[f'cluster_{cluster_label}'] = cluster_assignments
    test_data[f'cluster_deposit_per_m2_{cluster_label}'] = test_data[f'cluster_{cluster_label}'].map(lambda cluster: cluster_deposit_per_m2[cluster])
    return test_data
