import pandas as pd
import torch
from src.features.clustering import haversine_distance

def calculate_cluster_deposit_per_m2(train_data, cluster_assignments, n_clusters):
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
    train_data[f'cluster_{cluster_label}'] = cluster_assignments.cpu().numpy()
    train_data[f'cluster_deposit_per_m2_{cluster_label}'] = train_data[f'cluster_{cluster_label}'].map(lambda cluster: cluster_deposit_per_m2[cluster])
    return train_data


def assign_clusters_to_test_data(test_data, centroids):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = torch.tensor(test_data[['latitude', 'longitude']].values, dtype=torch.float32).to(device)
    centroids = centroids.to(device)
    distances = torch.stack([haversine_distance(data_tensor[:, 0], data_tensor[:, 1], c[0], c[1]) for c in centroids], dim=1)
    
    return torch.argmin(distances, dim=1).cpu()

def add_test_data_clusters_and_deposit(test_data, cluster_assignments, cluster_deposit_per_m2, cluster_label):
    test_data[f'cluster_{cluster_label}'] = cluster_assignments
    test_data[f'cluster_deposit_per_m2_{cluster_label}'] = test_data[f'cluster_{cluster_label}'].map(lambda cluster: cluster_deposit_per_m2[cluster])
    return test_data
