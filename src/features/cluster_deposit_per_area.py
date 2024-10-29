import pandas as pd
import torch
from src.features.clustering import haversine_distance

def calculate_cluster_deposit_per_m2(train_data, cluster_assignments, n_clusters):
    cluster_deposit_per_m2 = {f'{year}{quarter}': [0] * n_clusters for year in range(2019, 2024) for quarter in ['Q1', 'Q2', 'Q3', 'Q4']}
    
    for i in range(n_clusters):
        cluster_indices = (cluster_assignments.cpu().numpy() == i)
        cluster_data = train_data[cluster_indices]
        
        if len(cluster_data) > 0:
            cluster_data['year'] = cluster_data['contract_year_month'] // 100
            cluster_data['month'] = cluster_data['contract_year_month'] % 100
            cluster_data['quarter'] = pd.to_datetime(cluster_data[['year', 'month']].assign(day=1)).dt.to_period('Q')
            for period, period_data in cluster_data.groupby('quarter'):
                year_quarter = f'{period.year}Q{period.quarter}'
                avg_deposit_per_m2 = period_data['deposit'].sum() / period_data['area_m2'].sum()
                cluster_deposit_per_m2[year_quarter][i] = avg_deposit_per_m2
    
    return cluster_deposit_per_m2


def assign_clusters_to_test_data(test_data, centroids):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = torch.tensor(test_data[['latitude', 'longitude']].values, dtype=torch.float32).to(device)
    centroids = centroids.to(device)
    distances = torch.stack([haversine_distance(data_tensor[:, 0], data_tensor[:, 1], c[0], c[1]) for c in centroids], dim=1)
    return torch.argmin(distances, dim=1).cpu()


def add_clusters_and_deposit(test_data, cluster_assignments, cluster_deposit_per_m2, cluster_label):
    test_data[f'cluster_{cluster_label}'] = cluster_assignments.cpu().numpy()

    # 각 클러스터에 해당하는 분기별 보증금을 test_data에 추가
    for year in range(2019, 2024):  # 2019년부터 2023년까지의 분기별 데이터를 추가
        for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
            year_quarter = f'{year}{quarter}'
            
            # 해당 year_quarter가 cluster_deposit_per_m2에 있을 경우에만 열을 추가
            if year_quarter in cluster_deposit_per_m2:
                test_data[f'cluster_{cluster_label}_{year_quarter}_deposit_per_m2'] = test_data[f'cluster_{cluster_label}'].map(
                    lambda cluster: cluster_deposit_per_m2[year_quarter][cluster] 
                )
    
    return test_data