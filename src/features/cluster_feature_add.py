import pandas as pd
import os
from src.features.cluster_deposit_per_area import calculate_cluster_deposit_per_m2, assign_clusters_to_test_data, add_clusters_and_deposit
from src.features.clustering import kmeans

def perform_clustering_and_calculate_deposit(train_data, test_data, n_clusters_list, max_iters, tol):
    if os.path.exists('./data/preprocessed/train_cluster_features.csv') and os.path.exists('./data/preprocessed/test_cluster_features.csv'):
        print("이미 처리된 파일이 존재합니다. cluster 연산 건너뜁니다.")
        return pd.read_csv('./data/preprocessed/train_cluster_features.csv'), pd.read_csv('./data/preprocessed/test_cluster_features.csv')


    for n_clusters in n_clusters_list:
        train_features = [train_data[['latitude', 'longitude']]]
        test_features = [test_data[['latitude', 'longitude']]]
        centroids, cluster_assignments = kmeans(train_data, n_clusters, max_iters, tol)

        cluster_deposit_per_m2 = calculate_cluster_deposit_per_m2(train_data, cluster_assignments, n_clusters)
        train_cluster_features = add_clusters_and_deposit(train_data, 
                                                                   cluster_assignments, 
                                                                   cluster_deposit_per_m2, 
                                                                   str(n_clusters))
        train_features.append(train_cluster_features.drop(columns=['latitude', 'longitude']))
        
        cluster_assignments_test = assign_clusters_to_test_data(test_data, centroids)
        
        test_cluster_features = add_clusters_and_deposit(test_data,  
                                                                   cluster_assignments_test, 
                                                                   cluster_deposit_per_m2, 
                                                                   str(n_clusters))
        test_features.append(test_cluster_features.drop(columns=['latitude', 'longitude']))

    # DataFrame 병합
    train_features_df = pd.concat(train_features, axis=1)
    test_features_df = pd.concat(test_features, axis=1)
    
    train_features_df.to_csv('./data/preprocessed/train_cluster_features.csv', index=False)
    test_features_df.to_csv('./data/preprocessed/test_cluster_features.csv', index=False)
    return train_features_df, test_features_df
