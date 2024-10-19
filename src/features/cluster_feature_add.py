import pandas as pd
from src.features.cluster_deposit_per_area import calculate_cluster_deposit_per_m2, add_cluster_results_to_train_data, assign_clusters_to_test_data, add_test_data_clusters_and_deposit
from src.features.clustering import kmeans

def perform_clustering_and_calculate_deposit(train_data, test_data, n_clusters_list, max_iters, tol):
    """
    Perform K-Means clustering and calculate cluster deposit per m².
    Return new features without modifying original train/test data.
    
    :param train_data: (pd.DataFrame) Training data
    :param test_data: (pd.DataFrame) Test data
    :param n_clusters_list: (list) List of numbers of clusters to use
    :param max_iters: (int) Maximum iterations for K-Means
    :param tol: (float) Convergence tolerance
    :return: (list of pd.DataFrames, list of pd.DataFrames) Cluster features for train_data and test_data
    """
    train_features = []
    test_features = []
    
    for n_clusters in n_clusters_list:
        # Perform K-Means clustering
        centroids, cluster_assignments = kmeans(train_data, n_clusters, max_iters, tol)
        
        # Calculate deposit per m² for each cluster in train_data
        cluster_deposit_per_m2 = calculate_cluster_deposit_per_m2(train_data, cluster_assignments, n_clusters)
        
        # Create clustering result features for train_data
        train_cluster_features = add_cluster_results_to_train_data(train_data[['latitude', 'longitude']], 
                                                                   cluster_assignments, 
                                                                   cluster_deposit_per_m2, 
                                                                   str(n_clusters))
        train_features.append(train_cluster_features)
        
        # Assign clusters to test_data
        cluster_assignments_test = assign_clusters_to_test_data(test_data, centroids)
        
        # Create clustering result features for test_data
        test_cluster_features = add_test_data_clusters_and_deposit(test_data[['latitude', 'longitude']], 
                                                                   cluster_assignments_test, 
                                                                   cluster_deposit_per_m2, 
                                                                   str(n_clusters))
        test_features.append(test_cluster_features)

    train_features_df = pd.concat(train_features, axis=1)
    test_features_df = pd.concat(test_features, axis=1)
    
    return train_features_df, test_features_df
