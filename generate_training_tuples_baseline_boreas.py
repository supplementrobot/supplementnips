# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
import tqdm

from datasets.oxford import TrainingTuple
# Import test set boundaries
from generate_test_sets_boreas import P1, P2, P3, P4, check_in_test_set

# Test set boundaries
P = [P1, P2, P3, P4]

# RUNS_FOLDER = "oxford/"
# FILENAME = "pointcloud_locations_20m_10overlap.csv"
# POINTCLOUD_FOLS = "/pointcloud_20m_10overlap/"



RUNS_FOLDER = "boreas/"
FILENAME = "pointcloud_locations_interval10.csv"
POINTCLOUD_FOLS = "/lidar_1_4096_interval10/"




def construct_query_dict(df_centroids, base_path, filename, ind_nn_r, ind_r_r=50):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)
    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]
        # Extract timestamp from the filename
        scan_filename = os.path.split(query)[1]
        assert os.path.splitext(scan_filename)[1] == '.npy', f"Expected .npy file: {scan_filename}"
        timestamp = int(os.path.splitext(scan_filename)[0])

        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos)

    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
    # parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    parser.add_argument('--dataset_root', 
        type=str, 
        # default='/home/user/vpr/benchmark_datasets', 
        # default='/home/user/vpr/BenchmarkBoreas', 
        # default='/home/user/vpr/BenchmarkBoreasv2', 
        default='/home/user/vpr/BenchmarkBoreasv3', 
        help='Dataset root folder')
    
    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root




    # all_folders = sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER)))
    # folders = []
    # # All runs are used for training (both full and partial)
    # index_list = range(len(all_folders) - 1)
    # print("Number of runs: " + str(len(index_list)))
    # for index in index_list:
    #     folders.append(all_folders[index])
    # print(folders)



    # folders = [
    #     'boreas-2020-11-26-13-58', # snow
    #     'boreas-2020-12-01-13-26', # snowing
    #     'boreas-2020-12-18-13-44', # sun snow
    #     'boreas-2021-05-13-16-11', # sun


    #     # 'boreas-2021-01-26-11-22', # snowing
    #     # 'boreas-2021-04-29-15-55', # raining

    #     # 'boreas-2021-06-17-17-52', # sun

    #     # 'boreas-2021-09-14-20-00'  # night
    # ]


    folders = [
        'boreas-2020-11-26-13-58', # snow
        'boreas-2020-12-01-13-26', # snowing
        'boreas-2020-12-18-13-44', # sun snow
        'boreas-2021-05-13-16-11', # sun

        'boreas-2021-07-20-17-33', # clouds rain
        'boreas-2021-07-27-14-43', # clouds



        # 'boreas-2021-01-26-11-22', # snowing
        # 'boreas-2021-04-29-15-55', # raining
        # 'boreas-2021-06-17-17-52', # sun
        # 'boreas-2021-09-14-20-00',  # night

        # 'boreas-2021-10-15-12-35', # clouds
        # 'boreas-2021-11-02-11-16', # sun clouds
    ]



    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

    for folder in tqdm.tqdm(folders):
        df_locations = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, folder, FILENAME), sep=',')
        df_locations['timestamp'] = RUNS_FOLDER + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(str) + '.npy'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

        for index, row in df_locations.iterrows():
            # if check_in_test_set(row['northing'], row['easting'], P):
            #     df_test = df_test.append(row, ignore_index=True)
            # else:
                df_train = df_train.append(row, ignore_index=True)

    print("Number of training submaps: " + str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
    # ind_nn_r is a threshold for positive elements - 10 is in original PointNetVLAD code for refined dataset
    construct_query_dict(df_train, base_path, "boreas_training_queries_baseline.pickle", ind_nn_r=10)

    # construct_query_dict(df_test, base_path, "test_queries_baseline.pickle", ind_nn_r=10)
