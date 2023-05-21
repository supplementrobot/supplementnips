# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse

# For training and test data splits
X_WIDTH = 150
Y_WIDTH = 150

# For Oxford
P1 = [5735712.768124, 620084.402381]
P2 = [5735611.299219, 620540.270327]
P3 = [5735237.358209, 620543.094379]
P4 = [5734749.303802, 619932.693364]

# For University Sector
P5 = [363621.292362, 142864.19756]
P6 = [364788.795462, 143125.746609]
P7 = [363597.507711, 144011.414174]

# For Residential Area
P8 = [360895.486453, 144999.915143]
P9 = [362357.024536, 144894.825301]
P10 = [361368.907155, 145209.663042]






P_DICT = {
    "oxford": [P1, P2, P3, P4], 
    "university": [P5, P6, P7], 
    "residential": [P8, P9, P10], 
    "business": [],
    'boreas': [],
    
}


def check_in_test_set(northing, easting, points):
    in_test_set = False
    for point in points:
        if point[0] - X_WIDTH < northing < point[0] + X_WIDTH and point[1] - Y_WIDTH < easting < point[1] + Y_WIDTH:
            in_test_set = True
            break
    return in_test_set


def output_to_file(output, base_path, filename):
    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name):
    database_trees = []
    test_trees = []
    for folder in folders:
        print(folder)
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')
        # df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
        # df_locations=df_locations.rename(columns={'timestamp':'file'})
        for index, row in df_locations.iterrows():

            
            # entire business district is in the test set
            if output_name == "business":
                df_test = df_test.append(row, ignore_index=True)
            elif output_name == 'boreas':
                df_test = df_test.append(row, ignore_index=True)
            elif check_in_test_set(row['northing'], row['easting'], p):
                df_test = df_test.append(row, ignore_index=True)


            df_database = df_database.append(row, ignore_index=True)

        database_tree = KDTree(df_database[['northing', 'easting']])
        test_tree = KDTree(df_test[['northing', 'easting']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)


    test_sets = []
    database_sets = []
    for folder in folders:
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(base_path, runs_folder, folder, filename), sep=',')

        if output_name == 'boreas':
            df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + \
                                        df_locations['timestamp'].astype(str) + '.npy'
        else:
            df_locations['timestamp'] = runs_folder + folder + pointcloud_fols + \
                                        df_locations['timestamp'].astype(str) + '.bin'
            


        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if output_name in ["business", 'boreas']:
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            
            elif check_in_test_set(row['northing'], row['easting'], p):
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                              'easting': row['easting']}
        database_sets.append(database)
        test_sets.append(test)

    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if i == j:
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                index = tree.query_radius(coor, r=25)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()

    output_to_file(database_sets, base_path, output_name + '_evaluation_database.pickle')
    output_to_file(test_sets, base_path, output_name + '_evaluation_query.pickle')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation datasets')
    # parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    
    parser.add_argument('--dataset_root', 
        type=str, 
        # default='/data/user/vpr/MinkLocMultimodal/benchmark_datasets', 
        # default='/home/user/vpr/BenchmarkBoreas', 
        # default='/home/user/vpr/BenchmarkBoreasv2', 
        default='/home/user/vpr/BenchmarkBoreasv3', 
        help='Dataset root folder')

    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root



    # # For Oxford
    # folders = []
    # runs_folder = "oxford/"
    # all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    # index_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 31, 32, 33, 38, 39, 43, 44]
    # print(len(index_list))
    # for index in index_list:
    #     folders.append(all_folders[index])
    # print(folders)
    # construct_query_and_database_sets(
    #     base_path=base_path,
    #     runs_folder=runs_folder,
    #     folders=folders,
    #     pointcloud_fols="/pointcloud_20m/",
    #     filename="pointcloud_locations_20m.csv",
    #     p=P_DICT["oxford"],
    #     output_name="oxford"
    # )



    # For Boreas
    folders = []
    runs_folder = "boreas/"
    # # -- index select
    # all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    # index_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 31, 32, 33, 38, 39, 43, 44]
    # print(len(index_list))
    # for index in index_list:
    #     folders.append(all_folders[index])
    # -- mannual select

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
        # 'boreas-2020-11-26-13-58', # snow
        # 'boreas-2020-12-01-13-26', # snowing
        # 'boreas-2020-12-18-13-44', # sun snow
        # 'boreas-2021-05-13-16-11', # sun

        # 'boreas-2021-07-20-17-33', # clouds rain
        # 'boreas-2021-07-27-14-43', # clouds



        'boreas-2021-01-26-11-22', # snowing
        'boreas-2021-04-29-15-55', # raining
        'boreas-2021-06-17-17-52', # sun
        'boreas-2021-09-14-20-00',  # night

        'boreas-2021-10-15-12-35', # clouds
        'boreas-2021-11-02-11-16', # sun clouds
    ]

    
    print(folders)
    construct_query_and_database_sets(
        base_path=base_path,
        runs_folder=runs_folder,
        folders=folders,
        pointcloud_fols="/lidar_1_4096_interval10/",
        filename="pointcloud_locations_interval10.csv",
        p=P_DICT["boreas"],
        output_name="boreas"
    )



    # # For University Sector
    # folders = []
    # runs_folder = "inhouse_datasets/"
    # all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    # uni_index = range(10, 15)
    # for index in uni_index:
    #     folders.append(all_folders[index])

    # print(folders)
    # construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
    #                                   "pointcloud_centroids_25.csv", P_DICT["university"], "university")

    # # For Residential Area
    # folders = []
    # runs_folder = "inhouse_datasets/"
    # all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    # res_index = range(5, 10)
    # for index in res_index:
    #     folders.append(all_folders[index])

    # print(folders)
    # construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
    #                                   "pointcloud_centroids_25.csv", P_DICT["residential"], "residential")

    # # For Business District
    # folders = []
    # runs_folder = "inhouse_datasets/"
    # all_folders = sorted(os.listdir(os.path.join(base_path, runs_folder)))
    # bus_index = range(5)
    # for index in bus_index:
    #     folders.append(all_folders[index])

    # print(folders)
    # construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud_25m_25/",
    #                                   "pointcloud_centroids_25.csv", P_DICT["business"], "business")
