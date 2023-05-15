import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import fnmatch
import os
import json
import seaborn as sns
import matplotlib as mpl
import open3d as o3d
import re
import shutil
import h5py
import csv
from decimal import Decimal

from data_utils.augmention import *


def filter_extreme_percentile(data, min =0.025,max = 0.975):
    series = pd.Series({'a': data})
    series = series.sort_values()
    q = series.quantile([min,max])
    return np.clip(series,q.iloc[0],q.iloc[1])


def balance_data_volums(num_kt, num_pd):
    random_KT_PD_num = {}
    if num_kt < num_pd:
        random_KT_PD_num['HC'] = 1.0
        random_KT_PD_num['PD'] = num_kt / num_pd
    else:
        random_KT_PD_num['HC'] = num_pd / num_kt
        random_KT_PD_num['PD'] = 1.0
    return random_KT_PD_num

def normalization_array_data(data, data_range):
    normalized_data = data.copy()
    for i in np.arange(data.shape[1]):
        min_value, max_value = data_range[:, i]
        normalized_data[:,i] = (normalized_data[:,i]-min_value)/(max_value-min_value)
    return normalized_data


def get_full_data_array(frame_data, index):
    dim = len(index)
    temp_data = np.zeros((1, dim))
    for j, stroke_idx in enumerate(frame_data):
        stroke = pd.DataFrame(stroke_idx)
        stroke_frame = stroke[index]
        stroke_data = stroke_frame.to_numpy()
        temp_data = np.vstack((temp_data, stroke_data))
    data = temp_data[1:,:]
    return data


def get_column_data_scale(data):
    min_scale = np.min(data, axis=0)
    max_scale = np.max(data, axis=0)
    data_scale = np.vstack((min_scale, max_scale))
    return data_scale











def check_save_data_path(path, sub_path, data):
    data_path = os.path.join(path, sub_path)  # '../data/dataset/KT/KT1/ptrace_000_json/aug/src
    print('      OUTPUT:'+os.path.join(data_path, '3d_voxel.npy'))
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
        os.mkdir(data_path)
    else:
        os.mkdir(data_path)
    print('        %s'%(str(data.shape)))
    with open(os.path.join(data_path, '3d_voxel.npy'), 'wb') as f:# '../data/dataset/KT/KT1/ptrace_000_json/aug/src/3d_voxel.npy
        np.save(f, data)


def generate_voxel_from_json(data, data_range):
    data = normalization_array_data(data, data_range)
    a_arr = np.array(data[:, 4])
    l_arr = np.array(data[:, 5])
    p_arr = np.array(data[:, 6])
    x_arr = np.array(data[:, 1])
    y_arr = np.array(data[:, 0])
    v_arr = np.array(data[:, 7])
    acc_arr = np.array(data[:, 8])
    jerk_arr = np.array(data[:, 9])

    # cmap = plt.cm.get_cmap('viridis')
    # color_cmap = cmap(np.arange(cmap.N))
    # colors = np.ones((acc_arr.shape[0], 3)).astype(np.float)
    # for i in np.arange(acc_arr.shape[0]):
    #     weight = acc_arr[i]
    #     new_weight = int(Decimal(255 * weight).quantize(Decimal("1."), rounding="ROUND_HALF_UP"))
    #     colors[i, :] = color_cmap[new_weight, :3]

    points = np.array([x_arr, y_arr, v_arr]).reshape(3, -1).T
    # points_color = colors

    points_color = np.array([a_arr, l_arr, p_arr]).reshape(3, -1).T

    # m, n = points_color.shape
    # points_color = np.zeros((m,n)).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack((points[:, 0], points[:, 1], points[:, 2])).transpose())
    pcd.colors = o3d.utility.Vector3dVector(np.vstack((points_color[:, 0], points_color[:, 1], points_color[:, 2])).transpose())
    # o3d.visualization.draw_geometries([pcd], window_name='Raw Point Cloud data', mesh_show_wireframe=True,
    #                                   mesh_show_back_face=True)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1/126)
    voxels = voxel_grid.get_voxels()
    coordianate_indices = np.stack(list(vx.grid_index for vx in voxels))
    x_length = np.max(coordianate_indices.T[0])
    y_width = np.max(coordianate_indices.T[1])
    z_height = np.max(coordianate_indices.T[2])
    ex_array = np.ones((x_length + 2, y_width + 2, z_height + 2, 3)).astype(np.float64)
    for i in range(0, len(voxels)):
        ex_array[voxels[i].grid_index[0]+1,
                 voxels[i].grid_index[1]+1,
                 voxels[i].grid_index[2]+1, :] = voxels[i].color

    return ex_array




def generate_aug_voxel_from_voxel(data, aug_method_dict, params_list, path):

    for method_index in aug_method_dict:
        aug_method = aug_method_dict[method_index]

        if (aug_method == 'src'):
            aug_data = data
            aug_data_range = get_column_data_scale(aug_data)
            voxel_array = generate_voxel_from_json(aug_data, aug_data_range)
            check_save_data_path(path, 'src', voxel_array)


        elif (aug_method == 'upsampling'):
            for sampling_factor in params_list['upsampling']:
                aug_data = up_sampling_transform(data, sampling_factor)
                aug_data_range = get_column_data_scale(aug_data)
                aug_voxel_array = generate_voxel_from_json(aug_data, aug_data_range)
                check_save_data_path(path, 'upsampling_{}'.format(sampling_factor), aug_voxel_array)


        elif (aug_method == 'downsampling'):
            for sampling_factor in params_list['downsampling']:
                aug_data = down_sampling_transform(data, sampling_factor)
                aug_data_range = get_column_data_scale(aug_data)
                aug_voxel_array = generate_voxel_from_json(aug_data, aug_data_range)
                check_save_data_path(path, 'downsampling_{}'.format(sampling_factor), aug_voxel_array)


        elif (aug_method == 'rotation'):
            for rotate_angle in params_list['rotation']:
                aug_voxel_array = rotation_transform(data, rotate_angle)
                check_save_data_path(path, 'rotation_{}'.format(round(rotate_angle*180/np.pi)), aug_voxel_array)


        elif (aug_method == 'brightness_shift'):
            for max_shift_value in params_list['brightness_shift']:
                aug_data = brightness_shift(data, max_shift_value)
                aug_data_range = get_column_data_scale(aug_data)
                aug_voxel_array = generate_voxel_from_json(aug_data, aug_data_range)
                check_save_data_path(path, 'brightness_shift_{}'.format(max_shift_value), aug_voxel_array)


        elif (aug_method == 'brightness_scaling'):
            for max_scale_value in params_list['brightness_scaling']:
                aug_data = brightness_scaling(data, max_scale_value)
                aug_data_range = get_column_data_scale(aug_data)
                aug_voxel_array = generate_voxel_from_json(aug_data, aug_data_range)
                check_save_data_path(path, 'brightness_scaling_{}'.format(max_scale_value), aug_voxel_array)


        elif (aug_method == 'jittering'):
            for max_jitter_value in params_list['jittering']:
                aug_data = location_jittering(data, max_jitter_value)
                aug_data_range = get_column_data_scale(aug_data)
                aug_voxel_array = generate_voxel_from_json(aug_data, aug_data_range)
                check_save_data_path(path, 'location_jittering_{}'.format(max_jitter_value), aug_voxel_array)
        else:
            print('          %s augmentation method does not exist.' % (aug_method))


def read_signal_csv(path, if_remove):
    # print('####%s' % (path))
    data = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            if len(row) == 1:
                original_data_length = int(row[0])
            if len(row) == 7:
                if if_remove:
                    if row[3] == '1':  # 这个值为0的话代表笔处于悬空，为1的话代表笔接触屏幕
                        row_new = [float(i) for i in row]
                        data.append(row_new)
                else:
                    row_new = [float(i) for i in row]
                    data.append(row_new)

    data = np.array(data)
    useful_data_lenght = len(data)
    # print('    original length:%d, useful length:%d' % (original_data_length, useful_data_lenght))
    return data


def get_augment_voxel_from_json(data_path, testee_type, testee_number, pattern, file_id, file, aug_method_dict, params_list, if_remove):
    json_file_path = os.path.join(data_path, 'raw_data', testee_type, testee_number, file)  # '../data/raw_data/KT/KT1/KT-01_8D6834FB_pcontinue_2017-11-05_16_45_31___0095aa3c709f457cbe43e800ee7a299f.json
    print('******INPUT FILE:' + json_file_path)

    json_to_voxel_path = os.path.join(data_path, 'dataset', testee_type, testee_number, file_id, 'aug')  # '../data/dataset/KT/KT1/ptrace_000_json/aug

    data = read_signal_csv(json_file_path, if_remove)
    m, n = data.shape
    x_coordinate = data[:,1]
    y_coordinate = data[:,0]
    time_coordinate = data[:, 2]

    x_coordinate_diff = np.diff(x_coordinate, n=1, axis=-1, append=0)
    y_coordinate_diff = np.diff(y_coordinate, n=1, axis=-1, append=0)
    time_coordinate_diff = np.diff(time_coordinate, n=1, axis=-1, append=0)

    velocity_list = np.array([pow( pow((x_coordinate_diff[idx]), 2) + pow((y_coordinate_diff[idx]), 2), 0.5 )/ time_coordinate_diff[idx] for idx in range(len(x_coordinate_diff))])
    velocity_list_diff = np.diff(velocity_list, n=1, axis=-1, append=0)

    acceleration_list = np.array([velocity_list_diff[idx] / time_coordinate_diff[idx] for idx in range(len(x_coordinate_diff))])
    acceleration_list_diff = np.diff(acceleration_list, n=1, axis=-1, append=0)

    jerk_list = np.array([acceleration_list_diff[idx] / time_coordinate_diff[idx] for idx in range(len(x_coordinate_diff))])

    velocity_list = np.reshape(velocity_list, (len(velocity_list), -1))
    acceleration_list = np.reshape(acceleration_list, (len(acceleration_list), -1))
    jerk_list = np.reshape(jerk_list, (len(jerk_list), -1))

    data = np.concatenate((data, velocity_list), axis=1)
    data = np.concatenate((data, acceleration_list), axis=1)
    data = np.concatenate((data, jerk_list),axis=1)

    data = data[int(m*0.05):int(m*0.9),:]
    if np.isnan(np.sum(data)):
        print('      this data has NAN value.')
    else:
        generate_aug_voxel_from_voxel(data, aug_method_dict, params_list, json_to_voxel_path)

