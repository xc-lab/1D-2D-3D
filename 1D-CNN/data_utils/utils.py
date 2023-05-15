import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import os
import shutil
import csv
from data_utils.augmentation import *



# def get_colomn_scalled_difference(data, order, scale, dim_id):
#     if dim_id:
#         data_temp = data.copy()
#         for i in np.arange(data.shape[1]):
#             if i in dim_id:
#                 data_temp[:, i] = np.diff(data[:, i], n=order, axis=-1, append=0)
#                 data_temp[:, i] = scale[i] * data_temp[:, i]
#         return data_temp[1:-2,:]
#     else:
#         return data

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


def check_save_data_path(path, sub_path, data, length):
    data_1 = data[:,0:2]
    data_2 = data[:,6:7]
    data_3 = data[:,7:]
    data = np.hstack((data_1, data_2, data_3))
    # data = data_1
    if data.shape[0] == length:
        data_path = os.path.join(path, sub_path)  # '../data/dataset/KT/KT1/ptrace_000_json/aug/src
        print('      OUTPUT:'+os.path.join(data_path, 'squeeze_signal.npy'))
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
            os.mkdir(data_path)
        else:
            os.mkdir(data_path)
        print('        %s'%(str(data.shape)))
        data_range = get_column_data_scale(data)
        normal_data = normalization_array_data(data, data_range)
        with open(os.path.join(data_path, 'squeeze_signal.npy'), 'wb') as f:# '../data/dataset/KT/KT1/ptrace_000_json/aug/src/squeeze_signal.npy
            np.save(f, normal_data) #[a,l,p,x,y,t,v]
    else:
        print('      LENGTH NOT %d !'%(length))


def generate_squeeze_from_signal(data, length):
     num = length
     temp_data = np.zeros((1, data.shape[1]))
     idx_list = np.arange(0, data.shape[0], int(data.shape[0]/num)).tolist()
     i_list = np.arange(len(idx_list))
     for i in i_list:
         if i == i_list[-1] or i >= (length-1):
            patch_data = data[idx_list[i]:, :]
            mean_scale = np.mean(patch_data, axis=0)
            temp_data = np.vstack((temp_data, mean_scale))
            break
         else:
            patch_data = data[idx_list[i]:idx_list[i+1], :]
            mean_scale = np.mean(patch_data, axis=0)
            temp_data = np.vstack((temp_data, mean_scale))
     squeeze_data = temp_data[1:, :]
     return squeeze_data


def get_column_data_scale(data):
    min_scale = np.min(data, axis=0)
    max_scale = np.max(data, axis=0)
    data_scale = np.vstack((min_scale, max_scale))
    return data_scale


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


def generate_aug_signal_from_signal(data, aug_method_dict, params_list, path, length):

    for method_index in aug_method_dict:
        aug_method = aug_method_dict[method_index]

        if (aug_method == 'src'):
            aug_data = data
            squeeze_data = generate_squeeze_from_signal(aug_data, length)
            check_save_data_path(path, 'src', squeeze_data, length)


        elif (aug_method == 'upsampling'):
            for sampling_factor in params_list['upsampling']:
                aug_data = up_sampling_transform(data, sampling_factor)
                squeeze_data = generate_squeeze_from_signal(aug_data, length)
                check_save_data_path(path, 'upsampling_{}'.format(sampling_factor), squeeze_data, length)


        elif (aug_method == 'downsampling'):
            for sampling_factor in params_list['downsampling']:
                aug_data = down_sampling_transform(data, sampling_factor)
                squeeze_data = generate_squeeze_from_signal(aug_data, length)
                check_save_data_path(path, 'downsampling_{}'.format(sampling_factor), squeeze_data, length)



        elif (aug_method == 'brightness_shift'):
            for max_shift_value in params_list['brightness_shift']:
                aug_data = brightness_shift(data, max_shift_value)
                squeeze_data = generate_squeeze_from_signal(aug_data, length)
                check_save_data_path(path, 'brightness_shift_{}'.format(max_shift_value), squeeze_data, length)


        elif (aug_method == 'brightness_scaling'):
            for max_scale_value in params_list['brightness_scaling']:
                aug_data = brightness_scaling(data, max_scale_value)
                squeeze_data = generate_squeeze_from_signal(aug_data, length)
                check_save_data_path(path, 'brightness_scaling_{}'.format(max_scale_value), squeeze_data, length)


        elif (aug_method == 'jittering'):
            for max_jitter_value in params_list['jittering']:
                aug_data = location_jittering(data, max_jitter_value)
                squeeze_data = generate_squeeze_from_signal(aug_data, length)
                check_save_data_path(path, 'location_jittering_{}'.format(max_jitter_value), squeeze_data, length)
        else:
            print('          %s augmentation method does not exist.' % (aug_method))



def get_augment_signal_from_json(data_path, testee_type, testee_number, file_id, file, aug_method_dict, params_list, length, if_remove):
    json_file_path = os.path.join(data_path, 'raw_data', testee_type, testee_number, file)  # '../data/raw_data/KT/KT1/KT-01_8D6834FB_pcontinue_2017-11-05_16_45_31___0095aa3c709f457cbe43e800ee7a299f.json
    print('******INPUT FILE:' + json_file_path)

    json_to_signal_path = os.path.join(data_path, 'dataset', testee_type, testee_number, file_id, 'aug')  # '../data/dataset/KT/KT1/ptrace_000_json/aug

    data = read_signal_csv(json_file_path, if_remove) #[y,x,t,b,a,l,p]

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

    data = np.concatenate((data, velocity_list),axis=1)
    data = np.concatenate((data, acceleration_list),axis=1)
    data = np.concatenate((data, jerk_list),axis=1)

    data = data[int(m*0.05):int(m*0.9),:]
    if np.isnan(np.sum(data)):
        print('      this data has NAN value.')
    elif data.shape[0]>length:
        generate_aug_signal_from_signal(data, aug_method_dict, params_list, json_to_signal_path, length)
