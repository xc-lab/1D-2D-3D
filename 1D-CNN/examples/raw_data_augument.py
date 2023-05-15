#  -*- coding: utf-8 -*-
'''
KT-00, PD-01
'''

import re
import os
import shutil
import random
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_utils.utils import get_augment_signal_from_json, balance_data_volums, read_signal_csv, generate_squeeze_from_signal, get_column_data_scale, normalization_array_data


def get_testing_squeeze_signal_from_json(data_path, pattern_lists, length, if_remove):

    print('JSON TO TEST SIGNAL')
    testee_type_files = os.listdir(os.path.join(data_path, 'test_data'))

    if not os.path.exists(os.path.join(data_path, 'testing_data')):  # '../data/testing_data
        os.mkdir(os.path.join(data_path, 'testing_data'))
    else:
        shutil.rmtree(os.path.join(data_path, 'testing_data'))
        os.mkdir(os.path.join(data_path, 'testing_data'))

    for l, testee_type in enumerate(testee_type_files):  # '../data/test_data
        if testee_type == 'HC':
            label_id = '00'
        elif testee_type == 'PD':
            label_id = '01'
        else:
            print('    %s class does not exit.' % (testee_type))
        testee_number_files = os.listdir(os.path.join(data_path, 'test_data', testee_type))

        for t, testee_number in enumerate(testee_number_files):  # '../data/test_data/KT
            testee_shape_files = os.listdir(os.path.join(data_path, 'test_data', testee_type, testee_number))

            for f, file in enumerate(testee_shape_files):  # '../data/raw_data/KT/KT1
                file_id = file[:-4]
                pattern_id = int(file[7:8])
                if pattern_id in pattern_lists:

                    json_file_path = os.path.join(data_path, 'test_data', testee_type, testee_number,
                                                  file)  # '../data/test_data/KT/KT1/KT-01_8D6834FB_pcontinue_2017-11-05_16_45_31___0095aa3c709f457cbe43e800ee7a299f.json
                    print('******INPUT FILE:' + json_file_path)
                    voxel_name = file_id + '_' + 'json' + '_' +label_id + '.npy' # 00026__1_1_json_00.npy

                    data = read_signal_csv(json_file_path, if_remove)  # [y,x,t,b,a,l,p,v]

                    m, n = data.shape
                    x_coordinate = data[:, 1]
                    y_coordinate = data[:, 0]
                    time_coordinate = data[:, 2]
                    x_coordinate_diff = np.diff(x_coordinate, n=1, axis=-1, append=0)
                    y_coordinate_diff = np.diff(y_coordinate, n=1, axis=-1, append=0)
                    time_coordinate_diff = np.diff(time_coordinate, n=1, axis=-1, append=0)
                    velocity_list = np.array([pow(pow((x_coordinate_diff[idx]), 2) + pow((y_coordinate_diff[idx]), 2), 0.5) / time_coordinate_diff[idx] for idx in range(len(x_coordinate_diff))])
                    velocity_list_diff = np.diff(velocity_list, n=1, axis=-1, append=0)
                    acceleration_list = np.array( [velocity_list_diff[idx] / time_coordinate_diff[idx] for idx in range(len(x_coordinate_diff))])
                    acceleration_list_diff = np.diff(acceleration_list, n=1, axis=-1, append=0)
                    jerk_list = np.array([acceleration_list_diff[idx] / time_coordinate_diff[idx] for idx in range(len(x_coordinate_diff))])

                    velocity_list = np.reshape(velocity_list, (len(velocity_list), -1))
                    acceleration_list = np.reshape(acceleration_list, (len(acceleration_list), -1))
                    jerk_list = np.reshape(jerk_list, (len(jerk_list), -1))

                    data = np.concatenate((data, velocity_list), axis=1)
                    data = np.concatenate((data, acceleration_list), axis=1)
                    data = np.concatenate((data, jerk_list), axis=1)

                    data = data[int(m * 0.05):int(m * 0.9), :]
                    if np.isnan(np.sum(data)):
                        print('      this data has NAN value.')
                    elif data.shape[0] > length:
                        squeeze_data = generate_squeeze_from_signal(data, length)

                        squeeze_data_1 = squeeze_data[:, 0:2]
                        squeeze_data_2 = squeeze_data[:, 6:7]
                        squeeze_data_3 = squeeze_data[:, 7:]
                        squeeze_data = np.hstack((squeeze_data_1, squeeze_data_2, squeeze_data_3))
                        #
                        # squeeze_data = squeeze_data[:, 0:2]

                        if squeeze_data.shape[0] == length:
                            squeeze_data_range = get_column_data_scale(squeeze_data)
                            normal_data = normalization_array_data(squeeze_data, squeeze_data_range)
                            print('        %s' % (str(normal_data.shape)))
                            print('        %s' % (os.path.join(data_path, 'testing_data', voxel_name)))
                            with open(os.path.join(data_path, 'testing_data', voxel_name),
                                      'wb') as f:  # '../data/testing_data/KT1_spiral_000_json_00.npy
                                np.save(f, normal_data)
                        else:
                            print('      LENGTH NOT %d !' % (length))






def get_aug_dataset(data_path, pattern_lists, length, if_remove):
    idx = 0
    print('JSON TO AUG SIGNAL')
    testee_type_files = os.listdir(os.path.join(data_path, 'raw_data'))
    if not os.path.exists(os.path.join(data_path, 'dataset')):  # '../data/dataset
        os.mkdir(os.path.join(data_path, 'dataset'))

    for l, testee_type in enumerate(testee_type_files):  # '../data/raw_data
        if not os.path.exists(os.path.join(data_path, 'dataset', testee_type)):  # '../data/dataset/KT
            os.mkdir(os.path.join(data_path, 'dataset', testee_type))
        testee_number_files = os.listdir(os.path.join(data_path, 'raw_data', testee_type))

        for t, testee_number in enumerate(testee_number_files):  # '../data/raw_data/KT
            if not os.path.exists(
                    os.path.join(data_path, 'dataset', testee_type, testee_number)):  # '../data/dataset/KT/KT1
                os.mkdir(os.path.join(data_path, 'dataset', testee_type, testee_number))
            testee_shape_files = os.listdir(os.path.join(data_path, 'raw_data', testee_type, testee_number))

            for f, file in enumerate(testee_shape_files):  # '../data/raw_data/KT/KT1
                file_id = file[:-4]
                pattern_id = int(file[7:8])
                if pattern_id in pattern_lists:
                    idx += 1
                    print(idx)
                    if not os.path.exists(os.path.join(data_path, 'dataset', testee_type, testee_number,
                                                       file_id)):  # '../data/dataset/KT/KT1/ptrace_000_json
                        os.mkdir(os.path.join(data_path, 'dataset', testee_type, testee_number,
                                              file_id))
                    if os.path.exists(os.path.join(data_path, 'dataset', testee_type, testee_number,
                                                   file_id,
                                                   'aug')):  # '../data/dataset/KT/KT1/ptrace_000_json/src
                        shutil.rmtree(os.path.join(data_path, 'dataset', testee_type, testee_number,
                                                   file_id, 'aug'))
                        os.mkdir(os.path.join(data_path, 'dataset', testee_type, testee_number,
                                              file_id, 'aug'))
                    else:
                        os.mkdir(os.path.join(data_path, 'dataset', testee_type, testee_number,
                                              file_id, 'aug'))
                    get_augment_signal_from_json(data_path, testee_type, testee_number, file_id, file,
                                                    aug_method_dict, params_list, length, if_remove)



def get_training_data_from_aug_dataset(path):

    print('AUG DATASET TO TRAINING DATASET')
    KT_n_train = 0
    KT_n_val = 0
    PD_n_train = 0
    PD_n_val = 0
    random_KT_PD_num = balance_data_volums(15, 16)
    training_data_path = os.path.join(path, 'training_data')

    if os.path.exists(training_data_path):
        shutil.rmtree(training_data_path)
        os.mkdir(training_data_path)
        os.mkdir(os.path.join(training_data_path, 'training'))
        os.mkdir(os.path.join(training_data_path, 'validation'))
    else:
        os.mkdir(training_data_path)
        os.mkdir(os.path.join(training_data_path, 'training'))
        os.mkdir(os.path.join(training_data_path, 'validation'))

    for l, testee_type in enumerate(os.listdir(os.path.join(path, 'dataset'))):
        if testee_type == 'HC':
            label_id = '00'
        elif testee_type == 'PD':
            label_id = '01'
        else:
            print('    %s class does not exit.' % (testee_type))
        for t, testee_number in enumerate(os.listdir(os.path.join(path, 'dataset', testee_type))):
            for n, file in enumerate(os.listdir(os.path.join(path, 'dataset', testee_type, testee_number))):
                for i, aug_file in enumerate(os.listdir(os.path.join(path, 'dataset', testee_type, testee_number, file, 'aug'))):

                    aug_file_path = os.path.join(path, 'dataset', testee_type, testee_number, file, 'aug', aug_file, 'squeeze_signal.npy')
                    aug_file_name = testee_number + '_' + file + '_' + aug_file + '_' + label_id + '.npy'
                    random_PD = random.random()
                    random_KT = random.random()

                    if testee_type == 'HC' and random_KT < random_KT_PD_num[testee_type]:
                        random_train_val = random.random()
                        if random_train_val < 0.9:
                            shutil.copy(aug_file_path, os.path.join(training_data_path, 'training', aug_file_name))
                            KT_n_train += 1
                        else:
                            shutil.copy(aug_file_path, os.path.join(training_data_path, 'validation', aug_file_name))
                            KT_n_val += 1
                    elif testee_type == 'PD' and random_PD < random_KT_PD_num[testee_type]:
                        random_train_val = random.random()
                        if random_train_val < 0.9:
                            shutil.copy(aug_file_path, os.path.join(training_data_path, 'training', aug_file_name))
                            PD_n_train += 1
                        else:
                            shutil.copy(aug_file_path, os.path.join(training_data_path, 'validation', aug_file_name))
                            PD_n_val += 1
                    print('    KT_:%d,%d, PD_:%d,%d.' % (KT_n_train, KT_n_val, PD_n_train, PD_n_val))


if __name__ == '__main__':

    params_list = {'src': 'copy',
                   'upsampling': [2,3,4],
                   'downsampling': [2,3],
                   'brightness_shift': [0.005, 0.01],
                   'brightness_scaling': [0.005, 0.01],
                   'jittering': [0.001, 0.0025, 0.005, 0.0075, 0.01],
                   }

    aug_method_dict = {1:'src', 2:'downsampling', 3:'upsampling', 6:'jittering'}
    # aug_method_dict = {1:'src', 4:'jittering', 5:'brightness_scaling'}
    # aug_method_dict = {1:'src', 4:'jittering'}


    length = 128
    pattern_lists = {1} #这里的1表示spiral形状
    if_remove = True

    path = '../data'
    get_aug_dataset(path, pattern_lists, length, if_remove)
    get_training_data_from_aug_dataset(path)
    get_testing_squeeze_signal_from_json(path, pattern_lists, length, if_remove)




