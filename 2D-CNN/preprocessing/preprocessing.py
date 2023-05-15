#  -*- coding: utf-8 -*-
'''
Contains all data preprocessing work, label PD is 1, KT is 0.
'''
import os
import shutil
import re
import json
import random
import numpy as np


from data_utils.utils import *

def get_testing_img_from_svc(data_path, pattern_lists, if_remove):

    print('SVC TO TEST IMG')
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

            for f, file in enumerate(testee_shape_files):  # '../data/test_data/KT/KT1
                file_id = file[:-4]
                pattern_id = int(file[7:8])
                if pattern_id in pattern_lists:

                        json_file_path = os.path.join(data_path, 'test_data', testee_type, testee_number, file)  # '../data/test_data/KT/KT1/KT-01_8D6834FB_pcontinue_2017-11-05_16_45_31___0095aa3c709f457cbe43e800ee7a299f.json
                        print('    '+json_file_path)

                        raw_data = read_signal_csv(json_file_path, if_remove)
                        data = np.delete(raw_data, [3], axis=1)

                        data_range = get_column_data_scale(data)
                        if np.isnan(np.sum(data)):
                            print('         *****this data has NAN value %s.'%(json_file_path))
                        elif (np.count_nonzero(data_range[0,:]-data_range[1,:]) != 6):
                            print('         *****this data has some feature values unchanged %s.' %(json_file_path))
                        else:
                            img_name = testee_type + '_' + file_id + '_' + label_id
                            generate_img_from_json(data, img_name, os.path.join(data_path, 'testing_data'), train=False)


def get_img_from_svc(data_path, pattern_lists, if_remove):

    print('SVC TO IMG')
    testee_type_files = os.listdir(os.path.join(data_path, 'raw_data'))

    if not os.path.exists(os.path.join(data_path, 'dataset')):  # '../data/dataset
        os.mkdir(os.path.join(data_path, 'dataset'))

    for l, testee_type in enumerate(testee_type_files):  # '../data/raw_data
        if not os.path.exists(os.path.join(data_path, 'dataset', testee_type)):  # '../data/dataset/KT
            os.mkdir(os.path.join(data_path, 'dataset', testee_type))
        testee_number_files = os.listdir(os.path.join(data_path, 'raw_data', testee_type))

        for t, testee_number in enumerate(testee_number_files):  # '../data/raw_data/KT
            if not os.path.exists(os.path.join(data_path, 'dataset', testee_type, testee_number)):  # '../data/dataset/KT/KT1
                os.mkdir(os.path.join(data_path, 'dataset', testee_type, testee_number))
            testee_shape_files = os.listdir(os.path.join(data_path, 'raw_data', testee_type, testee_number))

            for f, file in enumerate(testee_shape_files):  # '../data/raw_data/KT/KT1
                file_id = file[:-4]
                pattern_id = int(file[7:8])
                if pattern_id in pattern_lists:
                    json_file_path = os.path.join(data_path, 'raw_data', testee_type, testee_number, file)  # '../data/raw_data/KT/KT1/KT-01_8D6834FB_pcontinue_2017-11-05_16_45_31___0095aa3c709f457cbe43e800ee7a299f.json
                    print('    '+json_file_path)

                    raw_data = read_signal_csv(json_file_path, if_remove)
                    data = np.delete(raw_data, [3], axis=1)
                    data_range = get_column_data_scale(data)
                    if np.isnan(np.sum(data)):
                        print('         *****this data has NAN value %s.'%(json_file_path))
                    elif (np.count_nonzero(data_range[0,:]-data_range[1,:]) != 6):
                        print('         *****this data has some feature values unchanged %s.' %(json_file_path))
                    else:
                        if not os.path.exists(os.path.join(data_path, 'dataset', testee_type, testee_number, file_id + '_' + 'svc')):  # '../data/dataset/KT/KT1/ptrace_000_json
                            os.mkdir(os.path.join(data_path, 'dataset', testee_type, testee_number, file_id + '_' + 'svc'))
                        json_to_img_path = os.path.join(data_path, 'dataset', testee_type, testee_number, file_id + '_' + 'svc', 'src')  # json数据生成的图片的保存路径
                        if os.path.exists(json_to_img_path):  # '../data/dataset/HC/00026/00026__1_1_svc/src
                            shutil.rmtree(json_to_img_path)
                            os.mkdir(json_to_img_path)
                        else:
                            os.mkdir(json_to_img_path)
                        generate_img_from_json(data, '1', json_to_img_path, train=True)




def get_augment_from_img(path, aug_method_dict, params_list):
    print('IMAGE TO AUGMENT')
    testee_type_files = os.listdir(os.path.join(path, 'dataset'))
    for l, testee_type in enumerate(testee_type_files):  # '../data/dataset
        testee_number_files = os.listdir(os.path.join(path, 'dataset', testee_type))

        for t, testee_number in enumerate(testee_number_files):  # '../data/dataset/KT
            testee_shape_files = os.listdir(os.path.join(path, 'dataset', testee_type, testee_number))

            for f, file in enumerate(testee_shape_files):  # '../data/raw_data/KT/KT1
                src_path = os.path.join(path, 'dataset', testee_type, testee_number, file)  # '../data/dataset/KT/KT1/ptrace_000_json
                print('    ' + src_path)

                img = cv2.imread(os.path.join(src_path, 'src', 'image.jpg'))
                aug_path = os.path.join(src_path, 'aug')  # '../data/dataset/KT/KT1/ptrace_000_json/aug
                if os.path.exists(aug_path):
                    shutil.rmtree(aug_path)
                    os.mkdir(aug_path)
                else:
                    os.mkdir(aug_path)
                generate_aug_image_from_image(img, aug_method_dict, params_list, aug_path)




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

                    aug_file_path = os.path.join(path, 'dataset', testee_type, testee_number, file, 'aug', aug_file, 'image.jpg')
                    aug_file_name = testee_number + '_' + file + '_' + aug_file + '_' + label_id + '.jpg'
                    random_PD = random.random()
                    random_KT = random.random()

                    if testee_type == 'HC' and random_KT < random_KT_PD_num[testee_type]:
                        random_train_val = random.random()
                        if random_train_val < 0.8:
                            shutil.copy(aug_file_path, os.path.join(training_data_path, 'training', aug_file_name))
                            KT_n_train += 1
                        else:
                            shutil.copy(aug_file_path, os.path.join(training_data_path, 'validation', aug_file_name))
                            KT_n_val += 1
                    elif testee_type == 'PD' and random_PD < random_KT_PD_num[testee_type]:
                        random_train_val = random.random()
                        if random_train_val < 0.8:
                            shutil.copy(aug_file_path, os.path.join(training_data_path, 'training', aug_file_name))
                            PD_n_train += 1
                        else:
                            shutil.copy(aug_file_path, os.path.join(training_data_path, 'validation', aug_file_name))
                            PD_n_val += 1
                    print('    KT_patch_img:%d,%d, PD_patch_img:%d,%d.' % (KT_n_train, KT_n_val, PD_n_train, PD_n_val))




if __name__ == '__main__':
    '''label——KT:0, PD:1'''
    # augmentation factor
    params_list = {'src': 'copy',
                   'mirror': ['bottom-up', 'left-right'],
                   'rescale': [0.1, 0.2, 0.25,0.5, 0.7, 0.75, 0.8, 1.5, 2],
                   'mosaic': [2, 3, 4],
                   'rotation': [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],

                   'brightness_shift': [0.01, 0.06, 0.1],
                   'brightness_scaling': [0.06, 0.1],
                   'gamma_correction': [0.5, 0.75, 1.25, 1.5],
                   'enhancement': ['HE', 'CLAHE'],
                   'add-noise': ['gauss', 's_p', 'poisson', 'speckle'],

                   'shearing': [0.02],
                   'elastic_transform': [0.02],
                   }

    aug_method_dict = {1: 'mirror', 4: 'shearing', 5:'src',  7:'brightness_scaling', 8:'brightness_shift'}

    path = '../data'
    pattern_lists = {1}
    if_remove = True

    get_img_from_svc(path, pattern_lists, if_remove)
    get_augment_from_img(path, aug_method_dict, params_list)
    get_training_data_from_aug_dataset(path)
    #
    get_testing_img_from_svc(path, pattern_lists, if_remove)


