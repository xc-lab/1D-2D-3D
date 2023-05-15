#  -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from PIL import Image
import os
import csv
import shutil
from decimal import Decimal

from data_utils.augmention import *

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



def normalization_array_data(data):
    normalized_data = data.copy()
    for i in np.arange(data.shape[1]):
        min_value = np.min(data[:, i])
        max_value = np.max(data[:, i])
        normalized_data[:,i]= (normalized_data[:,i]-min_value)/(max_value-min_value)
    return normalized_data


def get_column_data_scale(data):
    min_scale = np.min(data, axis=0)
    max_scale = np.max(data, axis=0)
    data_scale = np.vstack((min_scale, max_scale))
    return data_scale



def get_colomn_scalled_difference(data):
    data_temp = data.copy()
    for i in np.arange(data.shape[1]):
        data_temp[:, i] = np.diff(data[:, i], n=1, axis=-1, append=0)
    return data_temp[:-1,:]


def generate_img_from_json(data, img_name, path, train=True):
    m, n = data.shape
    data = data[int(m*0.05):int(m*0.95),:]

    colors = data[:-1,3:6]
    colors = normalization_array_data(colors)

    x_coordinate = data[:, 1]
    y_coordinate = data[:, 0]
    time_coordinate = data[:, 2]

    x_coordinate_diff = np.diff(x_coordinate, n=1, axis=-1, append=0)[:-1]
    y_coordinate_diff = np.diff(y_coordinate, n=1, axis=-1, append=0)[:-1]
    time_coordinate_diff = np.diff(time_coordinate, n=1, axis=-1, append=0)[:-1]

    velocity_list = np.array(
        [pow(pow((x_coordinate_diff[idx]), 2) + pow((y_coordinate_diff[idx]), 2), 0.5) / time_coordinate_diff[idx] for idx in
         range(len(x_coordinate_diff))])
    # velocity_list_diff = np.diff(velocity_list, n=1, axis=-1, append=0)
    #
    # acceleration_list = np.array(
    #     [velocity_list_diff[idx] / time_coordinate_diff[idx] for idx in range(len(x_coordinate_diff))])
    # acceleration_list_diff = np.diff(acceleration_list, n=1, axis=-1, append=0)
    #
    # jerk_list = np.array(
    #     [acceleration_list_diff[idx] / time_coordinate_diff[idx] for idx in range(len(x_coordinate_diff))])[:-2]
    #
    # velocity_list = np.reshape(velocity_list, (len(velocity_list), -1))[:-2]
    # velocity_list = normalization_array_data(velocity_list)
    # acceleration_list = np.reshape(acceleration_list, (len(acceleration_list), -1))[:-2]
    # acceleration_list = normalization_array_data(acceleration_list)
    # jerk_list = np.reshape(jerk_list, (len(jerk_list), -1))
    # jerk_list = normalization_array_data(jerk_list)
    #
    # colors = np.hstack([velocity_list, acceleration_list, jerk_list])

    # cmap = plt.cm.get_cmap('viridis')
    # color_cmap = cmap(np.arange(cmap.N))
    # colors = np.ones((jerk_list.shape[0], 3)).astype(np.float)
    # for i in np.arange(jerk_list.shape[0]):
    #     weight = jerk_list[i][0]
    #     new_weight = int(Decimal(255 * weight).quantize(Decimal("1."), rounding = "ROUND_HALF_UP"))
    #     colors[i, :] = color_cmap[new_weight, :3]

    p = velocity_list
    p = np.reshape(p, (len(p), -1))
    p = normalization_array_data(p)
    press = p

    # p = data[:-3, 5]
    # p = np.reshape(p, (len(p), -1))
    # p = normalization_array_data(p)
    # press = p

    X = np.array(data[:-3, 1])
    X = np.reshape(X, (len(X), -1))
    Y = np.array(data[:-3, 0])
    Y = np.reshape(Y, (len(Y), -1))

    x_range = (np.ceil(np.max(X) - np.min(X)))
    y_range = (np.ceil(np.max(Y) - np.min(Y)))
    plt.figure(figsize=(int(x_range/400), int(y_range/400)))
    plt.scatter(X, Y, s=press*40, c=colors)
    # plt.scatter(X, Y, s=press*40)

    plt.axis('off')
    # plt.show()
    if train:
        plt.savefig(os.path.join(path, 'image.jpg'), dpi=1000, bbox_inches='tight', pad_inches=0)
        plt.close('all')
    else:
        plt.savefig(os.path.join(path, img_name+'.jpg'), dpi=1000, bbox_inches='tight', pad_inches=0)
        plt.close('all')



def check_save_data_path(path, sub_path, image):
    data_path = os.path.join(path, sub_path)  # '../data/dataset/KT/KT1/ptrace_000_json/aug/src
    print('        '+os.path.join(data_path, 'image.jpg'))
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
        os.mkdir(data_path)
    else:
        os.mkdir(data_path)
    new_image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(data_path, 'image.jpg'), new_image) # '../data/dataset/KT/KT1/ptrace_000_json/aug/src/image.jpg


def generate_aug_image_from_image(image, aug_method_dict, params_list, path):
    for method_index in aug_method_dict:
        aug_method = aug_method_dict[method_index]

        if (aug_method == 'src'):
            check_save_data_path(path, 'src', image)


        elif (aug_method == 'mirror'):
            for flip_type in params_list['mirror']:
                if flip_type == 'bottom-up':
                    aug_img = flip_img(image, 'V')
                    check_save_data_path(path, 'mirror_v', aug_img)
                elif flip_type == 'left-right':
                    aug_img = flip_img(image, 'H')
                    check_save_data_path(path, 'mirror_h', aug_img)


        elif (aug_method == 'rescale'):
            for scale_factor in params_list['rescale']:
                aug_img = rescale_transform(image, scale_factor)
                check_save_data_path(path, 'rescale_{}'.format(scale_factor), aug_img)


        elif (aug_method == 'mosaic'):
            for mosaic_factor in params_list['mosaic']:
                aug_img = mosaic_transform(image, mosaic_factor)
                check_save_data_path(path, 'mosaic_{}'.format(mosaic_factor), aug_img)


        elif (aug_method == 'rotation'):
            for rotate_angle in params_list['rotation']:
                aug_img = rotation_transform(image, rotate_angle)
                check_save_data_path(path, 'rotation_{}'.format(rotate_angle), aug_img)


        elif (aug_method == 'brightness_shift'):
            for max_shift_value in params_list['brightness_shift']:
                aug_img = brightness_shift(image, max_shift_value)
                check_save_data_path(path, 'brightness_shift_{}'.format(max_shift_value), aug_img)


        elif (aug_method == 'brightness_scaling'):
            for max_scale_value in params_list['brightness_scaling']:
                aug_img = brightness_scaling(image, max_scale_value)
                check_save_data_path(path, 'brightness_scaling_{}'.format(max_scale_value), aug_img)


        elif (aug_method == 'gamma_correction'):
            for gamma_val in params_list['gamma_correction']:
                aug_img = gamma_correction(image, gamma_val)
                check_save_data_path(path, 'Gamma_{}'.format(gamma_val), aug_img)


        elif (aug_method == 'enhancement'):
            for enhance_method in params_list['enhancement']:
                if (enhance_method == 'HE'):
                    aug_img = hist_equation_color(image)
                    check_save_data_path(path, 'HE', aug_img)
                elif (enhance_method == 'CLAHE'):
                    aug_img = adapt_hist_equation_color(image)
                    check_save_data_path(path, 'AHE', aug_img)


        elif (aug_method == 'shearing'):
            for shaer_val in params_list['shearing']:
                is_show_grid = False
                aug_img = horizontal_shear(image, shaer_val, is_show_grid)
                check_save_data_path(path, 'shearing_{}'.format(shaer_val), aug_img)


        elif (aug_method == 'elastic_transform'):
            for elastic_level in params_list['elastic_transform']:
                is_show_grid = False
                aug_img = elastic_transform(image, elastic_level, is_show_grid)
                check_save_data_path(path, 'elastic_deform_{}'.format(elastic_level), aug_img)


        elif (aug_method == 'add-noise'):
            for noise_type in params_list['add-noise']:
                aug_img = add_noisy(image, noise_type)
                check_save_data_path(path, 'add_' + noise_type + '_noise', aug_img)
        else:
            print('          %s augmentation method does not exist.' % (aug_method))




def balance_data_volums(num_kt, num_pd):
    random_KT_PD_num = {}
    if num_kt < num_pd:
        random_KT_PD_num['HC'] = 1.0
        random_KT_PD_num['PD'] = num_kt / num_pd
    else:
        random_KT_PD_num['HC'] = num_pd / num_kt
        random_KT_PD_num['PD'] = 1.0
    return random_KT_PD_num







