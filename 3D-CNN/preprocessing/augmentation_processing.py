import os

from data_utils.utils import *

def get_testing_3d_voxel_from_json(data_path, pattern_list, if_remove):

    print('JSON TO TEST voxel')
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
                if pattern_id in pattern_list:
                    pattern = 'spiral'
                    json_file_path = os.path.join(data_path, 'test_data', testee_type, testee_number,
                                                  file)  # '../data/test_data/KT/KT1/KT-01_8D6834FB_pcontinue_2017-11-05_16_45_31___0095aa3c709f457cbe43e800ee7a299f.json
                    print('******INPUT FILE:' + json_file_path)

                    voxel_name = file_id + '_' + label_id + '.npy' # KT1_spiral_000_json_00.npy

                    data = read_signal_csv(json_file_path, if_remove)

                    m, n = data.shape
                    x_coordinate = data[:, 1]
                    y_coordinate = data[:, 0]
                    time_coordinate = data[:, 2]

                    x_coordinate_diff = np.diff(x_coordinate, n=1, axis=-1, append=0)
                    y_coordinate_diff = np.diff(y_coordinate, n=1, axis=-1, append=0)
                    time_coordinate_diff = np.diff(time_coordinate, n=1, axis=-1, append=0)

                    velocity_list = np.array([pow(pow((x_coordinate_diff[idx]), 2) + pow((y_coordinate_diff[idx]),
                                                                                         2), 0.5) /
                                              time_coordinate_diff[idx] for idx in range(len(x_coordinate_diff))])
                    velocity_list_diff = np.diff(velocity_list, n=1, axis=-1, append=0)

                    acceleration_list = np.array(
                        [velocity_list_diff[idx] / time_coordinate_diff[idx] for idx in range(len(x_coordinate_diff))])
                    acceleration_list_diff = np.diff(acceleration_list, n=1, axis=-1, append=0)

                    jerk_list = np.array([acceleration_list_diff[idx] / time_coordinate_diff[idx] for idx in
                                          range(len(x_coordinate_diff))])

                    velocity_list = np.reshape(velocity_list, (len(velocity_list), -1))
                    acceleration_list = np.reshape(acceleration_list, (len(acceleration_list), -1))
                    jerk_list = np.reshape(jerk_list, (len(jerk_list), -1))


                    data = np.concatenate((data, velocity_list), axis=1)
                    data = np.concatenate((data, acceleration_list), axis=1)
                    data = np.concatenate((data, jerk_list), axis=1)

                    data = data[int(m * 0.05):int(m * 0.9), :]
                    data_range = get_column_data_scale(data)
                    if np.isnan(np.sum(data)):
                        print('      this data has NAN value.')
                    else:
                        voxel_array = generate_voxel_from_json(data, data_range)
                        print('        %s' % (str(voxel_array.shape)))
                        print('        %s'%(os.path.join(data_path, 'testing_data', voxel_name)))
                        with open(os.path.join(data_path, 'testing_data', voxel_name), 'wb') as f:  # '../data/testing_data/KT1_spiral_000_json_00.npy
                            np.save(f, voxel_array)





def get_3d_voxel_from_json(data_path, pattern_list, if_remove):
    idx = 0
    print('JSON TO AUG VOXEL')
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
                if pattern_id in pattern_list:
                    pattern = 'spiral'

                    idx += 1
                    print(idx)
                    if not os.path.exists(os.path.join(data_path, 'dataset', testee_type, testee_number, file_id )):  # '../data/dataset/KT/KT1/ptrace_000_json
                        os.mkdir(os.path.join(data_path, 'dataset', testee_type, testee_number, file_id))
                    if os.path.exists(os.path.join(data_path, 'dataset', testee_type, testee_number,
                                                   file_id, 'aug')):  # '../data/dataset/KT/KT1/ptrace_000_json/src
                        shutil.rmtree(os.path.join(data_path, 'dataset', testee_type, testee_number,
                                                   file_id, 'aug'))
                        os.mkdir(os.path.join(data_path, 'dataset', testee_type, testee_number,
                                              file_id, 'aug'))
                    else:
                        os.mkdir(os.path.join(data_path, 'dataset', testee_type, testee_number,
                                              file_id, 'aug'))

                    get_augment_voxel_from_json(data_path, testee_type, testee_number, pattern, file_id, file, aug_method_dict, params_list, if_remove)




def get_training_data_from_aug_dataset(path):

    print('AUG DATASET TO TRAINING DATASET')
    KT_n_train = 0
    KT_n_val = 0
    PD_n_train = 0
    PD_n_val = 0
    random_KT_PD_num = balance_data_volums(28, 29)
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

                    aug_file_path = os.path.join(path, 'dataset', testee_type, testee_number, file, 'aug', aug_file, '3d_voxel.npy')
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


if __name__=='__main__':
    # GENERATE 3D DATA FOR EVERY TEST
    params_list = {'src': 'copy',
                   'upsampling': [2,3,4,5],
                   'downsampling': [2],
                   'rotation': [np.pi / 2, np.pi, np.pi * 3 / 2],
                   'brightness_shift': [0.001, 0.0025, 0.005, 0.0075,
                                        0.01],
                   'brightness_scaling': [0.001, 0.0025, 0.005, 0.0075,
                                        0.01],
                   'jittering': [0.0001, 0.00025, 0.0005, 0.00075, 0.001],
                   }

    aug_method_dict = {1: 'src', 2: 'downsampling', 3: 'upsampling', 4:'rotation', 5: 'brightness_shift', 6: 'brightness_scaling',
                       7: 'jittering'}
    #
    # aug_method_dict = {1: 'src', 2: 'downsampling', 3: 'upsampling', 6: 'brightness_shift', 7: 'brightness_scaling',
    #                    }

    data_path = '../data'  # need on DWPD json failid
    pattern_list = [1]
    if_remove = True

    get_3d_voxel_from_json(data_path, pattern_list, if_remove)
    get_training_data_from_aug_dataset(data_path)
    get_testing_3d_voxel_from_json(data_path, pattern_list, if_remove)