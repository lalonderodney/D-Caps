'''
Diagnosing Colorectal Polyps in the Wild with Capsule Networks (D-Caps)
Original Paper by Rodney LaLonde, Pujan Kandel, Concetto Spampinato, Michael B. Wallace, and Ulas Bagci
Paper published at ISBI 2020: arXiv version (https://arxiv.org/abs/2001.03305)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file handles everything data related: Loading the data, splitting it, etc.
'''

from __future__ import print_function

from collections import Counter
import os
from glob import glob

import csv
import cv2
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from utils import safe_mkdir

debug = False

def load_data(root, exp_name, exp=0, split=0, k_folds=4, val_split=0.1):
    # Main functionality of loading and spliting the data
    def _load_data():
        with open(os.path.join(root, 'split_lists', exp_name, 'train_split_' + str(split) + '.csv'), 'r') as f:
            reader = csv.reader(f)
            training_list = list(reader)
        with open(os.path.join(root, 'split_lists', exp_name, 'test_split_' + str(split) + '.csv'), 'r') as f:
            reader = csv.reader(f)
            test_list = list(reader)
        X, y = np.hsplit(np.asarray(training_list), 2)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=12, stratify=y)
        new_train_list, val_list = np.hstack((X_train, y_train)), np.hstack((X_val, y_val))
        return new_train_list, val_list, np.asarray(test_list)

    # Try-catch to handle calling split data before load only if files are not found.
    try:
        new_training_list, validation_list, testing_list = _load_data()
        return new_training_list, validation_list, testing_list
    except:
        # Create the training and test splits if not found
        split_data(root, exp_name, exp_num=exp, num_splits=k_folds)
        try:
            new_training_list, validation_list, testing_list = _load_data()
            return new_training_list, validation_list, testing_list
        except Exception as e:
            print(e)
            print('Failed to load data, see load_data in load_polyp_data.py')
            exit(1)


def split_data_for_flow(root, out_dir, exp_name, resize_option, resize_shape, train_list, val_list, test_list):
    def _load_imgs(data_list, phase):
        data_info_list = []
        for patient_num_label in tqdm(data_list, desc=phase):
            files = []
            for ext in ('*.jpg', '*.JPG', '*.tif', '*.tiff', '*.png', '*.PNG'):
                files.extend(sorted(glob(os.path.join(root, patient_num_label[0], ext).replace('\\', '/'))))
            if not files:
                print('WARNING: No Images found in {}. Ensure the path is set up properly in the compute '
                      'class samples function in load polyp data.'.format(os.path.join(root, patient_num_label[0])))
            for f in files:
                img = cv2.imread(f)
                try:
                    img = img.astype(np.float32)
                except:
                    print('Unable to load image: {}. Please check the file for corruption.'.format(f))
                    continue
                data_info_list.append([f, patient_num_label[1], img.shape[0], img.shape[1]])
        # Balance sample amounts
        np_data_list = np.asarray(data_info_list)
        if phase == 'Load_train':
            n_classes = len(np.unique(np_data_list[:,1]))
            max_samples = 0
            split_np_list = []
            for n in range(n_classes):
                split_np_list.append(np_data_list[np_data_list[:, 1] == '{}'.format(n)])
                amt = len(split_np_list[n])
                if amt > max_samples:
                    max_samples = amt

            out_list = np.empty((n_classes * max_samples,5), dtype='|S255')
            for n in range(n_classes):
                res_lis = np.resize(split_np_list[n],(max_samples,4))
                out_list[n*max_samples:(n+1)*max_samples,:] = np.hstack((res_lis, np.expand_dims(res_lis[:,0],-1)))

            counts = Counter(out_list[:, 4])
            for s, num in tqdm(counts.items(), desc='Renaming duplicate images'):
                if num > 1:  # ignore strings that only appear once
                    for suffix in range(1, num + 1):  # suffix starts at 1 and increases by 1 each time
                        out_list[out_list[:, 4].tolist().index(s), 4] = \
                            '{}_{}.{}'.format(s.decode('utf-8').replace('\\', '/')[:-4],
                                              suffix, s.decode('utf-8').replace('\\', '/')[-3:])  # replace each appearance of s

            return out_list
        else:
            return np.hstack((np_data_list, np.expand_dims(np_data_list[:,0],-1)))


    def _compute_out_shape(height_list, width_list):
        if resize_option == 'resize_max':
            out_shape = [np.max(height_list), np.max(width_list)]
        elif resize_option == 'resize_min' or resize_option == 'crop_min':
            out_shape = [np.min(height_list), np.min(width_list)]
        elif resize_option == 'resize_avg':
            out_shape = [int(np.mean(height_list)), int(np.mean(width_list))]
        elif resize_option == 'resize_std':
            out_shape = resize_shape
        else:
            raise NotImplementedError(
                'Error: Encountered resize choice which is not implemented in load_polyp_data.py.')
        if resize_shape[0] is not None:
            out_shape[0] = resize_shape[0]
        if resize_shape[1] is not None:
            out_shape[1] = resize_shape[1]
        return out_shape[0], out_shape[1]

    def _random_crop(img, crop_shape, mask=None):
        assert img.shape[2] == 3
        height, width = img.shape[0], img.shape[1]
        dy, dx = crop_shape
        if dy is None:
            dy = height
        if dx is None:
            dx = width
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        if mask is not None:
            return [img[y:(y+dy), x:(x+dx), :], mask[y:(y+dy), x:(x+dx)]]
        else:
            return [img[y:(y + dy), x:(x + dx), :]]


    def _save_imgs(lst, phase, hei, wid):
        class_list = exp_name.split('vs')
        class_map = dict()
        for k, v in enumerate(class_list):
            class_map[k] = '{}_{}'.format(k,v)
            try:
                safe_mkdir(os.path.join(out_dir, phase, class_map[k]))
            except:
                pass
        for i, f in enumerate(tqdm(lst, desc='Creating {} images'.format(phase))):
            img_out_name = os.path.join(out_dir, phase, class_map[int(f[1])],
                                '{}_{}.jpg'.format(os.path.basename(os.path.dirname(f[4])),
                                                   os.path.basename(f[4])[:-4])).replace('\\', '/')
            if not os.path.isfile(img_out_name):
                try:
                    im = cv2.imread(f[0].decode('utf-8'))
                except AttributeError:
                    im = cv2.imread(f[0])
                try:
                    im = im.astype(np.float32)
                except:
                    print('Unable to load image: {}. Please check the file for corruption.'.format(f[0]))
                    continue
                if im.shape[0] != hei or im.shape[1] != wid:
                    if resize_option == 'crop_min':
                        out_im = _random_crop(im, (hei,wid))
                    else:
                        out_im = resize(im, (hei,wid), mode='reflect', preserve_range=True)
                else:
                    out_im = im
                cv2.imwrite(img_out_name, out_im)


    def _compute_num_images():
        n_train = len(glob(os.path.join(out_dir, 'train', '*', '*.jpg')))
        n_val = len(glob(os.path.join(out_dir, 'val', '*', '*.jpg')))
        n_test = len(glob(os.path.join(out_dir, 'test', '*', '*.jpg')))
        return n_train, n_val, n_test


    train_info_array = np.asarray(_load_imgs(train_list, 'Load_train'))
    val_info_array = np.asarray(_load_imgs(val_list, 'Load_val'))
    test_info_array = np.asarray(_load_imgs(test_list, 'Load_test'))
    train_height, train_width = _compute_out_shape(train_info_array[:,2].astype(int), train_info_array[:,3].astype(int))
    val_height, val_width = _compute_out_shape(val_info_array[:,2].astype(int), val_info_array[:,3].astype(int))
    test_height, test_width = _compute_out_shape(test_info_array[:,2].astype(int), test_info_array[:,3].astype(int))

    _save_imgs(train_info_array, 'train', train_height, train_width)
    _save_imgs(val_info_array, 'val', val_height, val_width)
    _save_imgs(test_info_array, 'test', test_height, test_width)

    num_train, num_val, num_test = _compute_num_images()

    return num_train, [train_height, train_width], num_val,  [val_height, val_width], \
           num_test, [test_height, test_width]


def split_data(root_path, exp_name, exp_num, num_splits=4):
    patient_list = []
    patient_list.extend(sorted(glob(os.path.join(root_path,'Images','*', '*'))))

    assert len(patient_list) != 0, 'Unable to find any files in {}'.format(os.path.join(root_path,'Images','*','*'))

    label_list = []
    for patient_num in patient_list:
        img_type = os.path.basename(os.path.dirname(patient_num))
        if img_type == 'Normal':
            label_list.append(0)
        elif img_type == 'HP' or img_type == 'Hyperplastic':
            label_list.append(1)
        elif img_type == 'Serrated' or img_type == 'SSA':
            label_list.append(2)
        elif img_type == 'TA' or img_type == 'TVA' or img_type == 'Adenoma':
            label_list.append(3)
        elif img_type == 'Cancer':
            label_list.append(4)
        elif img_type == 'NewAdenomas':
            label_list.append(5)
            pass # This is a holdout testing set. Do not add to training, val, or testing. Only task after cross-validation is complete.
        else:
            raise Exception('Encountered unknown image type: {}'.format(img_type))

    outdir = os.path.join(root_path, 'split_lists', exp_name)
    try:
        safe_mkdir(outdir)
    except:
        pass

        patient_list = np.asarray(patient_list)
    label_list = np.asarray(label_list)

    if exp_num == 0:
        to_delete = np.append(np.argwhere(label_list==0), np.append(np.argwhere(label_list==2), np.append(np.argwhere(label_list==4), np.argwhere(label_list==5))))
        final_img_list = np.delete(patient_list, to_delete)
        final_label_list = np.delete(label_list, to_delete)
        final_label_list[final_label_list==1] = 0
        final_label_list[final_label_list==3] = 1
    elif exp_num == 1:
        to_delete = np.append(np.argwhere(label_list==0), np.append(np.argwhere(label_list==4), np.argwhere(label_list==5)))
        final_img_list = np.delete(patient_list, to_delete)
        final_label_list = np.delete(label_list, to_delete)
        final_label_list[final_label_list==1] = 0
        final_label_list[final_label_list==2] = 1
        final_label_list[final_label_list==3] = 1
    elif exp_num == 2:
        to_delete = np.append(np.argwhere(label_list==0), np.append(np.argwhere(label_list==3), np.append(np.argwhere(label_list==4), np.argwhere(label_list==5))))
        final_img_list = np.delete(patient_list, to_delete)
        final_label_list = np.delete(label_list, to_delete)
        final_label_list[final_label_list==1] = 0
        final_label_list[final_label_list==2] = 1
    else:
        raise Exception('Experiment number undefined.')

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=12)
    n = 0
    for train_index, test_index in skf.split(final_img_list, final_label_list):
        with open(os.path.join(outdir,'train_split_' + str(n) + '.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in train_index:
                writer.writerow([final_img_list[i].split(root_path)[1].replace('\\', '/'), final_label_list[i]])
        with open(os.path.join(outdir,'test_split_' + str(n) + '.csv'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for i in test_index:
                writer.writerow([final_img_list[i].split(root_path)[1].replace('\\', '/'), final_label_list[i]])
        n += 1
