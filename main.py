'''
Diagnosing Colorectal Polyps in the Wild with Capsule Networks (D-Caps)
Original Paper by Rodney LaLonde, Pujan Kandel, Concetto Spampinato, Michael B. Wallace, and Ulas Bagci
Paper published at ISBI 2020: arXiv version (https://arxiv.org/abs/2001.03305)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is the main file for the project. From here you can train, test, and manipulate the D-Caps of models.
Please see the README for detailed instructions for this project.
'''

from __future__ import print_function

import os
import argparse
import csv
import cv2
from time import gmtime, strftime
from glob import glob
time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

from load_polyp_data import load_data, split_data_for_flow
from model_helper import create_model
from utils import safe_mkdir

def main(args):
    # Set the dictionary of possible experiments
    if args.experiment == 0:
        args.exp_name = 'HPvsA'
    elif args.experiment == 1:
        args.exp_name = 'HPvsA_SSA'
    elif args.experiment == 2:
        args.exp_name = 'HPvsSSA'
    else:
        raise Exception('Experiment number undefined.')

    # Directory to save images for using flow_from_directory
    args.output_name = 'split-' + str(args.split_num) + '_batch-' + str(args.batch_size) + \
                       '_shuff-' + str(args.shuffle_data) + '_aug-' + str(args.aug_data) + \
                       '_lr-' + str(args.initial_lr) + '_recon-' + str(args.recon_wei) + \
                       '_cpre-' + str(args.use_custom_pretrained) + \
                       '_r1-' + str(args.routings1) + '_r2-' + str(args.routings2)
    args.time = time

    args.img_dir = os.path.join(args.root_dir, 'experiment_splits', args.exp_name, 'split_{}'.format(args.split_num))
    safe_mkdir(args.img_dir)

    # Create all the output directories
    args.check_dir = os.path.join(args.root_dir, 'saved_models', args.exp_name, args.net)
    safe_mkdir(args.check_dir)

    args.log_dir = os.path.join(args.root_dir, 'logs', args.exp_name, args.net)
    safe_mkdir(args.log_dir)

    args.img_aug_dir = os.path.join(args.root_dir, 'logs', args.exp_name, args.net, 'aug_imgs')
    safe_mkdir(args.img_aug_dir)

    args.tf_log_dir = os.path.join(args.log_dir, 'tf_logs', args.time)
    safe_mkdir(args.tf_log_dir)

    args.output_dir = os.path.join(args.root_dir, 'plots', args.exp_name, args.net)
    safe_mkdir(args.output_dir)

    # Set net input to (None, None, 3) to allow for variable size color inputs
    net_input_shape = [None, None, 3]
    args.crop_shape = [args.crop_hei, args.crop_wid]
    args.resize_shape = [args.resize_hei, args.resize_wid]

    if args.create_images or args.only_create_images:
        # Load the training, validation, and testing data
        train_list, val_list, test_list = load_data(root=args.root_dir, exp_name=args.exp_name, exp=args.experiment,
                                                    split=args.split_num, k_folds=args.k_fold_splits,
                                                    val_split=args.val_split)
        print('Found {} patients for training, {} for validation, and {} for testing. Note: For patients with more '
              'than one polyp of the same type, all images for the type are placed into either the training or testing '
              'set together.'.format(len(train_list), len(val_list), len(test_list)))

        # Split data for flow_from_directory
        train_samples, train_shape, val_samples, val_shape, test_samples, test_shape = \
            split_data_for_flow(root=args.root_dir, out_dir=args.img_dir, exp_name=args.exp_name,
                                resize_option=args.form_batches, resize_shape=args.resize_shape,
                                train_list=train_list, val_list=val_list, test_list=test_list)
    else:
        train_imgs = glob(os.path.join(args.img_dir, 'train', '*', '*.jpg'))
        assert train_imgs, 'No images found. Please set --create_images to 1 to check your --data_root_path.'
        train_shape = list(cv2.imread(train_imgs[0]).shape[:2])
        train_samples = len(train_imgs)
        val_samples = len(glob(os.path.join(args.img_dir, 'val', '*', '*.jpg')))
        test_samples = len(glob(os.path.join(args.img_dir, 'test', '*', '*.jpg')))

    if args.only_create_images:
        print('Finished creating images, exiting.')
        exit(0)

    if args.resize_shape[0] is not None:
        train_shape[0] = args.resize_shape[0]
    if args.resize_shape[1] is not None:
        train_shape[1] = args.resize_shape[1]

    train_shape = val_shape = test_shape = (
    train_shape[0] // (2 ** 6) * (2 ** 6), train_shape[1] // (2 ** 6) * (2 ** 6))  # Assume 6 downsamples
    net_input_shape = (train_shape[0], train_shape[1], net_input_shape[2])
    model_list = create_model(args=args, input_shape=net_input_shape)
    model_list[0].summary()

    # Run the chosen functions
    if args.train:
        from train import train
        # Run training
        train(args=args, u_model=model_list[0], train_samples=train_samples, val_samples=val_samples,
              train_shape=train_shape, val_shape=val_shape)

    if args.test:
        from test import test
        # Run testing
        test_model = (model_list[1] if args.net.find('caps') != -1 else model_list[0])
        test(args=args, u_model=test_model, val_samples=val_samples, val_shape=val_shape,
             test_samples=test_samples, test_shape=test_shape)

    if args.manip and args.net.find('caps') != -1:
        from manip import manip
        # Run manipulation of d-caps
        manip(args, test_list, model_list[2])

    if args.pred:
        try:
            with open(os.path.join(args.root_dir, 'split_lists', 'pred_split_' + str(args.split_num) + '.csv'), 'r') as f:
                reader = csv.reader(f)
                pred_list = list(reader)
            from predict import predict
            predict(args, pred_list, model_list, net_input_shape)
        except Exception as e:
            print(e)
            print('Unable to load prediction list inside main.py, skipping Predict.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colorectal Polyp Diagnosis')
    parser.add_argument('--data_root_dir', type=str, required=True,
                        help='The root directory where your datasets are stored.')
    parser.add_argument('--dataset', type=str, default='Mayo',
                        help='The root directory for your data.')
    parser.add_argument('--experiment', type=int, default=0, choices=[0,1,2],
                        help='0: HP vs Adenoma, 1: HP vs SSA & Adenoma, 2: HP vs SSA '
                             'Change this in main.py and load_poly_data if you want to run different experiments.')
    parser.add_argument('--create_images', type=int, default=1, choices=[0,1],
                        help='Set to 1 to make images.')
    parser.add_argument('--only_create_images', type=int, default=0, choices=[0,1],
                        help='Quit after making images.')

    parser.add_argument('--net', type=str.lower, default='dcaps',
                        choices=['dcaps', 'inceptionv3'],
                        help='Choose your network.')
    parser.add_argument('--test_weights_path', type=str, default='',
                        help='/path/to/trained_model.hdf5 from root. Set to "" for none.')
    parser.add_argument('--use_custom_pretrained', type=int, default=0, choices=[0,1],
                        help='Set to 1 to enable using pre-trained weights set in --weights_path arg.')

    parser.add_argument('--k_fold_splits', type=int, default=10,
                        help='Number of training splits to create for k-fold cross-validation.')
    parser.add_argument('--split_num', type=int, default=0,
                        help='Which training split to train/test on.')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Percentage between 0 and 1 of training split to use as validation.')

    parser.add_argument('--train', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable training.')
    parser.add_argument('--test', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable testing.')
    parser.add_argument('--pred', type=int, default=0, choices=[0,1],
                        help='Set to 1 to enable prediction.')
    parser.add_argument('--manip', type=int, default=1, choices=[0,1],
                        help='Set to 1 to enable manipulation.')

    parser.add_argument('--shuffle_data', type=int, default=1, choices=[0,1],
                        help='Whether or not to shuffle the training data (both per epoch and in slice order.')
    parser.add_argument('--aug_data', type=int, default=1, choices=[0,1],
                        help='Whether or not to use data augmentation during training.')

    parser.add_argument('--form_batches', type=str.lower, default='resize_avg',
                        choices=['resize_max', 'resize_min', 'resize_std', 'resize_avg', 'crop_min'],
                        help='To form batches for mini-batch training, all samples in a batch must be the same size. '
                             'When differences occur choose one of these options... '
                             '    resize_max: resize all images to the largest width and height values.'
                             '    resize_min: resize all images to the smallest width and height values. '
                             '    resize_std: resize all images to a standard size, specify --resize_hei --resize_wid.'
                             '    resize_avg: resize all images to the average width and height values.'
                             '    crop_min: crop images using random crop function to smallest height and width values.')
    parser.add_argument('--crop_hei', type=int, default=None,
                        help="Random image crop height for training")
    parser.add_argument('--crop_wid', type=int, default=None,
                        help="Random image crop width for training")
    parser.add_argument('--resize_hei', type=int, default=256,
                        help="Image resize height for forming equal size batches")
    parser.add_argument('--resize_wid', type=int, default=320,
                        help="Image resize width for forming equal size batches")

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training/testing.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train for.')
    parser.add_argument('--initial_lr', type=float, default=0.001,
                        help='Initial learning rate for Adam.')
    parser.add_argument('--recon_wei', type=float, default=0.0005,
                        help="If using dcaps: The coefficient (weighting) for the loss of decoder")
    parser.add_argument('--k_size', type=int, default=5,
                        help='Kernel size for dcaps.')
    parser.add_argument('--output_atoms', type=int, default=16,
                        help='Number of output atoms for dcaps.')
    parser.add_argument('--routings1', type=int, default=3,
                        help="If using dcaps: The number of iterations used in routing algorithm for layers which "
                             "maintain spatial resolution. should > 0")
    parser.add_argument('--routings2', type=int, default=3,
                        help="If using dcaps: The number of iterations used in routing algorithm for layers which "
                             "change spatial resolution. should > 0")

    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Set the verbose value for training. 0: Silent, 1: per iteration, 2: per epoch.')

    parser.add_argument('--save_prefix', type=str, default='',
                        help='Prefix to append to saved CSV.')

    parser.add_argument('--thresh_level', type=float, default=0.0,
                        help='Enter 0.0 for dynamic thresholding, else set value')
    parser.add_argument('--compute_dice', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--compute_jaccard', type=int, default=1,
                        help='0 or 1')
    parser.add_argument('--compute_assd', type=int, default=0,
                        help='0 or 1')

    parser.add_argument('--which_gpus', type=str, default="0",
                        help='Enter "-2" for CPU only, else input the GPU_ID e.g. 0 or 1 or 2... '
                             'Currently only single GPU training is supported.')

    arguments = parser.parse_args()

    # Ensure training, testing, and manip are not all turned off
    assert (arguments.train or arguments.test or arguments.manip or arguments.pred), \
        'Cannot have train, test, pred, and manip all set to 0, Nothing to do.'

    assert not (arguments.use_custom_pretrained and arguments.use_default_pretrained), \
        'Cannot use custom pretrained weights and the default pretrained weights at the same time.'

    # Set root to the dataset chosen (need trailing slash for annoying os.path.join issue)
    arguments.root_dir = os.path.join(arguments.data_root_dir, arguments.dataset)
    if arguments.root_dir[-1] != '/':
        arguments.root_dir += '/'

    # Mask the GPUs for TensorFlow
    if arguments.which_gpus == -2:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        try:
            gpu = int(arguments.which_gpus)
        except:
            raise NotImplementedError('Invalid GPU id given! Must be an interger >= 0.')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    main(arguments)
