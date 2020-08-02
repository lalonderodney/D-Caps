'''
Diagnosing Colorectal Polyps in the Wild with Capsule Networks (D-Caps)
Original Paper by Rodney LaLonde, Pujan Kandel, Concetto Spampinato, Michael B. Wallace, and Ulas Bagci
Paper published at ISBI 2020: arXiv version (https://arxiv.org/abs/2001.03305)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for training models. Please see the README for details about training.
'''

from __future__ import print_function

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard

from model_helper import compile_model

# debug is for visualizing the created images
debug = False

def get_callbacks(arguments):
    monitor_name = 'val_loss'

    csv_logger = CSVLogger(os.path.join(arguments.log_dir, arguments.output_name + '_log_' + arguments.time + '.csv'), separator=',')
    tb = TensorBoard(arguments.tf_log_dir, batch_size=arguments.batch_size, histogram_freq=0)
    model_checkpoint = ModelCheckpoint(os.path.join(arguments.check_dir, arguments.output_name + '_model_' + arguments.time + '.hdf5'),
                                       monitor=monitor_name, save_best_only=True, save_weights_only=True,
                                       verbose=1, mode='min')
    lr_reducer = ReduceLROnPlateau(monitor=monitor_name, factor=0.05, cooldown=0, patience=10,verbose=1, mode='min')
    early_stopper = EarlyStopping(monitor=monitor_name, min_delta=0, patience=35, verbose=0, mode='min')

    return [model_checkpoint, csv_logger, lr_reducer, early_stopper, tb]


def plot_training(training_history, arguments):
    if arguments.net.find('caps') != -1:
        caps = 'out_caps_'
    else:
        caps = ''

    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    f.suptitle(arguments.net, fontsize=18)

    ax1.plot(training_history.history[caps+'precision'])
    ax1.plot(training_history.history[caps+'recall'])
    ax1.plot(training_history.history[caps+'binary_accuracy'])
    ax1.plot(training_history.history['val_'+caps+'precision'])
    ax1.plot(training_history.history['val_'+caps+'recall'])
    ax1.plot(training_history.history['val_'+caps+'binary_accuracy'])

    ax1.set_title('Precision, Recall, and Accuracy')
    ax1.legend(['Train_Precision', 'Train_Recall', 'Train_Accuracy', 'Val_Precision', 'Val_Recall', 'Val_Accuracy'],
               loc='lower right')
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    ax1.set_xticks(np.arange(0, len(training_history.history[caps+'precision'])))
    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')

    ax2.plot(training_history.history[caps+'loss'])
    ax2.plot(training_history.history['val_'+caps+'loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train', 'Val'], loc='upper right')
    ax1.set_xticks(np.arange(0, len(training_history.history[caps+'loss'])))
    ax2.grid(True)
    gridlines2 = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines2:
        line.set_linestyle('-.')

    f.savefig(os.path.join(arguments.output_dir, arguments.output_name + '_plots_' + arguments.time + '.png'))
    plt.close()

def train(args, u_model, train_samples, val_samples, train_shape, val_shape):
    # Compile the loaded model
    model = compile_model(args=args, uncomp_model=u_model)

    # Load pre-trained weights
    if args.use_custom_pretrained:
        try:
            model.load_weights(args.weights_path)
        except Exception as e:
            print(e)
            print('!!! Failed to load custom weights file. Training without pre-trained weights. !!!')

    # Set the callbacks
    callbacks = get_callbacks(args)

    if args.aug_data:
        train_datagen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=45,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1./255)

        val_datagen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rescale=1./255)
    else:
        train_datagen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rotation_range=0,
            width_shift_range=0.,
            height_shift_range=0.,
            shear_range=0.,
            zoom_range=0.,
            fill_mode='nearest',
            horizontal_flip=False,
            vertical_flip=False,
            rescale=1./255)

        val_datagen = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
            rescale=1./255)

    if debug:
        save_dir = args.img_aug_dir
    else:
        save_dir = None

    def caps_data_gen(gen):
        while True:
            batch = gen.next()
            x_batch = np.copy(batch[0])
            for i, x in enumerate(batch[0]):
                x2 = np.copy(x)
                x2 = x2 + abs(np.min(x2))
                x2 /= (np.max(x2) + 1e-7)
                x2 = (x2 - 0.5) * 2.
                x_batch[i,...] = x2
            yield [x_batch, batch[1]], [batch[1], x_batch]

    def norm_data_gen(gen):
        while True:
            batch = gen.next()
            x_batch = np.copy(batch[0])
            for i, x in enumerate(batch[0]):
                x2 = np.copy(x)
                x2 = x2 + abs(np.min(x2))
                x2 /= (np.max(x2) + 1e-7)
                x2 = (x2 - 0.5) * 2.
                x_batch[i,...] = x2
            yield [x_batch, batch[1]]

    if args.net.find('caps') != -1:
        train_gen = caps_data_gen(train_datagen.flow_from_directory(os.path.join(args.img_dir, 'train'),
                                                                    target_size=train_shape,
                                                                    class_mode='binary',
                                                                    batch_size=args.batch_size,
                                                                    seed=12,
                                                                    save_to_dir=save_dir))
        val_gen = caps_data_gen(val_datagen.flow_from_directory(os.path.join(args.img_dir, 'val'),
                                                                target_size=val_shape,
                                                                class_mode='binary',
                                                                batch_size=args.batch_size,
                                                                seed=12))
    else:
        train_gen = norm_data_gen(train_datagen.flow_from_directory(os.path.join(args.img_dir, 'train'),
                                                                    target_size=train_shape,
                                                                    class_mode='binary',
                                                                    batch_size=args.batch_size,
                                                                    seed=12,
                                                                    save_to_dir=save_dir))
        val_gen = norm_data_gen(val_datagen.flow_from_directory(os.path.join(args.img_dir, 'val'),
                                                                target_size=val_shape,
                                                                class_mode='binary',
                                                                batch_size=args.batch_size,
                                                                seed=12))
    # Settings
    train_steps = train_samples//args.batch_size
    val_steps = val_samples//args.batch_size
    workers = 8
    multiproc = True
    if args.which_gpus == '-2':
        workers = 1 # If running on CPU we don't need workers due to slower processing.
        multiproc = False
        if train_steps > 200:
            train_steps = 200
            val_steps = 20 # CPU too slow, need to test/save more frequently.

    # Run training
    history = model.fit_generator(train_gen,
                                  max_queue_size=80, workers=workers, use_multiprocessing=multiproc,
                                  steps_per_epoch=train_steps,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  epochs=args.epochs,
                                  class_weight={0:1., 1:1.},
                                  callbacks=callbacks,
                                  verbose=args.verbose,
                                  shuffle=True)

    # Plot the training data collected
    plot_training(history, args)
