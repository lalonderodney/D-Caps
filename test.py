'''
Diagnosing Colorectal Polyps in the Wild with Capsule Networks (D-Caps)
Original Paper by Rodney LaLonde, Pujan Kandel, Concetto Spampinato, Michael B. Wallace, and Ulas Bagci
Paper published at ISBI 2020: arXiv version (https://arxiv.org/abs/2001.03305)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for testing models. Please see the README for details about training.
'''

from __future__ import print_function

import warnings
import os
from collections import Counter

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=config))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_curve
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.preprocessing.image import ImageDataGenerator

from model_helper import compile_model
from utils import safe_mkdir

def compute_scores(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        acc = 1. * (tp + tn) / (tp + tn + fp + fn + 1e-7)
        sen = 1. * tp / (tp + fn + 1e-7)
        spec = 1. * tn / (tn + fp + 1e-7)
    except Exception as e:
        print('WARNING: (THIS IS NOT AN ERROR) Encountered issue in computing scores! Skipping and setting to nan. '
              'This is most likely caused by having no images within this group.')
        print(e)
        tn = fp = fn = tp = acc = sen = spec = np.nan
    return np.asarray([acc, sen, spec, tn, fp, fn, tp])

def find_thresh_level(scores, y_true, metric='pseudof1'):
    # ONLY TO BE USED ON VALIDATION SET TO FIND MAX THRESH LEVEL!!!
    _, _, thresholds = precision_recall_curve(y_true=y_true, probas_pred=scores)
    predicted_class_list = np.transpose(np.squeeze(np.where(scores > thresholds, 1, 0)))
    max_pseudof1 = 0; max_sen = 0; max_spec = 0; max_acc = 0; thresh = 0
    for i, y_pred in enumerate(tqdm(predicted_class_list, desc='Finding threshold level.')):
        acc, sen, spec, _, _, _, _ = compute_scores(y_true=y_true, y_pred=y_pred)
        pseudof1 = 2 * spec * sen / (spec + sen)
        if metric == 'pseudof1' and pseudof1 > max_pseudof1:
            max_pseudof1 = pseudof1
            max_sen = sen
            max_spec = spec
            max_acc = acc
            thresh = thresholds[i]
        elif metric == 'acc' and acc > max_acc:
            max_pseudof1 = pseudof1
            max_sen = sen
            max_spec = spec
            max_acc = acc
            thresh = thresholds[i]

    return thresh, [max_acc, max_sen, max_spec]

def test(args, u_model, val_samples, val_shape, test_samples, test_shape):
    out_dir = os.path.join(args.root_dir, 'results', args.exp_name, args.net)
    try:
        safe_mkdir(out_dir)
    except:
        pass

    # Compile the loaded model
    model = compile_model(args=args, uncomp_model=u_model)

    # Load testing weights
    if args.test_weights_path != '':
        try:
            model.load_weights(args.test_weights_path)
            output_filename = os.path.join(out_dir, os.path.basename(args.test_weights_path)[:-5] + '.csv')
        except Exception as e:
            print(e)
            raise NotImplementedError('Failed to load weights file in test.py')
    else:
        try:
            model.load_weights(os.path.join(args.check_dir, args.output_name + '_model_' + args.time + '.hdf5'))
            output_filename = os.path.join(out_dir, args.output_name + '_model_' + args.time + '.csv')
        except Exception as e:
            print(e)
            raise NotImplementedError('Failed to load weights from training.')

    test_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rescale=1./255)

    # VALIDATION SECTION
    if args.thresh_level == 0.:
        # We use this section to choose a threshold which maximizes the harmonic mean between sensitivity and specificity.
        y_true_val = []
        def val_data_gen(gen):
            while True:
                batch = gen.next()
                y_true_val.append(batch[1][0])
                x_batch = np.copy(batch[0])
                for i, x in enumerate(batch[0]):
                    x2 = np.copy(x)
                    x2 = x2 + abs(np.min(x2))
                    x2 /= (np.max(x2) + 1e-7)
                    x2 = (x2 - 0.5) * 2.
                    x_batch[i,...] = x2
                yield x_batch

        val_flow_gen = test_datagen.flow_from_directory(os.path.join(args.img_dir, 'val'),
                                                         target_size=val_shape,
                                                         class_mode='binary',
                                                         batch_size=1,
                                                         seed=12,
                                                         shuffle=False)

        val_flow_gen.reset()
        val_gen = val_data_gen(val_flow_gen)
        val_results = model.predict_generator(val_gen, max_queue_size=1, workers=1, use_multiprocessing=False,
                                              steps=val_samples, verbose=args.verbose)
        if args.net.find('caps') != -1:
            val_scores = val_results[0]
            val_reconstructions = val_results[1]
        else:
            val_scores = val_results
        val_y_true = np.asarray(y_true_val[:-(len(y_true_val) - len(val_flow_gen.filenames))])
        thresh, [val_acc, val_sen, val_spec] = find_thresh_level(val_scores, val_y_true, 'pseudof1')
    else:
        thresh = args.thresh_level

    # TESTING SECTION
    y_true_test = []
    def test_data_gen(gen):
        while True:
            batch = gen.next()
            y_true_test.append(batch[1][0])
            x_batch = np.copy(batch[0])
            for i, x in enumerate(batch[0]):
                x2 = np.copy(x)
                x2 = x2 + abs(np.min(x2))
                x2 /= (np.max(x2) + 1e-7)
                x2 = (x2 - 0.5) * 2.
                x_batch[i, ...] = x2
            yield x_batch

    test_flow_gen = test_datagen.flow_from_directory(os.path.join(args.img_dir, 'test'),
                                                     target_size=test_shape,
                                                     class_mode='binary',
                                                     batch_size=1,
                                                     seed=12,
                                                     shuffle=False)

    filenames = np.asarray(test_flow_gen.filenames)
    test_flow_gen.reset()
    test_gen = test_data_gen(test_flow_gen)
    results = model.predict_generator(test_gen, max_queue_size=1, workers=1, use_multiprocessing=False,
                                      steps=test_samples, verbose=args.verbose)
    if args.net.find('caps') != -1:
        test_scores = results[0]
        reconstructions = results[1]
    else:
        test_scores = results
    test_y_true = np.asarray(y_true_test[:-(len(y_true_test) - len(test_flow_gen.filenames))])

    y_true_check= []
    polyp_ids = []
    for f in tqdm(filenames, desc='Loading filenames'):
        y_true_check.append(f[0])
        temp = os.path.basename(f).split('_')
        try:
            polyp_ids.append('m_{}_{}'.format(os.path.dirname(f)[2:], temp[1][:6]))
        except:
            polyp_ids.append('m_{}_{}'.format(os.path.dirname(f)[2:], temp[0][:6]))

    y_true_check = np.asarray(y_true_check,dtype=np.float32)
    assert np.array_equal(test_y_true, y_true_check), 'Error: Order of images and labels not preserved! ' \
                                                      'Cannot match images to labels.'

    unique_polyp_results_ALL = []; unique_polyp_results_NBI = []; unique_polyp_results_NBIF = []
    unique_polyp_results_NBIN = []; unique_polyp_results_WL = []; unique_polyp_results_WLF = []
    unique_polyp_results_WLN = []; unique_polyp_results_NEAR = []; unique_polyp_results_FAR = []
    unique_polyp_labels = []; unique_polyp_names = []
    counts = Counter(polyp_ids)
    for s, num in tqdm(counts.items(), desc='Computing Scores'):
        current_polyp_results_ALL = []; current_polyp_results_NBI = []; current_polyp_results_NBIF = []
        current_polyp_results_NBIN = []; current_polyp_results_WL = []; current_polyp_results_WLF = []
        current_polyp_results_WLN = []; current_polyp_results_NEAR = []; current_polyp_results_FAR = []
        current_polyp_name = s
        current_polyp_label = test_y_true[polyp_ids.index(s)]
        for _ in range(1, num + 1):  # loop over all images of same polyp
            pos = polyp_ids.index(s)
            current_image_score = test_scores[pos][0]
            current_polyp_results_ALL.append(current_image_score)
            current_filename = os.path.basename(filenames[pos])
            split_name = current_filename.split('-')
            if len(split_name) < 4:
                print('Encountered improperly named image. Please fix: {}.'.format(current_filename))
                continue
            if split_name[3] == 'NBI':
                current_polyp_results_NBI.append(current_image_score)
            elif split_name[3] == 'NBIF':
                current_polyp_results_NBIF.append(current_image_score)
                current_polyp_results_NBI.append(current_image_score)
                current_polyp_results_FAR.append(current_image_score)
            elif split_name[3] == 'NBIN':
                current_polyp_results_NBIN.append(current_image_score)
                current_polyp_results_NBI.append(current_image_score)
                current_polyp_results_NEAR.append(current_image_score)
            elif split_name[3] == 'WL':
                current_polyp_results_WL.append(current_image_score)
            elif split_name[3] == 'WLF':
                current_polyp_results_WLF.append(current_image_score)
                current_polyp_results_WL.append(current_image_score)
                current_polyp_results_FAR.append(current_image_score)
            elif split_name[3] == 'WLN':
                current_polyp_results_WLN.append(current_image_score)
                current_polyp_results_WL.append(current_image_score)
                current_polyp_results_NEAR.append(current_image_score)
            else:
                Warning('Encountered unexpected imaging type: {}.'.format(split_name[3]))
            polyp_ids[pos] = s + '_c'  # mark the image as seen

        unique_polyp_names.append(current_polyp_name)
        unique_polyp_labels.append(current_polyp_label)
        unique_polyp_results_ALL.append(np.mean(np.asarray(current_polyp_results_ALL)))

        if current_polyp_results_NBI:
            unique_polyp_results_NBI.append(np.mean(np.asarray(current_polyp_results_NBI)))
        else:
            unique_polyp_results_NBI.append(np.nan)

        if current_polyp_results_NBIF:
            unique_polyp_results_NBIF.append(np.mean(np.asarray(current_polyp_results_NBIF)))
        else:
            unique_polyp_results_NBIF.append(np.nan)

        if current_polyp_results_NBIN:
            unique_polyp_results_NBIN.append(np.mean(np.asarray(current_polyp_results_NBIN)))
        else:
            unique_polyp_results_NBIN.append(np.nan)

        if current_polyp_results_WL:
            unique_polyp_results_WL.append(np.mean(np.asarray(current_polyp_results_WL)))
        else:
            unique_polyp_results_WL.append(np.nan)

        if current_polyp_results_WLF:
            unique_polyp_results_WLF.append(np.mean(np.asarray(current_polyp_results_WLF)))
        else:
            unique_polyp_results_WLF.append(np.nan)

        if current_polyp_results_WLN:
            unique_polyp_results_WLN.append(np.mean(np.asarray(current_polyp_results_WLN)))
        else:
            unique_polyp_results_WLN.append(np.nan)

        if current_polyp_results_NEAR:
            unique_polyp_results_NEAR.append(np.mean(np.asarray(current_polyp_results_NEAR)))
        else:
            unique_polyp_results_NEAR.append(np.nan)

        if current_polyp_results_FAR:
            unique_polyp_results_FAR.append(np.mean(np.asarray(current_polyp_results_FAR)))
        else:
            unique_polyp_results_FAR.append(np.nan)

    unique_polyp_labels = np.asarray(unique_polyp_labels)
    warnings.filterwarnings("ignore")
    predictions_IMAGES = np.where(test_scores > thresh, 1., 0.)
    predictions_ALL = np.where(unique_polyp_results_ALL > thresh, 1., 0.)
    try:
        predictions_ALL[np.argwhere(np.isnan(unique_polyp_results_ALL))] = np.nan
    except:
        predictions_ALL = np.asarray(unique_polyp_results_ALL)
    predictions_NBI = np.where(unique_polyp_results_NBI > thresh, 1., 0.)
    try:
        predictions_NBI[np.argwhere(np.isnan(unique_polyp_results_NBI))] = np.nan
    except:
        predictions_NBI = np.asarray(unique_polyp_results_NBI)
    predictions_NBIF = np.where(unique_polyp_results_NBIF > thresh, 1., 0.)
    try:
        predictions_NBIF[np.argwhere(np.isnan(unique_polyp_results_NBIF))] = np.nan
    except:
        predictions_NBIF = np.asarray(unique_polyp_results_NBIF)
    predictions_NBIN = np.where(unique_polyp_results_NBIN > thresh, 1., 0.)
    try:
        predictions_NBIN[np.argwhere(np.isnan(unique_polyp_results_NBIN))] = np.nan
    except:
        predictions_NBIN = np.asarray(unique_polyp_results_NBIN)
    predictions_WL = np.where(unique_polyp_results_WL > thresh, 1., 0.)
    try:
        predictions_WL[np.argwhere(np.isnan(unique_polyp_results_WL))] = np.nan
    except:
        predictions_WL = np.asarray(unique_polyp_results_WL)
    predictions_WLF = np.where(unique_polyp_results_WLF > thresh, 1., 0.)
    try:
        predictions_WLF[np.argwhere(np.isnan(unique_polyp_results_WLF))] = np.nan
    except:
        predictions_WLF = np.asarray(unique_polyp_results_WLF)
    predictions_WLN = np.where(unique_polyp_results_WLN > thresh, 1., 0.)
    try:
        predictions_WLN[np.argwhere(np.isnan(unique_polyp_results_WLN))] = np.nan
    except:
        predictions_WLN = np.asarray(unique_polyp_results_WLN)
    predictions_NEAR = np.where(unique_polyp_results_NEAR > thresh, 1., 0.)
    try:
        predictions_NEAR[np.argwhere(np.isnan(unique_polyp_results_NEAR))] = np.nan
    except:
        predictions_NEAR = np.asarray(unique_polyp_results_NEAR)
    predictions_FAR = np.where(unique_polyp_results_FAR > thresh, 1., 0.)
    try:
        predictions_FAR[np.argwhere(np.isnan(unique_polyp_results_FAR))] = np.nan
    except:
        predictions_FAR = np.asarray(unique_polyp_results_FAR)
    warnings.resetwarnings()

    scores_IMAGEWISE = compute_scores(y_true=test_y_true, y_pred=predictions_IMAGES)
    scores_ALL = compute_scores(y_true=np.squeeze(unique_polyp_labels[np.argwhere(np.isfinite(unique_polyp_results_ALL))]),
                                y_pred=np.squeeze(predictions_ALL[np.argwhere(np.isfinite(unique_polyp_results_ALL))]))
    scores_NBI = compute_scores(y_true=np.squeeze(unique_polyp_labels[np.argwhere(np.isfinite(unique_polyp_results_NBI))]),
                                y_pred=np.squeeze(predictions_NBI[np.argwhere(np.isfinite(unique_polyp_results_NBI))]))
    scores_NBIF = compute_scores(y_true=np.squeeze(unique_polyp_labels[np.argwhere(np.isfinite(unique_polyp_results_NBIF))]),
                                y_pred=np.squeeze(predictions_NBIF[np.argwhere(np.isfinite(unique_polyp_results_NBIF))]))
    scores_NBIN = compute_scores(y_true=np.squeeze(unique_polyp_labels[np.argwhere(np.isfinite(unique_polyp_results_NBIN))]),
                                y_pred=np.squeeze(predictions_NBIN[np.argwhere(np.isfinite(unique_polyp_results_NBIN))]))
    scores_WL = compute_scores(y_true=np.squeeze(unique_polyp_labels[np.argwhere(np.isfinite(unique_polyp_results_WL))]),
                                y_pred=np.squeeze(predictions_WL[np.argwhere(np.isfinite(unique_polyp_results_WL))]))
    scores_WLF = compute_scores(y_true=np.squeeze(unique_polyp_labels[np.argwhere(np.isfinite(unique_polyp_results_WLF))]),
                                y_pred=np.squeeze(predictions_WLF[np.argwhere(np.isfinite(unique_polyp_results_WLF))]))
    scores_WLN = compute_scores(y_true=np.squeeze(unique_polyp_labels[np.argwhere(np.isfinite(unique_polyp_results_WLN))]),
                                y_pred=np.squeeze(predictions_WLN[np.argwhere(np.isfinite(unique_polyp_results_WLN))]))
    scores_NEAR = compute_scores(y_true=np.squeeze(unique_polyp_labels[np.argwhere(np.isfinite(unique_polyp_results_NEAR))]),
                                y_pred=np.squeeze(predictions_NEAR[np.argwhere(np.isfinite(unique_polyp_results_NEAR))]))
    scores_FAR = compute_scores(y_true=np.squeeze(unique_polyp_labels[np.argwhere(np.isfinite(unique_polyp_results_FAR))]),
                                y_pred=np.squeeze(predictions_FAR[np.argwhere(np.isfinite(unique_polyp_results_FAR))]))

    np.savetxt(output_filename, np.stack([scores_IMAGEWISE, scores_ALL, scores_NBI,
                                          scores_NBIF, scores_NBIN, scores_WL, scores_WLF, scores_WLN, scores_NEAR,
                                          scores_FAR], axis=0),
               delimiter=',')
    print('- Testing Complete! Results on ALL Polyps -\nAccuracy: {}\nSensitivity: {}\nSpecificity: {}'.format(scores_ALL[0], scores_ALL[1], scores_ALL[2]))
