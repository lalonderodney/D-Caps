'''
Diagnosing Colorectal Polyps in the Wild with Capsule Networks (D-Caps)
Original Paper by Rodney LaLonde, Pujan Kandel, Concetto Spampinato, Michael B. Wallace, and Ulas Bagci
Paper published at ISBI 2020: arXiv version (https://arxiv.org/abs/2001.03305)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is a helper file for choosing which model to create.
'''

import tensorflow as tf
from keras.metrics import binary_accuracy
from keras.optimizers import Adam

from utils import as_keras_metric

def create_model(args, input_shape):
    if args.net == 'dcaps':
        from diagnosiscapsules import DiagnosisCapsules
        return DiagnosisCapsules(input_shape=input_shape, n_class=2, k_size=args.k_size,
                                 output_atoms=args.output_atoms, routings1=args.routings1, routings2=args.routings2)

    elif args.net == 'inceptionv3':
        from keras.models import Model
        from keras.layers import GlobalAveragePooling2D, Dense
        from keras.applications.inception_v3 import InceptionV3

        # Only train from scratch for fair comparison.
        base_model = InceptionV3(include_top=False, weights=None, input_shape=input_shape)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # and a logistic layer -- let's say we have 2 classes
        predictions = Dense(1, activation='sigmoid', name='out')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        return [model]
    else:
        raise Exception('Unknown network type specified: {}'.format(args.net))


def get_loss(net, recon_wei):
    if net.find('caps') != -1:
        return {'out_caps': 'binary_crossentropy', 'out_recon': 'MAE'}, {'out_caps': 1., 'out_recon': recon_wei}
    else:
        return 'binary_crossentropy', None


def compile_model(args, uncomp_model):
    try:
        opt = Adam(lr=args.initial_lr, beta_1=0.99, beta_2=0.999, decay=1e-6, amsgrad=True)
    except:
        opt = Adam(lr=args.initial_lr, beta_1=0.99, beta_2=0.999, decay=1e-6)

    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)

    metrics = [precision, recall, binary_accuracy]

    if args.net.find('caps') != -1:
        metrics = {'out_caps': metrics}

    loss, loss_weighting = get_loss(net=args.net, recon_wei=args.recon_wei)

    uncomp_model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return uncomp_model
