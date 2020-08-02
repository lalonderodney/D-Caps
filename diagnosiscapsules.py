'''
Diagnosing Colorectal Polyps in the Wild with Capsule Networks (D-Caps)
Original Paper by Rodney LaLonde, Pujan Kandel, Concetto Spampinato, Michael B. Wallace, and Ulas Bagci
Paper published at ISBI 2020: arXiv version (https://arxiv.org/abs/2001.03305)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This is where the D-Caps model is defined.
'''

from keras import layers, models
from keras import backend as K
from capsule_layers import ConvCapsuleLayer, Mask, Length, ExpandDim, RemoveDim

K.set_image_data_format('channels_last')

def DiagnosisCapsules(input_shape, n_class=2, k_size=5, output_atoms=16, routings1=3, routings2=3):
    """
    A Capsule Network on Medical Image Diagnosis.
    :param input_shape: data shape
    :param n_class: number of classes
    :param k_size: kernel size for convolutional capsules
    :param output_atoms: number of atoms in D-Caps layer
    :param routings1: number of routing iterations when stride is 1
    :param routings2: number of routing iterations when stride is > 1
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    if n_class == 2:
        n_class = 1 # binary output

    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=16, kernel_size=k_size, strides=2, padding='same', activation='relu', name='conv1')(x)

    # Reshape layer to be 1 capsule x [filters] atoms
    conv1_reshaped = ExpandDim(name='expand_dim')(conv1)

    # Layer 1: Primary Capsule: Conv cap with routing 1
    primary_caps = ConvCapsuleLayer(kernel_size=k_size, num_capsule=2, num_atoms=16, strides=2, padding='same',
                                    routings=1, name='primarycaps')(conv1_reshaped)

    # Layer 2: Convolutional Capsule
    conv_cap_2_1 = ConvCapsuleLayer(kernel_size=k_size, num_capsule=4, num_atoms=16, strides=1, padding='same',
                                    routings=routings1, name='conv_cap_2_1')(primary_caps)

    # Layer 2: Convolutional Capsule
    conv_cap_2_2 = ConvCapsuleLayer(kernel_size=k_size, num_capsule=4, num_atoms=32, strides=2, padding='same',
                                    routings=routings2, name='conv_cap_2_2')(conv_cap_2_1)

    # Layer 3: Convolutional Capsule
    conv_cap_3_1 = ConvCapsuleLayer(kernel_size=k_size, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=routings1, name='conv_cap_3_1')(conv_cap_2_2)

    # Layer 3: Convolutional Capsule
    conv_cap_3_2 = ConvCapsuleLayer(kernel_size=k_size, num_capsule=8, num_atoms=64, strides=2, padding='same',
                                    routings=routings2, name='conv_cap_3_2')(conv_cap_3_1)

    # Layer 4: Convolutional Capsule
    conv_cap_4_1 = ConvCapsuleLayer(kernel_size=k_size, num_capsule=8, num_atoms=32, strides=1, padding='same',
                                    routings=routings1, name='conv_cap_4_1')(conv_cap_3_2)

    # Layer 3: Convolutional Capsule
    conv_cap_4_2 = ConvCapsuleLayer(kernel_size=k_size, num_capsule=n_class, num_atoms=output_atoms, strides=2,
                                    padding='same', routings=routings2, name='conv_cap_4_2')(conv_cap_4_1)

    if n_class > 1:
        # Perform GAP on each capsule type.
        class_caps_list = []
        for i in range(n_class):
            in_shape = conv_cap_4_2.get_shape().as_list()
            one_class_capsule = layers.Lambda(lambda x: x[:, :, :, i, :], output_shape=in_shape[1:3]+in_shape[4:])(conv_cap_4_2)
            gap = layers.GlobalAveragePooling2D(name='gap_{}'.format(i))(one_class_capsule)

            # Put capsule dimension back for length and recon
            class_caps_list.append(ExpandDim(name='expand_gap_{}'.format(i))(gap))

        class_caps = layers.Concatenate(axis=-2, name='class_caps')(class_caps_list)
    else:
        # Remove capsule dim, perform GAP, put capsule dim back
        conv_cap_4_2_reshaped = RemoveDim(name='conv_cap_4_2_reshaped')(conv_cap_4_2)
        gap = layers.GlobalAveragePooling2D(name='gap')(conv_cap_4_2_reshaped)
        class_caps = ExpandDim(name='expand_gap')(gap)

    # Output layer which predicts classes
    out_caps = Length(num_classes=n_class, name='out_caps')(class_caps)

    # Decoder network.
    _, C, A = class_caps.get_shape()
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([class_caps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(class_caps)  # Mask using the capsule with maximal length. For prediction

    def shared_reconstructor(mask_layer):
        recon_1 = layers.Dense(input_shape[0]//(2**6) * input_shape[1]//(2**6), kernel_initializer='he_normal', activation='relu', name='recon_1',
                               input_shape=(A.value,))(mask_layer)

        recon_1a = layers.Reshape((input_shape[0]//(2**6), input_shape[1]//(2**6), 1), name='recon_1a')(recon_1)

        recon_2 = layers.Conv2DTranspose(filters=128, kernel_size=5, strides=(8,8), padding='same',
                                         kernel_initializer='he_normal',
                                         activation='relu', name='recon_2')(recon_1a)

        recon_3 = layers.Conv2DTranspose(filters=64, kernel_size=5, strides=(8,8), padding='same',
                                         kernel_initializer='he_normal',
                                         activation='relu', name='recon_3')(recon_2)

        out_recon = layers.Conv2D(filters=3, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                  activation='tanh', name='out_recon')(recon_3)

        return out_recon

    # Models for training and evaluation (prediction)
    train_model = models.Model(inputs=[x, y], outputs=[out_caps, shared_reconstructor(masked_by_y)])
    eval_model = models.Model(inputs=x, outputs=[out_caps, shared_reconstructor(masked)])

    # manipulate model
    noise = layers.Input(shape=((C.value, A.value)))
    noised_class_caps = layers.Add()([class_caps, noise])
    masked_noised_y = Mask()([noised_class_caps, y])
    manipulate_model = models.Model(inputs=[x, y, noise], outputs=shared_reconstructor(masked_noised_y))

    return train_model, eval_model, manipulate_model
