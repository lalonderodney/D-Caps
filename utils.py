import os
import errno

import tensorflow as tf
from keras import backend as K

def safe_mkdir(dir_to_make: str) -> None:
    '''
    Attempts to make a directory following the Pythonic EAFP strategy which prevents race conditions.

    :param dir_to_make: The directory path to attempt to make.
    :return: None
    '''
    try:
        os.makedirs(dir_to_make)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('ERROR: Unable to create directory: {}'.format(dir_to_make), e)
            raise

def as_keras_metric(method):
    import functools
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper