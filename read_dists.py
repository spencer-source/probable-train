
import math
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def read_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "ip": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "lab": tf.io.FixedLenFeature([], tf.int64)  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    dist = tf.io.parse_tensor(example['ip'], out_type=tf.float32)
    lbl = example['lab']
    
    return dist, lbl 

def load_dataset(filenames, ds_size, shuffle=True, reshuffle=True, shuffle_size=10_000):
    cycle_len = len(filenames)
    
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.shuffle(tf.cast(tf.shape(filenames)[0], tf.int64), reshuffle_each_iteration=reshuffle)

    dataset = dataset.interleave(
      lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP').map(read_tfrecord), 
      cycle_length=cycle_len, block_length=100)
    
    if shuffle:
        dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=reshuffle)

    return dataset

FILENAME_PATTERN = [f"./tfrecords/train_{i}.tfrecords" for i in [30, 50, 100]]
filenames = tf.io.gfile.glob(FILENAME_PATTERN)
cyc_len = len(filenames)

ds_size = 8_000 * 9 * cyc_len
shuffle_buffer = 8_000//100 * 9 * cyc_len

train_ds = load_dataset(filenames=filenames, ds_size=ds_size, shuffle_size = shuffle_buffer)

FILENAME_PATTERN = [f"./tfrecords/validate_{i}.tfrecords" for i in [30, 50, 100]]
filenames = tf.io.gfile.glob(FILENAME_PATTERN)
cyc_len = len(filenames)

ds_size = 1_000 * 9 * cyc_len
shuffle_buffer = 1_000//100 * 9 * cyc_len

val_ds = load_dataset(filenames=filenames, ds_size=ds_size, reshuffle=False, shuffle_size = shuffle_buffer)

FILENAME_PATTERN = [f"./tfrecords/test_{i}.tfrecords" for i in [30, 50, 100]]
filenames = tf.io.gfile.glob(FILENAME_PATTERN)
cyc_len = len(filenames)

ds_size = 1_000 * 9 * cyc_len
shuffle_buffer = 1_000//100 * 9 * cyc_len

test_ds = load_dataset(filenames=filenames, ds_size=ds_size, reshuffle=False, shuffle_size = shuffle_buffer)



