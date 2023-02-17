import tensorflow as tf

# for variable length blocks
def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 1))
  length = tf.reduce_sum(used, 0)
  length = tf.cast(length, tf.int64)
  return length

def right_padding(x, max_len=128):
    padding = max_len - tf.shape(x)[0]
    x = tf.pad(x, [[0, padding]])
    return tf.expand_dims(x, -1)

def left_padding(x, max_len=128):
    padding = max_len - tf.shape(x)[0]
    x = tf.pad(x, [[padding, 0]])
    return tf.expand_dims(x, -1)

def _padlen_power_2(x):
    length = tf.cast(tf.shape(x)[0], tf.float32)
    padding = 2**tf.math.ceil(tf.math.log(length) / tf.math.log(2.0))
    padding = padding - length
    padding = tf.cast(padding, tf.int32)
    x = tf.pad(x, [[0, padding]])
    return tf.expand_dims(x, -1)

def group_padded(dataset, batch_size=256):
  # and segmenting, or framing, 4 vectors
  padded_ds = dataset.map(lambda x,y: (right_padding(x), y))
  key_func = lambda x,y: tf.cast(tf.shape(x)[0], tf.int64)
  reduce_func = lambda key, dataset: dataset.batch(batch_size)
  
  dataset = padded_ds.group_by_window(key_func, reduce_func, batch_size)

  return dataset.prefetch(tf.data.AUTOTUNE)


def group_unpadded(dataset, batch_size=256):
  
  key_func = lambda x,y: tf.cast(tf.shape(x)[0], tf.int64)
  reduce_func = lambda key, dataset: dataset.batch(batch_size)
  dataset = dataset.group_by_window(key_func, reduce_func, batch_size)
  
  return dataset.prefetch(tf.data.AUTOTUNE)

