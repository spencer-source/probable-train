
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from itertools import cycle, repeat

tfd = tfp.distributions

distributions = [
    tfd.Uniform(low=0.0, high=1.0),
    tfd.Normal(loc=0.0, scale=1.0),
    tfd.Logistic(loc=0.0, scale=1.0),
    tfd.Exponential(rate=1.0),
    tfd.Laplace(loc=0.0, scale=1.0),
    tfd.HalfNormal(scale=1.0),
    tfd.LogLogistic(loc=0.0, scale=1.0),
    tfd.LogNormal(loc=0.0, scale=1.0),
    tfd.Gumbel(loc=0.0, scale=1.0)    
]

labels = {}
for i in range(len(distributions)):
    labels[distributions[i].name]=i

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value is tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])
  )
  
def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])
  )

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def parse_single_dist(dist, lab):
  
  #define the dictionary -- the structure -- of our single distribution example
  data = {
        'ip' : _bytes_feature(serialize_array(dist)),
        'lab' : _int64_feature(lab)
    }
    
  out = tf.train.Example(features=tf.train.Features(feature=data))

  return out.SerializeToString()


def write_dists_to_tfr(dists, labels, sample_size, sample_count:int=10_000, filename:str="tfrecords/"):
  
  filename= filename+f"{sample_size}.tfrecords"
  options = tf.io.TFRecordOptions(compression_type='GZIP')
  writer = tf.io.TFRecordWriter(filename, options=options) 
  initial_len = len(dists)
  sample_shard = sample_count//100
  
  total = 0
  cycle_dists = cycle(dists)
  for dist in cycle_dists:
    tf.random.set_seed(seed()) # from seed stream
    cur_dist = dist.sample(sample_shape=(sample_shard, sample_size))
    cur_dist_split = tf.unstack(cur_dist, sample_shard, axis=0)
    cur_lbl = labels[dist.name]
    
    count = 0
    for nn in cur_dist_split:
      out = parse_single_dist(dist=nn, lab=cur_lbl)
      writer.write(out)
      count += 1
    total += count
    if total >= initial_len*sample_count:
      break
  
  writer.close()
  
  return total


sample_cnt = 8_000
seed = tfp.util.SeedStream(123, salt='stochastic')
for nd in [30, 50, 100, 250, 500, 750, 1_000, 10_000]:
  total = write_dists_to_tfr(distributions, labels, sample_size=nd, sample_count=sample_cnt, filename="tfrecords/train_")
  print(total)

for nd in [30, 50, 100, 250, 500, 750, 1_000, 10_000]:
  total = write_dists_to_tfr(distributions, labels, sample_size=nd, sample_count=1_000, filename="tfrecords/validate_")
  print(total)  

test_distributions = [
    tfd.Uniform(low=0.0, high=2.0),
    tfd.Normal(loc=0.0, scale=2.0),
    tfd.Logistic(loc=0.0, scale=2.0),
    tfd.Exponential(rate=2.0),
    tfd.Laplace(loc=0.0, scale=2.0),
    tfd.HalfNormal(scale=2.0),
    tfd.LogLogistic(loc=0.0, scale=2.0),
    tfd.LogNormal(loc=0.0, scale=2.0),
    tfd.Gumbel(loc=0.0, scale=2.0)    
]

test_labels = {}
for i in range(len(test_distributions)):
    test_labels[test_distributions[i].name]=i


for nd in [30, 50, 100, 250, 500, 750, 1_000, 10_000]:
  total = write_dists_to_tfr(test_distributions, test_labels, sample_size=nd, sample_count=1_000, filename="tfrecords/test_")
  print(total)  



