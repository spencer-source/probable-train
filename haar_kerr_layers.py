
import math
import tensorflow as tf
import tensorflow_addons as tfa

from collections.abc import Iterable


def two_chan_4(stage=0):
  U2_L2 = tf.constant([[1., -1, 0, 0], [0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=tf.float32)
  if stage==1:
    U2_L2 = 1/math.sqrt(2) * U2_L2
  return U2_L2

# second stage, three channel encoding matrix (Haar)
def three_chan_4():
  s2 = math.sqrt(2)
  H2 = tf.constant([[s2, 0, 0, 0], [0, s2, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]], dtype=tf.float32)
  H2 = 0.5 * tf.matmul(H2, two_chan_4(stage=0))
  return tf.constant(H2)


class HaarWaveLayer(tf.keras.layers.Layer):
    """Haar Wavelet Transform"""
    def __init__(self, haar_ker):
        self.haar_ker = haar_ker
        super(HaarWaveLayer, self).__init__()
    
    def build(self, input_shape):
        if input_shape[0] is None:
            self.bs = -1
        else:
            self.bs = input_shape[0]
        self.nx = input_shape[1]
        self.cn = input_shape[2]
    
    def compute_output_shape(self, input_shape):
        nx = input_shape[1]
        cn = input_shape[2]
        hn = math.ceil(xn / 4)
        out_shape = tf.TensorShape([input_shape[0], hn, 4*cn])
        return(out_shape)

    def call(self, inputs):
        haar_ker = self.haar_ker
        bs = self.bs
        nx = self.nx
        cn = self.cn
        cl = tf.math.ceil(nx/4)
        cl = tf.cast(cl, tf.int32)
        t1 = tf.transpose(inputs, perm=[0, 2, 1])
        s1 = tf.reshape(t1, [-1, 4*cl, 1])
        f1 = tf.reshape(haar_ker[0], [4, 1, 1])
        f2 = tf.reshape(haar_ker[1], [4, 1, 1])
        f3 = tf.reshape(haar_ker[2], [4, 1, 1])
        f4 = tf.reshape(haar_ker[3], [4, 1, 1])
        res1 = tf.nn.conv1d(s1, f1, stride=4, padding='VALID')
        res2 = tf.nn.conv1d(s1, f2, stride=4, padding='VALID')
        res3 = tf.nn.conv1d(s1, f3, stride=4, padding='VALID')
        res4 = tf.nn.conv1d(s1, f4, stride=4, padding='VALID')
        r = tf.concat([res1, res2, res3, res4], axis=-1)
        r = tf.reshape(r, [bs, cn, cl, 4])
        r = tf.transpose(r, [0, 2, 3, 1])
        r = tf.reshape(r, [bs, cl, 4*cn])
        return r


class ConvLayer(tf.keras.layers.Layer):
    def __init__(
        self, kernel_num=4, kernel_size=1, strides=1, dilation_rate=1, padding='same',
        **kwargs
    ): 
        super().__init__(**kwargs)
        # conv layer
        self.kern_n = int(kernel_num)
        self.kern_s = kernel_size
        self.strides = strides
        self.d_rate = dilation_rate
        self.padding = padding
        self.tlu = tfa.layers.TLU()
    
    def build(self, input_shape):
        ishape = input_shape[1:]
        self.conv = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv1D(
                filters=self.kern_n, kernel_size=self.kern_s, strides=self.strides, padding=self.padding, 
                dilation_rate=self.d_rate, activation=tfa.activations.snake, input_shape=ishape,
            ), 
            data_init=False
        )
            
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.tlu(x)
        return x

class DyadicConv(tf.keras.layers.Layer):
    def __init__(self, lev=4, strides=1, **kwargs):
        super(DyadicConv, self).__init__(**kwargs)
        self.conv1 = ConvLayer(kernel_num=lev, strides=strides)
        self.conv2 = ConvLayer(kernel_num=lev, strides=strides)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x


class UpSample(tf.keras.layers.Layer):
    def __init__(self, p: Iterable[int], lev=4, strides=1, **kwargs):
        super(UpSample, self).__init__(**kwargs)
    
        self.up_1 = DyadicConv(lev * 2**(p[0] + 1), strides=strides) 
        self.up_2 = DyadicConv(lev * 2**(p[1] + 1), strides=strides) 
        
    def call(self, inputs):
        x = self.up_1(inputs)
        x = self.up_2(x)
        return x

class DownSample(tf.keras.layers.Layer):
    def __init__(self, p: Iterable[int], lev=4, strides=2, **kwargs):
        super(DownSample, self).__init__(**kwargs)
        
        self.down_1 = ConvLayer(lev * 2**(p[0] + 1), strides=strides)
        self.down_2 = ConvLayer(lev * 2**(p[1] + 1), strides=strides)
        
    def call(self, inputs):
        x = self.down_1(inputs)
        x = self.down_2(x)
        return x


class WaveletBlock(tf.keras.layers.Layer):
    def __init__(self, lev=4):
        self.haar1 = HaarWaveLayer(three_chan_4())
        self.haar2 = HaarWaveLayer(three_chan_4())
        
        self.gn1 = tfa.layers.GroupNormalization(groups=2)
        self.gn2 = tfa.layers.GroupNormalization(groups=4)
        
        self.dyadic = DyadicConv(lev, 1)
        
        self.down_sample1 = DownSample((0, 1))
        self.down_sample2 = DownSample((2, 3))

        self.up_sample1 = UpSample((0, 1))
        self.up_sample2 = UpSample((2, 3))

        self.cat1 = tf.keras.layers.Concatenate()
        self.cat2 = tf.keras.layers.Concatenate()
        super(WaveletBlock, self).__init__()

    def call(self, inputs):
        h0 = inputs
        x = self.dyadic(inputs)
        
        h1 = self.haar1(h0)
        h1 = self.gn1(h1)
        h2 = self.haar2(h1)
        h2 = self.gn2(h2)
        
        x = self.down_sample1(x)
        x = self.up_sample1(x)
        x = self.cat1([x, h1])
        
        x = self.down_sample2(x)
        x = self.up_sample2(x)
        x = self.cat2([x, h2])
        return x 

x = tf.random.uniform([1, 128, 1])
y = WaveletBlock()(x)
plt.clf()
plt.plot(x[0])
plt.show()

x = tf.random.normal([1, 128, 1])
def make_model(input_shape):
    ip = tf.keras.Input([128, 1])
    x = WaveletBlock(4)(ip)
    x = tfa.layers.SpectralNormalization(
        tf.keras.layers.Conv1D(512, 1, activation="swish"), 2
    )(x)
    x = tfa.layers.TLU()(x)
    x = tf.keras.layers.GlobalAvgPool1D()(x)
    y = tf.keras.layers.Dense(9, activation="softmax")(x)
    model = tf.keras.Model(ip, y)
    return model

def get_compiled_model(input_shape):
    model = make_model(input_shape)
    
    model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    return model

model = get_compiled_model([128, 1])

training = group_padded(train_ds, 64)
validation = group_padded(val_ds, 64)
testing = group_padded(test_ds, 64)

model.fit(training, validation_data=validation, epochs=5)
model.evaluate(testing)




