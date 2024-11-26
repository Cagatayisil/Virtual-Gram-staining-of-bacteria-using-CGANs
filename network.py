import tensorflow as tf
import numpy as np

def Conv(name, x, filter_size, in_filters, out_filters, strides):
    with tf.compat.v1.variable_scope(name):
        n = 1 / np.sqrt(filter_size * filter_size * in_filters)
        kernel = tf.compat.v1.get_variable('filter', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                 initializer=tf.compat.v1.random_uniform_initializer(minval=-n, maxval=n))
        bias = tf.compat.v1.get_variable('bias', [out_filters], tf.float32,
                               initializer=tf.compat.v1.random_uniform_initializer(minval=-n, maxval=n))

        return tf.nn.conv2d(input=x, filters=kernel, strides=[1, strides, strides, 1], padding='SAME') + bias


def Channel_attention(name, x, ratio, n_feats):
    _res = x

    x = tf.reduce_mean(input_tensor=x, axis=[1, 2], keepdims=True)
    x = Conv(name + '_conv1', x, 1, n_feats, n_feats // ratio, 1)
    x = tf.nn.relu(x)

    x = Conv(name + '_conv2', x, 1, n_feats // ratio, n_feats, 1)
    x = tf.nn.sigmoid(x)
    x = tf.multiply(x, _res)

    return x

def RCA_Block(name, x, filter_size, ratio, n_feats):
    _res = x

    x = Conv(name + '_conv1', x, filter_size, n_feats, n_feats, 1)
    x = tf.nn.relu(x)
    x = Conv(name + '_conv2', x, filter_size, n_feats, n_feats, 1)

    x = Channel_attention(name + '_CA', x, ratio, n_feats)

    x = x + _res

    return x

def conv2d(inp, shp, name, strides=(1,1,1,1), padding='SAME'):
    with tf.device('/cpu:0'):
        filters = tf.compat.v1.get_variable(name + '/filters', shp, initializer=tf.compat.v1.truncated_normal_initializer(stddev=np.sqrt(2.0/(shp[0]*shp[1]*shp[3]))))
        biases = tf.compat.v1.get_variable(name + '/biases', [shp[-1]], initializer=tf.compat.v1.constant_initializer(0))
    return tf.nn.bias_add(tf.nn.conv2d(input=inp, filters=filters, strides=strides, padding=padding), biases)

def leakyRelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def Silu(x):
    return x*tf.math.sigmoid(x)

def residual_block(inp, name):
    ch = inp.get_shape().as_list()[-1]
    conv1 = conv2d(inp, [3,3,ch,ch], name + '/conv1')
    conv1 = leakyRelu(conv1)
    conv2 = conv2d(conv1, [3,3,ch,ch], name + '/conv2')
    conv2 = leakyRelu(conv2)
    return conv2 + inp

def normal_block(inp, name):
    ch = inp.get_shape().as_list()[-1]
    conv1 = conv2d(inp, [3,3,ch,ch], name + '/conv1')
    conv1 = leakyRelu(conv1)
    conv2 = conv2d(conv1, [3,3,ch,ch*2], name + '/conv2', strides=(1,2,2,1))
    conv2 = leakyRelu(conv2)
    return conv2

def fc_layer(inp, shp, name):
    with tf.device('/cpu:0'):
        weights = tf.compat.v1.get_variable(name + '/weights', shp, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
        biases = tf.compat.v1.get_variable(name + '/biases', [shp[-1]], initializer=tf.compat.v1.constant_initializer(0))
    return tf.nn.bias_add(tf.matmul(inp, weights), biases)

def normalize(x):
    x1=x
    #x1,x2 = tf.split(x, 2, axis = -1)
    # a = tf.reduce_min(x1,axis = [1,2,3],keepdims=True)
    # print(a.get_shape)
    # exit()
    x1 = (x1-tf.reduce_min(input_tensor=x1,axis = [1,2,3],keepdims=True))/(tf.reduce_max(input_tensor=x1,axis = [1,2,3],keepdims=True)-tf.reduce_min(input_tensor=x1,axis = [1,2,3],keepdims=True))
    #x2 = (x2-tf.reduce_min(x2,axis = [1,2,3],keepdims=True))/(tf.reduce_max(x2,axis = [1,2,3],keepdims=True)-tf.reduce_min(x2,axis = [1,2,3],keepdims=True))
    return x1 #tf.concat([x1,x2],-1)

class Generator(object):
    def __init__(self, inp, config):
        N = config.n_levels
        self.dic = {}
        in_ch = inp.get_shape().as_list()[-1]
        cur = leakyRelu(conv2d(inp, [3,3,in_ch,config.n_channels], 'conv1'))
        for i in range(N):
            cur = self.down(cur, 'down{}'.format(i+1))

        ch = cur.get_shape().as_list()[-1]
        cur = leakyRelu(conv2d(cur, [3,3,ch,ch], 'center'))

        for i in range(N):
            cur = self.up(cur, config.image_size >> (N-i-1), 'up{}'.format(N-i))
        ch = cur.get_shape().as_list()[-1]
        self.output = conv2d(cur, [3,3,ch,3], 'last_layer')

    def down(self, inp, name):
        ch = inp.get_shape().as_list()[-1]
        conv1 = leakyRelu(conv2d(inp, [3,3,ch,ch], name + '/conv1'))
        conv2 = leakyRelu(conv2d(conv1, [3,3,ch,ch*2], name + '/conv2'))
        tmp = tf.pad(tensor=inp, paddings=[[0,0], [0,0], [0,0], [0,ch]], mode='CONSTANT')
        self.dic[name] = conv2 + tmp
        return tf.nn.avg_pool2d(input=self.dic[name], ksize=(1,2,2,1), strides=(1,2,2,1), padding='SAME')

    def up(self, inp, size, name):
        ch = inp.get_shape().as_list()[-1]
        image = tf.compat.v1.image.resize_bilinear(inp, [size, size])
        if name.replace('up', 'down') in self.dic:

            image = tf.concat([image, self.dic[name.replace('up', 'down')]], axis=3)
        conv1 = leakyRelu(conv2d(image, [3,3,2*ch,ch], name + '/conv1'))
        return  leakyRelu(conv2d(conv1, [3,3,ch,ch//2], name + '/conv2'))

class Discriminator(object):
    def __init__(self, inp, config):
        in_ch = inp.get_shape().as_list()[-1]
        cur = leakyRelu(conv2d(inp, [3,3,in_ch,config.n_channels], 'conv1'))
        for i in range(5):
            cur = normal_block(cur, 'n_block{}'.format(i))
            print(cur.get_shape())
        cur = tf.reduce_mean(input_tensor=cur, axis=(1,2))
        ch = cur.get_shape().as_list()[-1]
        cur = leakyRelu(fc_layer(cur, [ch, ch], 'fcl1'))
        self.output = tf.nn.sigmoid(fc_layer(cur, [ch, 1], 'fcl2'))
