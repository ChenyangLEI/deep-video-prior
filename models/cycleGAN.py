import tensorflow as tf
import tensorflow.contrib.slim as slim


def weight_variable(shape, name=None):
    # initialize weighted variables.
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial, name=name)

def conv2d(x, W, strides=[1, 1, 1, 1], p='SAME', name=None):
    # set convolution layers.
    assert isinstance(x, tf.Tensor)
    return tf.nn.conv2d(x, W, strides=strides, padding=p, name=name)

def batch_norm(x):
    assert isinstance(x, tf.Tensor)
    # reduce dimension 1, 2, 3, which would produce batch mean and batch variance.
    mean, var = tf.nn.moments(x, axes=[1, 2, 3])
    return tf.nn.batch_normalization(x, mean, var, 0, 1, 1e-5)

def relu(x):
    assert isinstance(x, tf.Tensor)
    return tf.nn.relu(x)

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x)

def deconv2d(x, W, strides=[1, 1, 1, 1], p='SAME', name=None):
    assert isinstance(x, tf.Tensor)
    _, _, c, _ = W.get_shape().as_list()
    b, h, w, _ = x.get_shape().as_list()
    # print(b,h,w)
    return tf.nn.conv2d_transpose(x, W, [b, strides[1] * h, strides[1] * w, c], strides=strides, padding=p, name=name)
    # return slim.conv2d_transpose(x,out_channel,[3,3],stride=2,padding='SAME',activation_fn=tf.nn.relu, normalizer_fn=nm,
                                # weights_initializer=tf.contrib.layers.xavier_initializer(),scope=name)

def slim_deconv2d(x, W, strides=2, p='SAME', name=None, out_channel=64):
    assert isinstance(x, tf.Tensor)

    return slim.conv2d_transpose(x,out_channel,[3,3],stride=2,padding='SAME',activation_fn=tf.nn.relu, normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),scope=name)

def max_pool_2x2(x):
    assert isinstance(x, tf.Tensor)
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class ResidualBlock():
    def __init__(self, idx, ksize=3, train=False, data_dict=None):
        self.W1 = weight_variable([ksize, ksize, 128, 128], name='R'+ str(idx) + '_conv1_w')
        self.W2 = weight_variable([ksize, ksize, 128, 128], name='R'+ str(idx) + '_conv2_w')
    def __call__(self, x, idx, strides=[1, 1, 1, 1]):
        # h = relu(batch_norm(conv2d(x, self.W1, strides, name='R' + str(idx) + '_conv1')))
        # h = batch_norm(conv2d(h, self.W2, name='R' + str(idx) + '_conv2'))
        h = relu(conv2d(x, self.W1, strides, name='R' + str(idx) + '_conv1'))
        h = conv2d(h, self.W2, name='R' + str(idx) + '_conv2')
        return x + h


class FastStyleNet():
    def __init__(self, train=True, data_dict=None):
        self.c1 = weight_variable([9, 9, 3, 32], name='t_conv1_w')
        self.c2 = weight_variable([4, 4, 32, 64], name='t_conv2_w')
        self.c3 = weight_variable([4, 4, 64, 128], name='t_conv3_w')
        self.r1 = ResidualBlock(1, train=train)
        self.r2 = ResidualBlock(2, train=train)
        self.r3 = ResidualBlock(3, train=train)
        self.r4 = ResidualBlock(4, train=train)
        self.r5 = ResidualBlock(5, train=train)
        self.d1 = weight_variable([4, 4, 64, 128], name='t_dconv1_w')
        self.d2 = weight_variable([4, 4, 32, 64], name='t_dconv2_w')
        self.d3 = weight_variable([9, 9, 3, 32], name='t_dconv3_w')            
    def __call__(self, h):
        # h = batch_norm(relu(conv2d(h, self.c1, name='t_conv1')))
        # h = batch_norm(relu(conv2d(h, self.c2, strides=[1, 2, 2, 1], name='t_conv2')))
        # h = batch_norm(relu(conv2d(h, self.c3, strides=[1, 2, 2, 1], name='t_conv3')))
        h = relu(conv2d(h, self.c1, name='t_conv1'))
        h = relu(conv2d(h, self.c2, strides=[1, 2, 2, 1], name='t_conv2'))
        h = relu(conv2d(h, self.c3, strides=[1, 2, 2, 1], name='t_conv3'))

        h = self.r1(h, 1)
        h = self.r2(h, 2)
        h = self.r3(h, 3)
        h = self.r4(h, 4)
        h = self.r5(h, 5)
        
        h = relu(deconv2d(h, self.d1, strides=[1, 2, 2, 1], name='t_deconv1'))
        h = relu(deconv2d(h, self.d2, strides=[1, 2, 2, 1], name='t_deconv2'))
        y = deconv2d(h, self.d3, name='t_deconv3')
        # print(y.)
        return tf.multiply((tf.tanh(y) + 1), tf.constant(127.5, tf.float32, shape=y.get_shape()), name='output')

def cycleGAN_Net(h):
    c1 = weight_variable([9, 9, 3, 32], name='t_conv1_w')
    c2 = weight_variable([4, 4, 32, 64], name='t_conv2_w')
    c3 = weight_variable([4, 4, 64, 128], name='t_conv3_w')
    r1 = ResidualBlock(1)
    r2 = ResidualBlock(2)
    r3 = ResidualBlock(3)
    r4 = ResidualBlock(4)
    r5 = ResidualBlock(5)
    d1 = weight_variable([4, 4, 64, 128], name='t_dconv1_w')
    d2 = weight_variable([4, 4, 32, 64], name='t_dconv2_w')
    d3 = weight_variable([9, 9, 3, 32], name='t_dconv3_w')     

    h = relu(conv2d(h, c1, name='t_conv1'))
    h = relu(conv2d(h, c2, strides=[1, 2, 2, 1], name='t_conv2'))
    h = relu(conv2d(h, c3, strides=[1, 2, 2, 1], name='t_conv3'))

    h = r1(h, 1)
    h = r2(h, 2)
    h = r3(h, 3)
    h = r4(h, 4)
    h = r5(h, 5)
    
    # h = slim_deconv2d(h, d1, strides=2, name='t_deconv1', out_channel=64)
    # h = slim_deconv2d(h, d2, strides=2, name='t_deconv2', out_channel=32)
    # y = slim_deconv2d(h, d3, strides=1, name='t_deconv3', out_channel=3)

    h = relu(deconv2d(h, d1, strides=[1, 2, 2, 1], name='t_deconv1'))
    h = relu(deconv2d(h, d2, strides=[1, 2, 2, 1], name='t_deconv2'))
    y = deconv2d(h, d3, name='t_deconv3')
    return tf.multiply((tf.tanh(y) + 1), tf.constant(127.5, tf.float32, shape=y.get_shape()), name='output')
    # return tf.nn.relu(y)


