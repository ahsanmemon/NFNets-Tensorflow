

import tensorflow as tf
from keras import backend as K
from .nfnet_layers import WSConv2D, SqueezeExcite, StochasticDepth


def GLU(inputs):
    y = None
    if type(inputs) is list:
        x, g = inputs
        d_x = K.int_shape(x)[-1]
        d_g = K.int_shape(g)[-1]
        if d_g // 2 == d_x: y, g = tf.split(g, num_or_size_splits = 2, axis = -1)
        else: assert d_g == d_x
    else: x, g = tf.split(inputs, num_or_size_splits = 2, axis = -1)
    _g = tf.sigmoid(g)
    _y = x * _g
    if y is not None:
        _y += (1 - _g) * y
    return _y

nfnet_params = {}

nfnet_params.update(**{
    'F0': {
        'width': [256, 512, 1536, 1536], 'depth': [1, 2, 6, 3],
        'train_imsize': 192, 'test_imsize': 256,
        'RA_level': '405', 'drop_rate': 0.2},
    'F1': {
        'width': [256, 512, 1536, 1536], 'depth': [2, 4, 12, 6],
        'train_imsize': 224, 'test_imsize': 320,
        'RA_level': '410', 'drop_rate': 0.3},
    'F2': {
        'width': [256, 512, 1536, 1536], 'depth': [3, 6, 18, 9],
        'train_imsize': 256, 'test_imsize': 352,
        'RA_level': '410', 'drop_rate': 0.4},
    'F3': {
        'width': [256, 512, 1536, 1536], 'depth': [4, 8, 24, 12],
        'train_imsize': 320, 'test_imsize': 416,
        'RA_level': '415', 'drop_rate': 0.4},
    'F4': {
        'width': [256, 512, 1536, 1536], 'depth': [5, 10, 30, 15],
        'train_imsize': 384, 'test_imsize': 512,
        'RA_level': '415', 'drop_rate': 0.5},
    'F5': {
        'width': [256, 512, 1536, 1536], 'depth': [6, 12, 36, 18],
        'train_imsize': 416, 'test_imsize': 544,
        'RA_level': '415', 'drop_rate': 0.5},
    'F6': {
        'width': [256, 512, 1536, 1536], 'depth': [7, 14, 42, 21],
        'train_imsize': 448, 'test_imsize': 576,
        'RA_level': '415', 'drop_rate': 0.5},
    'F7': {
        'width': [256, 512, 1536, 1536], 'depth': [8, 16, 48, 24],
        'train_imsize': 480, 'test_imsize': 608,
        'RA_level': '415', 'drop_rate': 0.5},
})

nfnet_params.update(**{ **{f'{key}+': {**nfnet_params[key], 'width': [384, 768, 2048, 2048],} for key in nfnet_params} })

# Nonlinearities with magic constants (gamma) baked in.
nonlinearities = {
    'identity': tf.keras.layers.Lambda(lambda x: x),
    'celu': tf.keras.layers.Lambda(lambda x: tf.nn.crelu(x) * 1.270926833152771),
    'elu': tf.keras.layers.Lambda(lambda x: tf.keras.activations.elu(x) * 1.2716004848480225),
    'gelu': tf.keras.layers.Lambda(lambda x: tf.keras.activations.gelu(x) * 1.7015043497085571),
#     'glu': lambda x: jax.nn.glu(x) * 1.8484294414520264,
    'leaky_relu': tf.keras.layers.Lambda(lambda x: tf.nn.leaky_relu(x) * 1.70590341091156),
    'log_sigmoid': tf.keras.layers.Lambda(lambda x: tf.math.log(tf.nn.sigmoid(x)) * 1.9193484783172607),
    'log_softmax': tf.keras.layers.Lambda(lambda x: tf.math.log(tf.nn.softmax(x)) * 1.0002083778381348),
    'relu': tf.keras.layers.Lambda(lambda x: tf.keras.activations.relu(x) * 1.7139588594436646),
    'relu6': tf.keras.layers.Lambda(lambda x: tf.nn.relu6(x) * 1.7131484746932983),
    'selu': tf.keras.layers.Lambda(lambda x: tf.keras.activations.selu(x) * 1.0008515119552612),
    'sigmoid': tf.keras.layers.Lambda(lambda x: tf.keras.activations.sigmoid(x) * 4.803835391998291),
    'silu': tf.keras.layers.Lambda(lambda x: tf.nn.silu(x) * 1.7881293296813965),
    'soft_sign': tf.keras.layers.Lambda(lambda x: tf.nn.softsign(x) * 2.338853120803833),
    'softplus': tf.keras.layers.Lambda(lambda x: tf.keras.activations.softplus(x) * 1.9203323125839233),
    'tanh': tf.keras.layers.Lambda(lambda x: tf.keras.activations.tanh(x) * 1.5939117670059204),
}


class NFNet(tf.keras.Model):
    variant_dict = nfnet_params

    def __init__(self, num_classes=None, variant = 'F0', width = 1.0, se_ratio = 0.5, alpha = 0.2, stochdepth_rate = 0.1, drop_rate = None, activation = 'gelu', fc_init = None, final_conv_mult = 2, final_conv_ch = None, use_two_convs = True, name = 'NFNet', include_top = True):

        super(NFNet, self).__init__(name = name)
        block_params = self.variant_dict[self.variant]

        self.width = width
        self.variant = variant
        self.se_ratio = se_ratio
        self.num_classes = num_classes
        self.include_top = include_top
        self.width_pattern = block_params['width']
        self.depth_pattern = block_params['depth']
        self.activation = nonlinearities[activation]
        self.test_imsize = block_params['test_imsize']
        self.train_imsize = block_params['train_imsize']
        self.big_pattern = block_params.get('big_width', [True] * 4)
        self.bneck_pattern = block_params.get('expansion', [0.5] * 4)
        self.group_pattern = block_params.get('group_width', [128] * 4)
        if drop_rate is None: self.drop_rate = block_params['drop_rate']
        else: self.drop_rate = drop_rate
        self.which_conv = WSConv2D
        ch = self.width_pattern[0] // 2
        self.stem = tf.keras.Sequential([self.which_conv(16, kernel_size = 3, strides = 2, padding = 'same', name = 'stem_conv0'), self.activation, self.which_conv(32, kernel_size = 3, strides = 1, padding = 'same', name = 'stem_conv1'), self.activation, self.which_conv(64, kernel_size = 3, strides = 1, padding = 'same', name = 'stem_conv2'), self.activation, self.which_conv(ch, kernel_size = 3, strides = 2, padding = 'same', name = 'stem_conv3'), ])
        self.blocks = []
        expected_std = 1.0
        num_blocks = sum(self.depth_pattern)
        index = 0
        stride_pattern = [1, 2, 2, 2]
        block_args = zip(self.width_pattern, self.depth_pattern, self.bneck_pattern, self.group_pattern, self.big_pattern, stride_pattern)
        for (block_width, stage_depth, expand_ratio, group_size, big_width, stride) in block_args:
            for block_index in range(stage_depth):
                beta = 1. / expected_std
                block_stochdepth_rate = stochdepth_rate * index / num_blocks
                out_ch = (int(block_width * self.width))
                self.blocks += [NFBlock(ch, out_ch, expansion = expand_ratio, se_ratio = se_ratio, group_size = group_size, stride = stride if block_index == 0 else 1, beta = beta, alpha = alpha, activation = self.activation, which_conv = self.which_conv, stochdepth_rate = block_stochdepth_rate, big_width = big_width, use_two_convs = use_two_convs, )]
                ch = out_ch
                index += 1
                if block_index == 0: expected_std = 1.0
                expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5
        if final_conv_mult is None:
            if final_conv_ch is None:
                raise ValueError('Must provide one of final_conv_mult or final_conv_ch')
            ch = final_conv_ch
        else:
            ch = int(final_conv_mult * ch)
        self.final_conv = self.which_conv(ch, kernel_size = 1, padding = 'same', name = 'final_conv')

        if include_top:
            # By default, initialize with N(0, 0.01)
            if fc_init is None:
                fc_init = tf.keras.initializers.RandomNormal(mean = 0, stddev = 0.01)
            self.fc = tf.keras.layers.Dense(self.num_classes, kernel_initializer = fc_init, use_bias = True)

    def call(self, x, training = True):
        outputs = {}
        out = self.stem(x)
        for i, block in enumerate(self.blocks): out, res_avg_var = block(out, training = training)
        out = self.activation(self.final_conv(out))
        if self.include_top:
            pool = tf.math.reduce_mean(out, [1, 2])
            outputs['pool'] = pool
            if self.drop_rate > 0.0 and training: pool = tf.keras.layers.Dropout(self.drop_rate)(pool)
            outputs['logits'] = self.fc(pool)
            return outputs
        else: return out


class NFBlock(tf.keras.Model):

    def __init__(self, in_ch, out_ch, expansion = 0.5, se_ratio = 0.5, kernel_shape = 3, group_size = 128, stride = 1, beta = 1.0, alpha = 0.2, which_conv = WSConv2D, activation = tf.keras.activations.gelu, big_width = True, use_two_convs = True, stochdepth_rate = None, name = None):
        super(NFBlock, self).__init__(name = name)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.expansion = expansion
        self.se_ratio = se_ratio
        self.kernel_shape = kernel_shape
        self.activation = activation
        self.beta, self.alpha = beta, alpha
        width = int((self.out_ch if big_width else self.in_ch) * expansion)
        self.groups = width // group_size
        self.width = group_size * self.groups
        self.stride = stride
        self.use_two_convs = use_two_convs
        self.conv0 = which_conv(filters = self.width, kernel_size = 1, padding = 'same', name = 'conv0')
        self.conv1 = which_conv(filters = self.width, kernel_size = kernel_shape, strides = stride, padding = 'same', groups = self.groups, name = 'conv1')
        if self.use_two_convs:
            self.conv1b = which_conv(filters = self.width, kernel_size = kernel_shape, strides = 1, padding = 'same', groups = self.groups, name = 'conv1b')
        self.conv2 = which_conv(filters = self.out_ch, kernel_size = 1, padding = 'same', name = 'conv2')
        self.use_projection = stride > 1 or self.in_ch != self.out_ch
        if self.use_projection:
            self.conv_shortcut = which_conv(filters = self.out_ch, kernel_size = 1, padding = 'same', name = 'conv_shortcut')
        self.se = SqueezeExcite(self.out_ch, self.out_ch, self.se_ratio)
        self._has_stochdepth = (stochdepth_rate is not None and stochdepth_rate > 0.0 and stochdepth_rate < 1.0)
        if self._has_stochdepth:
            self.stoch_depth = StochasticDepth(stochdepth_rate)

    def call(self, x, training):
        out = self.activation(x) * self.beta
        if self.stride > 1:
            shortcut = tf.keras.layers.AveragePooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same')(out)
            if self.use_projection:
                shortcut = self.conv_shortcut(shortcut)
        elif self.use_projection:
            shortcut = self.conv_shortcut(out)
        else:
            shortcut = x
        out = self.conv0(out)
        out = self.conv1(self.activation(out))
        if self.use_two_convs:
            out = self.conv1b(self.activation(out))
        out = self.conv2(self.activation(out))
        out = (self.se(out) * 2) * out
        res_avg_var = tf.math.reduce_mean(tf.math.reduce_variance(out, axis = [0, 1, 2]))
        if self._has_stochdepth:
            out = self.stoch_depth(out, training)
        out = out * self.add_weight(name = 'skip_gain', shape = (), initializer = "zeros", trainable = True, dtype = out.dtype)
        return out * self.alpha + shortcut, res_avg_var


def NFNetF0(num_classes=None, width = 1.0, se_ratio = 0.5, alpha = 0.2, stochdepth_rate = 0.1, drop_rate = None, activation = 'gelu', fc_init = None, final_conv_mult = 2, final_conv_ch = None, use_two_convs = True, name = 'NFNet', include_top = True): return NFNet(num_classes=num_classes, variant = 'F0', width = width, se_ratio = se_ratio, alpha = alpha, stochdepth_rate = stochdepth_rate, drop_rate = drop_rate, activation = activation, fc_init = fc_init, final_conv_mult = final_conv_mult, final_conv_ch = final_conv_ch, use_two_convs = use_two_convs, name = name, include_top = include_top)
def NFNetF1(num_classes=None, width = 1.0, se_ratio = 0.5, alpha = 0.2, stochdepth_rate = 0.1, drop_rate = None, activation = 'gelu', fc_init = None, final_conv_mult = 2, final_conv_ch = None, use_two_convs = True, name = 'NFNet', include_top = True): return NFNet(num_classes=num_classes, variant = 'F1', width = width, se_ratio = se_ratio, alpha = alpha, stochdepth_rate = stochdepth_rate, drop_rate = drop_rate, activation = activation, fc_init = fc_init, final_conv_mult = final_conv_mult, final_conv_ch = final_conv_ch, use_two_convs = use_two_convs, name = name, include_top = include_top)
def NFNetF2(num_classes=None, width = 1.0, se_ratio = 0.5, alpha = 0.2, stochdepth_rate = 0.1, drop_rate = None, activation = 'gelu', fc_init = None, final_conv_mult = 2, final_conv_ch = None, use_two_convs = True, name = 'NFNet', include_top = True): return NFNet(num_classes=num_classes, variant = 'F2', width = width, se_ratio = se_ratio, alpha = alpha, stochdepth_rate = stochdepth_rate, drop_rate = drop_rate, activation = activation, fc_init = fc_init, final_conv_mult = final_conv_mult, final_conv_ch = final_conv_ch, use_two_convs = use_two_convs, name = name, include_top = include_top)
def NFNetF3(num_classes=None, width = 1.0, se_ratio = 0.5, alpha = 0.2, stochdepth_rate = 0.1, drop_rate = None, activation = 'gelu', fc_init = None, final_conv_mult = 2, final_conv_ch = None, use_two_convs = True, name = 'NFNet', include_top = True): return NFNet(num_classes=num_classes, variant = 'F3', width = width, se_ratio = se_ratio, alpha = alpha, stochdepth_rate = stochdepth_rate, drop_rate = drop_rate, activation = activation, fc_init = fc_init, final_conv_mult = final_conv_mult, final_conv_ch = final_conv_ch, use_two_convs = use_two_convs, name = name, include_top = include_top)
def NFNetF4(num_classes=None, width = 1.0, se_ratio = 0.5, alpha = 0.2, stochdepth_rate = 0.1, drop_rate = None, activation = 'gelu', fc_init = None, final_conv_mult = 2, final_conv_ch = None, use_two_convs = True, name = 'NFNet', include_top = True): return NFNet(num_classes=num_classes, variant = 'F4', width = width, se_ratio = se_ratio, alpha = alpha, stochdepth_rate = stochdepth_rate, drop_rate = drop_rate, activation = activation, fc_init = fc_init, final_conv_mult = final_conv_mult, final_conv_ch = final_conv_ch, use_two_convs = use_two_convs, name = name, include_top = include_top)
def NFNetF5(num_classes=None, width = 1.0, se_ratio = 0.5, alpha = 0.2, stochdepth_rate = 0.1, drop_rate = None, activation = 'gelu', fc_init = None, final_conv_mult = 2, final_conv_ch = None, use_two_convs = True, name = 'NFNet', include_top = True): return NFNet(num_classes=num_classes, variant = 'F5', width = width, se_ratio = se_ratio, alpha = alpha, stochdepth_rate = stochdepth_rate, drop_rate = drop_rate, activation = activation, fc_init = fc_init, final_conv_mult = final_conv_mult, final_conv_ch = final_conv_ch, use_two_convs = use_two_convs, name = name, include_top = include_top)
def NFNetF6(num_classes=None, width = 1.0, se_ratio = 0.5, alpha = 0.2, stochdepth_rate = 0.1, drop_rate = None, activation = 'gelu', fc_init = None, final_conv_mult = 2, final_conv_ch = None, use_two_convs = True, name = 'NFNet', include_top = True): return NFNet(num_classes=num_classes, variant = 'F6', width = width, se_ratio = se_ratio, alpha = alpha, stochdepth_rate = stochdepth_rate, drop_rate = drop_rate, activation = activation, fc_init = fc_init, final_conv_mult = final_conv_mult, final_conv_ch = final_conv_ch, use_two_convs = use_two_convs, name = name, include_top = include_top)
def NFNetF7(num_classes=None, width = 1.0, se_ratio = 0.5, alpha = 0.2, stochdepth_rate = 0.1, drop_rate = None, activation = 'gelu', fc_init = None, final_conv_mult = 2, final_conv_ch = None, use_two_convs = True, name = 'NFNet', include_top = True): return NFNet(num_classes=num_classes, variant = 'F7', width = width, se_ratio = se_ratio, alpha = alpha, stochdepth_rate = stochdepth_rate, drop_rate = drop_rate, activation = activation, fc_init = fc_init, final_conv_mult = final_conv_mult, final_conv_ch = final_conv_ch, use_two_convs = use_two_convs, name = name, include_top = include_top)
