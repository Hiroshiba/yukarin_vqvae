import tensorflow as tf
import tensorflow_addons as tfa


def WeightNormConv1D(*args, **kwargs):
    # return tfa.layers.WeightNormalization(tf.keras.layers.Conv1D(*args, **kwargs))
    return tf.keras.layers.Conv1D(*args, **kwargs)


class ResidualConv1DGLU(tf.keras.Model):
    def __init__(
        self, hidden_size: int, kernel_size: int, dilation_rate: int, is_last: bool
    ):
        super().__init__()
        self.is_last = is_last

        self.dilated_conv = WeightNormConv1D(
            hidden_size * 2,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
        )
        self.conv_cond = WeightNormConv1D(hidden_size * 2, kernel_size=1)
        self.conv_skip = WeightNormConv1D(hidden_size, kernel_size=1)
        self.conv_out = (
            WeightNormConv1D(hidden_size, kernel_size=1) if not is_last else None
        )

    def call(self, x, c):
        h = self.dilated_conv(x)
        x_tanh, x_sigmoid = tf.split(h, num_or_size_splits=2, axis=2)

        c = self.conv_cond(c)
        c_tanh, c_sigmoid = tf.split(c, num_or_size_splits=2, axis=2)

        x_tanh, x_sigmoid = x_tanh + c_tanh, x_sigmoid + c_sigmoid
        h = tf.nn.tanh(x_tanh) * tf.nn.sigmoid(x_sigmoid)

        s = self.conv_skip(h)

        if not self.is_last:
            h = self.conv_out(h)
            h = h + x
            return h, s
        else:
            return None, s


class WaveNet(tf.keras.Model):
    def __init__(self, hidden_size: int, bin_size: int):
        super().__init__()

        self.prev_conv = WeightNormConv1D(hidden_size, kernel_size=1)

        self.blocks = [
            ResidualConv1DGLU(
                hidden_size,
                kernel_size=2,
                dilation_rate=2 ** i_block,
                is_last=(i_stack == 2 - 1 and i_block == 10 - 1),
            )
            for i_stack in range(2)
            for i_block in range(10)
        ]

        self.add = tf.keras.layers.Add()
        self.post_layers = tf.keras.Sequential(
            [
                tf.keras.layers.ReLU(),
                WeightNormConv1D(hidden_size, kernel_size=1),
                tf.keras.layers.ReLU(),
                WeightNormConv1D(bin_size, kernel_size=1),
            ]
        )

    def call(self, x, c):
        h = self.prev_conv(x)

        skips = []
        for block in self.blocks.layers:
            h, s = block(h, c)
            skips.append(s)

        h = self.add(skips)
        h = self.post_layers(h)
        return h
