import tensorflow as tf


class ResidualBlock(tf.keras.Model):
    def __init__(
        self, hidden_size: int, bottleneck_hidden_size: int,
    ):
        super().__init__()

        self.conv1 = tf.keras.layers.Conv1D(
            bottleneck_hidden_size, kernel_size=3, padding="same",
        )
        self.conv2 = tf.keras.layers.Conv1D(hidden_size, kernel_size=1)

    def call(self, x):
        h = self.conv1(tf.nn.relu(x))
        h = self.conv2(tf.nn.relu(h))
        return x + h
