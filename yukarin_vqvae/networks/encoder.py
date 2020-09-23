import tensorflow as tf

from yukarin_vqvae.networks.residual import ResidualBlock


class Encoder(tf.keras.Model):
    def __init__(
        self,
        scaling_layer_num: int,
        scaling_hidden_size: int,
        residual_layer_num: int,
        residual_hidden_size: int,
        speaker_size: int,
        speaker_embedding_size: int,
        quantizer_embedding_size: int,
    ):
        super().__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=speaker_size, output_dim=speaker_embedding_size,
        )

        self.concat = tf.keras.layers.Concatenate()

        self.convs = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    scaling_hidden_size,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    activation="relu" if i < scaling_layer_num - 1 else None,
                )
                for i in range(scaling_layer_num)
            ]
        )

        self.blocks = tf.keras.Sequential(
            [
                ResidualBlock(
                    hidden_size=scaling_hidden_size,
                    bottleneck_hidden_size=residual_hidden_size,
                )
                for i in range(residual_layer_num)
            ]
        )

        self.post_conv = tf.keras.layers.Conv1D(quantizer_embedding_size, kernel_size=1)

    def __call__(self, wave, speaker_id):
        s = self.embedding(speaker_id)
        s = tf.broadcast_to(
            s, shape=(tf.shape(s)[0], tf.shape(wave)[1], tf.shape(s)[2])
        )
        h = self.concat([wave, s])
        h = self.convs(h)
        h = self.blocks(h)
        h = self.post_conv(h)
        return h
