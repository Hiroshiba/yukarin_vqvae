import tensorflow as tf
from yukarin_vqvae.networks.residual import ResidualBlock
from yukarin_vqvae.networks.wavenet import WaveNet


class Decoder(tf.keras.Model):
    def __init__(
        self,
        scaling_layer_num: int,
        scaling_hidden_size: int,
        residual_layer_num: int,
        residual_hidden_size: int,
        vocoder_hidden_size: int,
        bin_size: int,
        speaker_size: int,
        embedding_size: int,
    ):
        super().__init__()

        self.embedding = tf.keras.layers.Embedding(
            input_dim=speaker_size, output_dim=embedding_size,
        )

        self.speaker_concat = tf.keras.layers.Concatenate()

        self.pre_conv = tf.keras.layers.Conv1D(scaling_hidden_size, kernel_size=1)

        self.blocks = tf.keras.Sequential(
            [
                ResidualBlock(
                    hidden_size=scaling_hidden_size,
                    bottleneck_hidden_size=residual_hidden_size,
                )
                for i in range(residual_layer_num)
            ]
        )

        self.convs = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1DTranspose(
                    scaling_hidden_size,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    activation="relu" if i < scaling_layer_num - 1 else None,
                )
                for i in range(scaling_layer_num)
            ]
        )

        # self.wave_concat = tf.keras.layers.Concatenate()
        # self.lstm = tf.keras.layers.LSTM(
        #     vocoder_hidden_size, return_sequences=True, return_state=True,
        # )
        # self.linear1 = tf.keras.layers.Dense(vocoder_hidden_size, activation="relu")
        # self.linear2 = tf.keras.layers.Dense(bin_size)
        self.vocoder = WaveNet(hidden_size=vocoder_hidden_size, bin_size=bin_size)

    def call(self, quantized, wave, speaker_id):
        s = self.embedding(speaker_id)
        s = tf.broadcast_to(
            s, shape=(tf.shape(s)[0], tf.shape(quantized)[1], tf.shape(s)[2])
        )
        h = self.speaker_concat([quantized, s])
        h = self.pre_conv(h)
        h = self.blocks(h)
        h = self.convs(h)

        # h = self.wave_concat([h, wave])
        # h, _, _ = self.lstm(h)
        # h = self.linear1(h)
        # h = self.linear2(h)
        h = self.vocoder(x=wave, c=h)
        return h
