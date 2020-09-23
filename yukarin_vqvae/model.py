import tensorflow as tf

from yukarin_vqvae.config import ModelConfig
from yukarin_vqvae.networks.predictor import Predictor


class Model(tf.keras.Model):
    def __init__(self, config: ModelConfig, predictor: Predictor, replica_size: int):
        super().__init__()
        self.config = config
        self.predictor = predictor
        self.replica_size = replica_size

        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.softmax_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

    def call(
        self, inputs, training: bool,
    ):
        wave = inputs["wave"]
        target = inputs["target"]
        speaker_id = inputs["speaker_id"]

        output = self.predictor(
            encoder_wave=wave[:, 1:],
            decoder_wave=wave[:, :-1],
            speaker_id=speaker_id,
            is_training=training,
        )
        encoded = output["encoded"]
        quantized = output["quantized"]
        decoded = output["decoded"]

        quantize_loss = tf.reduce_mean(self.mse(tf.stop_gradient(quantized), encoded))
        if self.config.quantize_loss_weight != 1:
            quantize_loss *= self.config.quantize_loss_weight
        self.add_metric(quantize_loss, name="quantize_loss")

        softmax_loss = tf.reduce_mean(
            self.softmax_cross_entropy(y_true=target, y_pred=decoded)
        )
        if self.config.softmax_loss_weight != 1:
            softmax_loss *= self.config.softmax_loss_weight
        self.add_metric(softmax_loss, name="softmax_loss")

        loss = quantize_loss + softmax_loss
        self.add_metric(loss, name="loss")

        loss /= tf.cast(self.replica_size, dtype=tf.float32)
        self.add_loss(loss)
