import tensorflow as tf
from yukarin_vqvae.config import NetworkConfig
from yukarin_vqvae.networks.decoder import Decoder
from yukarin_vqvae.networks.encoder import Encoder
from yukarin_vqvae.networks.quantizer import Quantizer


class Predictor(tf.keras.Model):
    def __init__(self, encoder: Encoder, quantizer: Quantizer, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

    def call(
        self,
        encoder_wave: tf.Tensor,
        decoder_wave: tf.Tensor,
        speaker_id: tf.Tensor,
        is_training: bool,
    ):
        encoded = self.encoder(encoder_wave, speaker_id=speaker_id)
        quantized = self.quantizer(encoded, is_training=is_training)
        decoded = self.decoder(
            quantized=quantized, wave=decoder_wave, speaker_id=speaker_id
        )
        return dict(encoded=encoded, quantized=quantized, decoded=decoded)


def create_predictor(config: NetworkConfig, quantizer_ema_decay: float):
    encoder = Encoder(
        scaling_layer_num=config.scaling_layer_num,
        scaling_hidden_size=config.scaling_hidden_size,
        residual_layer_num=config.residual_layer_num,
        residual_hidden_size=config.residual_hidden_size,
        speaker_size=config.speaker_size,
        speaker_embedding_size=config.speaker_embedding_size,
        quantizer_embedding_size=config.quantizer_embedding_size,
    )
    quantizer = Quantizer(
        embedding_num=config.quantizer_embedding_num,
        embedding_size=config.quantizer_embedding_size,
        ema_decay=quantizer_ema_decay,
    )
    decoder = Decoder(
        scaling_layer_num=config.scaling_layer_num,
        scaling_hidden_size=config.scaling_hidden_size,
        residual_layer_num=config.residual_layer_num,
        residual_hidden_size=config.residual_hidden_size,
        vocoder_type=config.vocoder_type,
        vocoder_hidden_size=config.vocoder_hidden_size,
        bin_size=config.bin_size,
        speaker_size=config.speaker_size,
        embedding_size=config.speaker_embedding_size,
    )
    return Predictor(encoder=encoder, quantizer=quantizer, decoder=decoder)
