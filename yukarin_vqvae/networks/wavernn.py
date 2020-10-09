import tensorflow as tf


class WaveRNN(tf.keras.Model):
    def __init__(self, hidden_size: int, bin_size: int):
        super().__init__()

        self.wave_concat = tf.keras.layers.Concatenate()
        self.lstm = tf.keras.layers.LSTM(
            hidden_size, return_sequences=True, return_state=True,
        )
        self.linear1 = tf.keras.layers.Dense(hidden_size, activation="relu")
        self.linear2 = tf.keras.layers.Dense(bin_size)

    def call(self, x, c):
        h = self.wave_concat([x, c])
        h, _, _ = self.lstm(h)
        h = self.linear1(h)
        h = self.linear2(h)
        return h
