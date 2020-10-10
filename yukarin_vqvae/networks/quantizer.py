import tensorflow as tf


class Quantizer(tf.keras.Model):
    def __init__(
        self,
        embedding_num: int,
        embedding_size: int,
        ema_decay: float,
    ):
        super().__init__()
        self.embedding_num = embedding_num
        self.ema_decay = ema_decay

    def build(self, shape):
        self.is_initialized = self.add_weight(
            name="is_initialized",
            dtype=tf.bool,
            initializer=tf.keras.initializers.Zeros(),
            trainable=False,
        )
        self.embedding = self.add_weight(
            name="embedding",
            shape=[self.embedding_num, int(shape[-1])],
            trainable=False,
            dtype=tf.float32,
        )
        self.count = self.add_weight(
            "count",
            shape=[self.embedding_num],
            initializer=tf.keras.initializers.Ones(),
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.SUM,
            dtype=tf.float32,
        )
        self.sum = self.add_weight(
            "sum",
            shape=[self.embedding_num, int(shape[-1])],
            synchronization=tf.VariableSynchronization.ON_READ,
            trainable=False,
            aggregation=tf.VariableAggregation.SUM,
            dtype=tf.float32,
        )

    def call(self, x, is_training: bool):
        embedding_size = tf.shape(x)[-1]
        flat_input = tf.reshape(x, [-1, embedding_size])

        if not self.is_initialized:
            self.is_initialized.assign(True)

            r = tf.random.uniform(
                shape=self.count.shape, maxval=tf.shape(flat_input)[0], dtype=tf.int32
            )
            init_embedding = tf.gather(flat_input, indices=r)
            self.embedding.assign(init_embedding)
            self.sum.assign(init_embedding)

        t_embbeding = tf.transpose(self.embedding, [1, 0])
        distance = (
            tf.reduce_sum(flat_input ** 2, 1, keepdims=True)
            - 2 * tf.matmul(flat_input, t_embbeding)
            + tf.reduce_sum(t_embbeding ** 2, 0, keepdims=True)
        )

        encoding_index = tf.argmin(distance, 1)

        if is_training:
            onehot = tf.one_hot(encoding_index, self.embedding_num, axis=0)
            count = tf.reduce_sum(onehot, axis=1)
            self.count.assign_add((1 - self.ema_decay) * (count - self.count))

            usage = tf.cast(self.count >= 1, dtype=self.count.dtype)
            self.count.assign(self.count * usage + (1 - usage))

            new_sum = tf.matmul(onehot, flat_input)
            self.sum.assign_add((1 - self.ema_decay) * (new_sum - self.sum))

            r = tf.random.uniform(
                shape=usage.shape, maxval=tf.shape(flat_input)[0], dtype=tf.int32
            )
            usage = tf.expand_dims(usage, axis=1)

            embedding = self.sum / tf.expand_dims(self.count, axis=1)
            embedding = usage * embedding + (1 - usage) * tf.gather(
                flat_input, indices=r
            )
            self.embedding.assign(embedding)

        encoding_index = tf.reshape(encoding_index, tf.shape(x)[:-1])

        quantized = tf.nn.embedding_lookup(self.embedding, encoding_index)
        if is_training:
            quantized = x + tf.stop_gradient(quantized - x)
        return quantized
