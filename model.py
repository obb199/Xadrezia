import numpy as np
from data_loader import tf


def gen_positional_encoding(n_patches, d_model):
    indexes = np.arange(d_model // 2)
    divisors = 10000 ** (2 * indexes / d_model)  # shape = d_model//2

    pos = np.arange(n_patches)
    pos = np.expand_dims(pos, -1)  # shape = (n_patchs, 1)

    pos_enc = np.zeros([n_patches, d_model], dtype='float32')

    # shape pos/divisors = (n_patchs, d_model//2)
    pos_enc[:, indexes * 2] = np.sin(pos / divisors)
    pos_enc[:, indexes * 2 + 1] = np.cos(pos / divisors)

    return pos_enc


class Convolution(tf.keras.layers.Layer):
    """
        Bloco convolucional residual com conexão skip.

        Args:
            filters: Número de filtros para as camadas convolucionais.
            prev_filters: Número de filtros da camada anterior.
            kernel_size: Tamanho do kernel para convolução.
            **kwargs: Argumentos adicionais para a classe Layer.
        """

    def __init__(self, filters, **kwargs):
        super(Convolution, self).__init__(**kwargs)
        filters = filters//4
        self.conv_1 = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False),
             tf.keras.layers.Activation('relu'),
             tf.keras.layers.BatchNormalization()])

        self.conv_2 = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(filters, kernel_size=(2, 2), padding='same', use_bias=False),
             tf.keras.layers.Activation('relu'),
             tf.keras.layers.BatchNormalization()])

        self.conv_3 = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(filters, kernel_size=(1, 5), padding='same', use_bias=False),
             tf.keras.layers.Conv2D(filters, kernel_size=(5, 1), padding='same', use_bias=False),
             tf.keras.layers.Activation('relu'),
             tf.keras.layers.BatchNormalization()])

        self.conv_4 = tf.keras.Sequential(
            [tf.keras.layers.Conv2D(filters, kernel_size=(8, 1), padding='same', use_bias=False),
             tf.keras.layers.Conv2D(filters, kernel_size=(1, 8), padding='same', use_bias=False),
             tf.keras.layers.Activation('relu'),
             tf.keras.layers.BatchNormalization()])

        self.se_block = tf.keras.Sequential([tf.keras.layers.GlobalAvgPool2D(),
                                          tf.keras.layers.Dense(filters * 4, activation='relu'),
                                          tf.keras.layers.BatchNormalization(),
                                          tf.keras.layers.Dense(filters, activation='relu'),
                                          tf.keras.layers.BatchNormalization(),
                                          tf.keras.layers.Dense(filters * 4, activation='sigmoid'),
                                          tf.keras.layers.Reshape([1, 1, filters * 4])])

        self.skip_connection = tf.keras.layers.Conv2D(filters * 4, kernel_size=1, padding='same', use_bias=False)

    def call(self, x):
        x_skip = self.skip_connection(x)
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        x = tf.concat([x1, x2, x3, x4], -1)
        x = self.se_block(x) * x

        return x + x_skip


class SqueezeAndExcitation(tf.keras.layers.Layer):
    def __init__(self, d_model, sigmoid=True, **kwargs):
        super(SqueezeAndExcitation, self).__init__(**kwargs)
        self.d_model = d_model

        self.last_activation = 'sigmoid' if sigmoid else 'tanh'

        self.se_block = tf.keras.Sequential([tf.keras.layers.GlobalAvgPool1D(),
                                             tf.keras.layers.Dense(d_model, activation='relu'),
                                             tf.keras.layers.BatchNormalization(),
                                             tf.keras.layers.Dense(d_model // 4, activation='relu'),
                                             tf.keras.layers.BatchNormalization(),
                                             tf.keras.layers.Dense(d_model, self.last_activation),
                                             tf.keras.layers.Reshape([1, d_model])])

    def call(self, x):
        return self.se_block(x) * x


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention implementation for chess position analysis.

    Args:
        d_model: Model dimensionality.
        n_patches: length of sequence.
        n_heads: Number of attention heads.
        **kwargs: Additional arguments for the Layer class.
"""

    def __init__(self, d_model, n_patches, n_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_patches = n_patches

        self.depth = d_model // n_heads
        self.scale = d_model ** 0.5

        self.q_w = tf.keras.layers.Dense(d_model)
        self.k_w = tf.keras.layers.Dense(d_model)
        self.v_w = tf.keras.layers.Dense(d_model)

        self.final_dense = tf.keras.layers.Dense(d_model, activation='relu')

    def split_heads(self, x, batch_size):
        """
        input shape: batch_size x patchs x d_model
        d_model = n_heads x depth
        intermediate shape: batch_size x patches x n_heads x depth
        output shape: batch_size x n_heads x n_patchs x depth
        """
        x = tf.reshape(x, (batch_size, self.n_patches, self.n_heads, self.depth))
        x = tf.transpose(x, [0, 2, 1, 3])
        return x

    def concat_heads(self, x, batch_size):
        """
        input shape = batch_size x n_heads x patches x depth
        intermediate = batch_size x patches x n_heads x depth
        output = batch_size x n_patches x d_model
        """

        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, self.n_patches, self.d_model))
        return x

    def call(self, x):
        batch_size = tf.shape(x)[0]
        q, k, v = self.q_w(x), self.k_w(x), self.v_w(x)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention = tf.matmul(q, k, transpose_b=True) / self.scale
        attention_score = tf.nn.softmax(attention)
        attention_result = tf.matmul(attention_score, v)

        concated_result = self.concat_heads(attention_result, batch_size)

        return self.final_dense(concated_result)


class Encoder(tf.keras.layers.Layer):
    """
        Encoder layer with self-attention and feed-forward network.

        Args:
            d_model: Model dimensionality.
            n_patches: sequence length
            n_heads: Number of attention heads.
            **kwargs: Additional arguments for the Layer class.
"""

    def __init__(self, d_model, n_patches, n_heads, dropout_rate=0.2, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_patches = n_patches

        self.mha = MultiHeadAttention(d_model, n_patches, n_heads)
        self.norm_1 = tf.keras.layers.LayerNormalization()

        self.mlp = tf.keras.Sequential([tf.keras.layers.Dense(d_model, activation='relu'),
                                        tf.keras.layers.LayerNormalization(),
                                        tf.keras.layers.Dropout(dropout_rate),
                                        tf.keras.layers.Dense(d_model, activation='relu'),
                                        tf.keras.layers.LayerNormalization(),
                                        tf.keras.layers.Dropout(dropout_rate)])

        self.norm_2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.mha(x) + x
        x = self.norm_1(x)
        x = self.mlp(x) + x
        x = self.norm_2(x)

        return x


class TransformerXadrezia(tf.keras.Model):
    def __init__(self, n_patches=64, projection_dim=384, n_heads=16, n_encoders=8, n_groups=2, dropout_rate=0.2, **kwargs):
        super(TransformerXadrezia, self).__init__(**kwargs)
        self.n_groups = n_groups

        self.conv = tf.keras.Sequential([Convolution(projection_dim//2), Convolution(projection_dim//2)])
        self.post_conv_reshape = tf.keras.layers.Reshape([n_patches, projection_dim//2])

        self.embedding = tf.keras.layers.Embedding(13, projection_dim//2)
        self.pos_enc = gen_positional_encoding(n_patches, projection_dim)

        self.squeeze_and_excitation = [SqueezeAndExcitation(projection_dim) for _ in range(n_groups)]
        self.encoders = [tf.keras.Sequential() for _ in range(n_groups)]
        for i in range(n_groups):
            for _ in range(n_encoders):
                self.encoders[i].add(Encoder(projection_dim, n_patches, n_heads, dropout_rate))
        #self.se_after_enc = SqueezeAndExcitation(projection_dim*n_groups)

        self.avg = tf.keras.layers.GlobalAveragePooling1D()
        self.probabilities = tf.keras.Sequential([tf.keras.layers.Dense(4098, activation='softmax')])

    def call(self, x):
        tokens = tf.math.argmax(x, axis=-1)
        tokens = tf.reshape(tokens, (-1, 64))
        tokens = tf.cast(tokens, dtype='int32')
        embedded_tokens = self.embedding(tokens)

        x = self.conv(x)
        x = self.post_conv_reshape(x)

        x = tf.concat([embedded_tokens, x], axis=-1) + self.pos_enc

        x = tf.concat([enc(se(x)) for enc, se in zip(self.encoders, self.squeeze_and_excitation)], axis=-1)
        x = self.avg(x)
        x = self.probabilities(x)
        return x
