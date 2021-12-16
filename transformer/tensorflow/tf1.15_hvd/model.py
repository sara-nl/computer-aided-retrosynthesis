import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import horovod.tensorflow as hvd
import pdb

from layers import PositionLayer, MaskLayerLeft, MaskLayerRight, MaskLayerTriangular, SelfLayer, LayerNormalization
from data.preprocessing import gen_left, gen_right


def buildNetwork(n_block, n_self, embedding_size, vocab_size, key_size, n_hidden, opts):
    print("Building network ...")

    # product
    l_in = layers.Input(shape=(None,))
    l_mask = layers.Input(shape=(None,))

    # reagents
    l_dec = layers.Input(shape=(None,))
    l_dmask = layers.Input(shape=(None,))

    # positional encodings for product and reagents, respectively
    l_pos = PositionLayer(embedding_size)(l_mask)
    l_dpos = PositionLayer(embedding_size)(l_dmask)

    l_emask = MaskLayerRight()([l_dmask, l_mask])
    l_right_mask = MaskLayerTriangular()(l_dmask)
    l_left_mask = MaskLayerLeft()(l_mask)

    # encoder
    l_voc = layers.Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=None)

    l_embed = layers.Add()([l_voc(l_in), l_pos])
    l_embed = layers.Dropout(rate=0.1)(l_embed)

    for layer in range(n_block):
        # self attention
        l_o = [SelfLayer(embedding_size, key_size)([l_embed, l_embed, l_embed, l_left_mask]) for i in range(n_self)]

        l_con = layers.Concatenate()(l_o)
        l_dense = layers.TimeDistributed(layers.Dense(embedding_size))(l_con)
        l_drop = layers.Dropout(rate=0.1)(l_dense)
        l_add = layers.Add()([l_drop, l_embed])
        l_att = LayerNormalization()(l_add)

        # position-wise
        l_c1 = layers.Conv1D(n_hidden, 1, activation='relu')(l_att)
        l_c2 = layers.Conv1D(embedding_size, 1)(l_c1)
        l_drop = layers.Dropout(rate=0.1)(l_c2)
        l_ff = layers.Add()([l_att, l_drop])
        l_embed = LayerNormalization()(l_ff)

        # bottleneck
    l_encoder = l_embed

    l_embed = layers.Add()([l_voc(l_dec), l_dpos])
    l_embed = layers.Dropout(rate=0.1)(l_embed)

    for layer in range(n_block):
        # self attention
        l_o = [SelfLayer(embedding_size, key_size)([l_embed, l_embed, l_embed, l_right_mask]) for i in range(n_self)]

        l_con = layers.Concatenate()(l_o)
        l_dense = layers.TimeDistributed(layers.Dense(embedding_size))(l_con)
        l_drop = layers.Dropout(rate=0.1)(l_dense)
        l_add = layers.Add()([l_drop, l_embed])
        l_att = LayerNormalization()(l_add)

        # attention to the encoder
        l_o = [SelfLayer(embedding_size, key_size)([l_att, l_encoder, l_encoder, l_emask]) for i in range(n_self)]
        l_con = layers.Concatenate()(l_o)
        l_dense = layers.TimeDistributed(layers.Dense(embedding_size))(l_con)
        l_drop = layers.Dropout(rate=0.1)(l_dense)
        l_add = layers.Add()([l_drop, l_att])
        l_att = LayerNormalization()(l_add)

        # position-wise
        l_c1 = layers.Conv1D(n_hidden, 1, activation='relu')(l_att)
        l_c2 = layers.Conv1D(embedding_size, 1)(l_c1)
        l_drop = layers.Dropout(rate=0.1)(l_c2)
        l_ff = layers.Add()([l_att, l_drop])
        l_embed = LayerNormalization()(l_ff)

    l_out = layers.TimeDistributed(layers.Dense(vocab_size,
                                                use_bias=False))(l_embed)

    mdl = tf.keras.Model([l_in, l_mask, l_dec, l_dmask], l_out)

    def masked_loss(y_true, y_pred):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
        mask = tf.cast(tf.not_equal(tf.reduce_sum(y_true, -1), 0), 'float32')
        loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
        loss = K.mean(loss)
        return loss

    def masked_acc(y_true, y_pred):
        mask = tf.cast(tf.not_equal(tf.reduce_sum(y_true, -1), 0), 'float32')
        eq = K.cast(K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), 'float32')
        eq = tf.reduce_sum(eq * mask, -1) / tf.reduce_sum(mask, -1)
        eq = K.mean(eq)
        return eq

    if opts.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0, epsilon=opts.eps)
    elif opts.optimizer == 'adamax':
        optimizer = tf.keras.optimizers.Adamax(learning_rate=0.0)
    elif opts.optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.0, nesterov=True, momentum=0.9)
    else:
        raise NotImplementedError

    if opts.horovod:
        optimizer = hvd.DistributedOptimizer(optimizer)

    mdl.compile(optimizer=optimizer, loss=masked_loss, metrics=['accuracy', masked_acc])

    # Divide the graph for faster execution. First, we calculate encoder's values.
    # Then we use encoder's values and the product mask as additional decoder's input.
    def mdl_encoder(product):
        v = gen_left([product], opts)
        enc = l_encoder.eval(feed_dict={l_in: v[0], l_mask: v[1], l_pos: v[2]})
        return enc, v[1]

        # And the decoder

    def mdl_decoder(res, product_encoded, product_mask, T=1.0):

        v = gen_right([res], opts)
        d = l_out.eval(feed_dict={l_encoder: product_encoded, l_dec: v[0],
                                  l_dmask: v[1], l_mask: product_mask, l_dpos: v[2]})
        prob = d[0, len(res), :] / T
        prob = np.exp(prob) / np.sum(np.exp(prob))
        return prob

    return mdl, mdl_encoder, mdl_decoder
