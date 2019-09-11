import tensorflow as tf
import numpy as np


# This is the recurrent cell  designed to predict churn
class ChurnRNNCell(tf.contrib.rnn.RNNCell):
    def __init__(self, n_h, n_phi_prior, n_phi_encoder, n_phi_z_decoder, n_phi_h_decoder):
        self.n_h = n_h
        self.n_phi_prior = n_phi_prior
        self.n_phi_encoder = n_phi_encoder
        self.n_phi_z_decoder = n_phi_z_decoder
        self.n_phi_h_decoder = n_phi_h_decoder
        self.lstm = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)
        self.n_sampling = 90

    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return (1, 1, 1, 1, 1, self.n_h, 1, 1, 1, 1)

    def zero_state(self, batch_size, dtype):
        c = tf.zeros(shape=[batch_size, self.n_h], dtype=dtype)
        h = tf.zeros(shape=[batch_size, self.n_h], dtype=dtype)
        state_tupple = tf.contrib.rnn.LSTMStateTuple(c, h)
        return state_tupple

    def predict_x(self, mu, sigma, phi_h_decoder, n_sampling):
        x_predicted = 0
        for i in range(n_sampling):
            epsilon_sampling = tf.random_normal(tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
            z_sampling = tf.multiply(epsilon_sampling, sigma) + mu
            with tf.variable_scope("Decoder"):
                with tf.variable_scope("Phi_z"):
                    phi_z_decoder_sampling = tf.layers.dense(z_sampling,
                                                             self.n_phi_z_decoder,
                                                             name='layer1',
                                                             activation=tf.nn.relu,
                                                             reuse=True)
                with tf.variable_scope("Lambda_x"):
                    lmbda_sampling = tf.layers.dense(tf.concat([phi_h_decoder, phi_z_decoder_sampling], axis=1),
                                                     1,
                                                     name='layer1',
                                                     activation=tf.nn.softplus,
                                                     reuse=True)
            with tf.variable_scope("X"):
                x_predicted += tf.random_gamma(alpha=1, beta=lmbda_sampling, dtype=tf.float32, shape=[])
        x_predicted = x_predicted / n_sampling
        return x_predicted

    def predict_f(self, mu, sigma, phi_h_decoder, n_sampling):
        f_predicted = 0
        for i in range(n_sampling):
            epsilon_sampling = tf.random_normal(tf.shape(mu), mean=0.0, stddev=1.0, dtype=tf.float32)
            z_sampling = tf.multiply(epsilon_sampling, sigma) + mu
            with tf.variable_scope("Decoder"):
                with tf.variable_scope("Phi_z"):
                    phi_z_decoder_sampling = tf.layers.dense(z_sampling,
                                                             self.n_phi_z_decoder,
                                                             name='layer1',
                                                             activation=tf.nn.relu,
                                                             reuse=True)
                with tf.variable_scope("Lambda_f"):
                    lmbda_sampling = tf.layers.dense(tf.concat([phi_h_decoder, phi_z_decoder_sampling], axis=1),
                                                     1,
                                                     name='layer1',
                                                     activation=tf.nn.softplus,
                                                     reuse=True)
            with tf.variable_scope("F"):
                f_predicted += tf.random_poisson(lmbda_sampling, dtype=tf.float32, shape=[])
        f_predicted = f_predicted / n_sampling
        return f_predicted

    def __call__(self, xfz, state, scope=None):
        c, h = state
        # shape of xfz is [batch_size * (2 + extra_feature_dim)]
        x = xfz[:, 0:1]
        f = xfz[:, 1:2]
        xhf = tf.concat([x, h, f], axis=1, name="xhf")
        with tf.variable_scope("Encoder"):
            with tf.variable_scope("Hidden"):
                phi_encoder = tf.layers.dense(xhf, self.n_phi_encoder, activation=tf.nn.relu, name='layer1')
            with tf.variable_scope("Mu"):
                mu_encoder = tf.layers.dense(phi_encoder, 1, name='layer1')
            with tf.variable_scope("Sigma"):
                sigma_encoder = tf.layers.dense(phi_encoder, 1, activation=tf.nn.softplus, name='layer1')

        with tf.variable_scope("Prior"):
            with tf.variable_scope("Hidden"):
                phi_prior = tf.layers.dense(h, self.n_phi_prior, activation=tf.nn.relu, name='layer1')
            with tf.variable_scope("Mu"):
                mu_prior = tf.layers.dense(phi_prior, 1, name='layer1')
            with tf.variable_scope("Sigma"):
                sigma_prior = tf.layers.dense(phi_prior, 1, activation=tf.nn.softplus, name='layer1')

        epsilon = tf.random_normal(tf.shape(mu_encoder), mean=0.0, stddev=1.0, dtype=tf.float32)
        z = tf.multiply(epsilon, sigma_encoder) + mu_encoder

        with tf.variable_scope("Decoder"):
            with tf.variable_scope("Phi_h"):
                phi_h_decoder = tf.layers.dense(h, self.n_phi_h_decoder, activation=tf.nn.relu, name='layer1')
            with tf.variable_scope("Phi_z"):
                phi_z_decoder = tf.layers.dense(z, self.n_phi_z_decoder, activation=tf.nn.relu, name='layer1')
            with tf.variable_scope("Lambda_x"):
                lmbda_x = tf.layers.dense(tf.concat(values=(phi_h_decoder, phi_z_decoder), axis=1),
                                          1, activation=tf.nn.softplus, name='layer1')
            with tf.variable_scope("Lambda_f"):
                lmbda_f = tf.layers.dense(tf.concat(values=(phi_h_decoder, phi_z_decoder), axis=1),
                                          1, activation=tf.nn.softplus, name='layer1')
        x_predicted = self.predict_x(mu_encoder, sigma_encoder, phi_h_decoder, self.n_sampling)
        f_predicted = self.predict_f(mu_encoder, sigma_encoder, phi_h_decoder, self.n_sampling)
        # TODO        f_predicted = 0
        xfz_withz = tf.concat([x, f, z], axis=1)
        lstm_state = tf.contrib.rnn.LSTMStateTuple(c, h)
        output, state_new = self.lstm(xfz_withz, lstm_state)  # return state or state_new??
        return (lmbda_x, lmbda_f, z, mu_encoder, sigma_encoder, h, mu_prior, sigma_prior, x_predicted,
                f_predicted), state_new
