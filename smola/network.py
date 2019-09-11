import tensorflow as tf
import numpy as np
from scipy import integrate
import datetime
import math


class Network:
    def __init__(self, setting):
        self.n_hidden = setting["n_hidden"]
        self.len_seq = setting["len_seq"]
        self.learning_rate = setting["learning_rate"]
        self.n_epoch = setting["n_epoch"]
        self.batch_size = setting["batch_size"]
        self.log_path = setting["log_path"]
        self.g_size = setting["g_size"]
        self.max_predict = setting["max_predict"]
        self.delta_t = setting["delta_t"]
        self.eps = 1e-20
        self.sur_loss = 0
        self.rec_loss = 0

    def soft_relu(self, x):
        return tf.log(1 + tf.exp(x))

    def survival_loss(self):
        '''
        the surrvival loss:
        -sum_{user}(P(Su|T, u)) = loss1 + loss2  + loss3
        loss1 = sum_{user}(sum_{i}(sum_{e(i-1) <= t <= min(b(i) , max_prdict}(lambda(t) * delta_t)))
        loss2 = -sum_{user}(sum_{i}(log(lambda(bi))))
        '''
        loss1 = tf.reduce_sum(tf.reduce_sum(self.big_lambda[:, :-1] * self.g_masked * self.delta_t, axis=2) * self.mask[:, 1:])
        ''' 
        b_e is the bi - e(i-1) matrix in one hot shape
        sum(b_e * biglambda) = biglambda(bi - e(i-1)) = lambda(bi)
        '''
        b_e = tf.one_hot(tf.cast((self.b[:, 1:] - self.e[:, :-1]) / self.delta_t, tf.int32), self.max_predict)
        loss2 = -tf.reduce_sum(tf.log(tf.reduce_sum(self.big_lambda[:, :-1] * b_e, axis=-1) + self.eps) * self.mask[:, 1:])
        '''in the paper we had loss3 but now we don't know the T so we won't add that'''
        #     loss3 = tf.reduce_sum(tf.gather(big_lambda, length - 1) * e_masked * delta_t)
        self.sur_loss = loss1 + loss2

    def recurrent_loss(self, a_pred):
        '''
        the recurrent loss:
        -sum_{user}(sum_{i}(a * log(a_pred) - log(a!) - a_pred))
        wich didn't include log(a!) because it's constant
        '''
        self.rec_loss = -tf.reduce_sum((self.a * tf.log(a_pred + self.eps) - a_pred) * self.mask)

    def create_network(self):
        ''' the basic rnn cell '''
        lstm = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

        '''
        d: a placeholder for date in vector with length d_size
        g: a placeholder for gap in vector with length g_size
        b & e & a : placeholder for begin and end and action(mark)
        g_masked: the gap between ends and begin mask shape
        '''
        # d = tf.placeholder(dtype=tf.float32, shape=[None, len_seq, d_size])
        self.b = tf.placeholder(dtype=tf.float32, shape=[None, self.len_seq])
        self.e = tf.placeholder(dtype=tf.float32, shape=[None, self.len_seq])
        self.g = tf.placeholder(dtype=tf.float32, shape=[None, self.len_seq - 1, self.g_size])
        self.g_masked = tf.placeholder(dtype=tf.float32, shape=[None, self.len_seq - 1, self.max_predict])
        self.a = tf.placeholder(dtype=tf.float32, shape=[None, self.len_seq])
        self.mask = tf.placeholder(dtype=tf.float32, shape=[None, self.len_seq])

        b_shape = tf.shape(self.b)

        inputs = tf.concat([self.g, tf.zeros([b_shape[0], 1, self.g_size])], axis=1)

        output, state = tf.nn.dynamic_rnn(lstm, inputs, dtype=tf.float32)

        '''the next 5 line is putting a dense layer with mac_predict +1 size'''
        w_lambda = tf.Variable(tf.random_normal([self.n_hidden, self.max_predict + 1], stddev=0.05))
        b_lambda = tf.Variable(tf.random_normal([self.max_predict + 1], stddev=0.05))

        output_reshaped = tf.reshape(output, [-1, self.n_hidden])
        output_modified = tf.matmul(output_reshaped, w_lambda) + b_lambda

        ''' the out put with prefered shape'''
        last_output = tf.reshape(output_modified, [b_shape[0], self.len_seq, self.max_predict + 1])

        self.big_lambda = self.soft_relu(last_output[:, :, :-1] )
        a_pred = self.soft_relu(last_output[:, :, -1])

        self.accuracy = tf.reduce_mean((a_pred - self.a) * (a_pred - self.a) * self.mask)

        '''as the paper said alpha = 0.5 has a good result'''
        alpha = tf.Variable(0.5, trainable=False)

        self.survival_loss()
        self.recurrent_loss(a_pred)
        self.loss = alpha * self.sur_loss + (1 - alpha) * self.rec_loss
#         self.loss = self.sur_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        print("network created successfuly!")

    def train_network(self, sess, train_e, train_b, train_g, train_mark, X_mask_train, train_g_masked, write_summary = False):
        '''
        :param sess: tf.Session
        :param train_e: end of sessions normalized
        :param train_b: begin of sessions normalized
        :param train_g: end{i} - begin{i-1} in vector shape
        :param train_g_masked: the gap between ends and begin mask shape
        :param train_mark: sessions mark
        :param X_mask_train: train mask
        :param write_summary: this indicates that whether we want to save the summaries in tensorboard or not
        :return:
        '''
        #     prepare the summary writers
        if write_summary:
            dt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            log = self.log_path + "report-" + dt
            print(log)
            file_writer = tf.summary.FileWriter(log)

        init = tf.global_variables_initializer()

        sess.run(init)
        for i in range(self.n_epoch):
            total_loss = 0
            total_mse = 0
            total_accuracy = 0

            n_batch = math.ceil(len(train_e) / self.batch_size)
            for j in range(n_batch):
#                 print(j)
                end = train_e[j * self.batch_size:(j + 1) * self.batch_size, :, 0]
                begin =  train_b[j * self.batch_size:(j + 1) * self.batch_size, :,0]
                action = train_mark[j * self.batch_size:(j + 1) * self.batch_size, :, 0]
                x_mask = X_mask_train[j * self.batch_size:(j + 1) * self.batch_size]
                gap = train_g[j * self.batch_size:(j + 1) * self.batch_size]
                gap_masked = train_g_masked[j * self.batch_size:(j + 1) * self.batch_size]
                
                l = np.sum(x_mask, axis=1).astype(int)
                if not write_summary:
                    train_loss, acc, _, b_l = sess.run(
                        [self.loss, self.accuracy, self.optimizer, self.big_lambda]
                        , feed_dict={self.b: begin, self.e: end, self.a: action, self.g: gap, self.mask: x_mask, self.g_masked: gap_masked})
                else:
                    train_loss, acc, _, b_l, summary = sess.run(
                        [self.loss, self.accuracy, self.optimizer, self.big_lambda, self.merged]
                        , feed_dict={self.b: begin, self.e: end, self.a: action, self.g: gap, self.mask: x_mask,
                                     self.g_masked: gap_masked})
                    file_writer.add_summary(summary, i * n_batch + j)

#                 mse_gap = np.array(begin[np.arange(len(begin)), l - 1] - end[np.arange(len(end)), l - 2])
#                 b_l = np.array(b_l[np.arange(len(b_l)), l - 2])
                
#                 mse = self.calculate_mse(b_l, mse_gap)
# #                 print("train MSE: {}".format(mse))
#                 total_mse += mse
# #                 print(np.shape(mse_gap))

# #                 if i == self.n_epoch - 1:
# #                     '''
# #                     computing the se
# #                     for time consuming issues we only compute se for last epoch
# #                     '''
# #                     mse = self.calculate_mse(b_l, mse_gap)
# #                     total_mse += mse

#                 total_accuracy += acc
#                 total_loss += train_loss


#             print("epoch #{}, train_loss = {}, train_mse = {}, accuracy = {}".format(
#                 i,
#                 total_loss / len(train_e) / self.len_seq,
#                 total_mse/n_batch,
#                 float(total_accuracy) / n_batch,
#             ))
#             if i == self.n_epoch - 1:
#                 print("sum_mse = {}".format(total_mse / n_batch))
        
        print("training finished!")

    def test_network(self, sess, test_e, test_b, test_mark, X_mask_test, test_g, test_g_masked):
        time_mse = []
        time_mae = []
        loss = []
        end, begin = test_e[:, :, 0], test_b[:, :, 0]
        action = test_mark[:, :, 0]
        x_mask = X_mask_test
        gap, gap_masked = test_g, test_g_masked
        l = np.sum(x_mask, axis=1).astype(int)

        test_loss, acc, b_l = sess.run([self.loss, self.accuracy, self.big_lambda]
                                       , feed_dict={self.b: begin, self.e: end, self.a: action, self.g: gap, self.mask: x_mask,
                                                    self.g_masked: gap_masked})

        mse_gap = np.array(begin[np.arange(len(begin)), l - 1] - end[np.arange(len(end)), l - 2])
        b_l = np.array(b_l[np.arange(len(b_l)), l - 2])
        mse = self.calculate_mse(b_l, mse_gap)
        mae = self.calculate_mae(b_l, mse_gap)

        time_mse.append(mse)
        time_mae.append(mae)
        loss.append(loss)
        res_dict = {}
        res_dict['Time MSE'] = time_mse
        res_dict['Time MAE'] = time_mae
        res_dict['Loss'] = loss
        
        print("test_loss = {} ,test_mse = {},test_mae={}, accuracy = {}".format(
            test_loss / len(test_e) / self.len_seq,
            mse,mae,
            float(acc)))
        
        print("test finished!")
        return res_dict

    def integrate_function(self, b_l):
        '''
        :param b_l: bif_lambda
        :return: integration of p(t|u) * t from 0 to infinity
        '''
        def func(t):
            '''
            p(t|u) = lambda(t) * exp(- integrate_{b(i-1)} to {t} lambda(s) delta_s) =
            big_lambda(t / delta_t) * exp(-sum{ b(i-1) <= s <= t}(big_lambda(s / delta_t) * delta_t)
            :param t:
            :return: t * p(t|u)
            '''
            index = int(t / self.delta_t)
            return t * b_l[index] * np.exp(
                -np.sum(b_l[:index] * self.delta_t)
                         )

        result = integrate.quad(func, 0, self.max_predict * self.delta_t)
        return result[0] - result[1]

    def calculate_mse(self,b_l, begins):
        '''
        :param b_l:
        :param begins: real begins
        :return: sum of mse of predicted begins and real begins
        '''
        mse = 0
        number = 0
        for i, real in enumerate(begins):
            pred = self.integrate_function(b_l[i])
            mse += (pred - real) * (pred - real)
            number += 1
        return mse / number
    def calculate_mae(self,b_l, begins):
        '''
        :param b_l:
        :param begins: real begins
        :return: sum of mse of predicted begins and real begins
        '''
        mse = 0
        number = 0
        for i, real in enumerate(begins):
            pred = self.integrate_function(b_l[i])
            mse += np.abs(pred - real)
            number += 1
        return mse / number