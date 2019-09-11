from scipy import integrate
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import math

class Network:

    def __init__(self, setting):
        self.n_hidden = setting["n_hidden"]
        self.len_seq = setting["len_seq"]
        self.y_size = setting["y_size"]
        self.learning_rate = setting["learning_rate"]
        self.n_epoch = setting["n_epoch"]
        self.batch_size = setting["batch_size"]
        self.log_path = setting["log_path"]

        self.eps = 1e-20

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    def create_network(self):
        ''' the basic rnn cell '''
        lstm = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

        '''
        the place holder for inputs:
        y is y_numeric in one hot shape
        '''
        self.y = tf.placeholder(dtype=tf.float64, shape=[None, self.len_seq, self.y_size])
        self.y_numeric = tf.placeholder(dtype=tf.int64, shape=[None, self.len_seq])
        self.t = tf.placeholder(dtype=tf.float64, shape=[None, self.len_seq, 1])
        self.mask = tf.placeholder(dtype=tf.float64, shape=[None, self.len_seq])
        length = tf.reduce_sum(self.mask, axis=1)

        '''d is tj+1 - tj matrix'''
        d = self.t[:, 1:self.len_seq, :] - self.t[:, 0:self.len_seq - 1, :]

        '''lstm inputs'''
        inputs = tf.concat(axis=2, values=(self.y, self.t))

        '''these are the marker generator variables'''
        self.Vm = [tf.Variable(tf.random_normal([self.n_hidden, 1], dtype=tf.float64), dtype=tf.float64) for i in range(self.y_size)]
        self.Bm = [tf.Variable(tf.random_normal([1], dtype=tf.float64), dtype=tf.float64) for i in range(self.y_size)]
        '''these are the likelihood variables'''
        self.Vf = tf.Variable(tf.random_normal([self.n_hidden, 1], dtype=tf.float64), dtype=tf.float64)
        self.Wf = tf.Variable(tf.random_normal([1], dtype=tf.float64), dtype=tf.float64)
        self.Bf = tf.Variable(tf.random_normal([1], dtype=tf.float64), dtype=tf.float64)

        '''running the lstm cell'''
        output, state = tf.nn.dynamic_rnn(lstm, inputs, sequence_length=length, dtype=tf.float64)

        '''
        converting output and y and d and mask to the list shape for get_loss function
        '''
        self.output_list = tf.transpose(output, [1, 0, 2])
        self.output_list = tf.reshape(self.output_list, [-1, self.n_hidden])
        self.output_list = tf.split(axis=0, num_or_size_splits=self.len_seq, value=self.output_list)
        y_list = tf.transpose(self.y_numeric)
        d_list = tf.transpose(d, [1, 0, 2])
        mask_reshaped = tf.transpose(self.mask)

        self.get_loss(d_list, y_list, self.output_list, mask_reshaped)

        ''' tensorboard summaries '''
#         tf.summary.scalar('log_flikelihood', tf.reduce_sum(self.gen_loss))
#         tf.summary.scalar('marker_generation', tf.reduce_sum(self.likeli_loss))
#         tf.summary.scalar('loss', self.loss)
#         tf.summary.scalar('W flikelihood', tf.reduce_sum(self.Wf))
#         tf.summary.scalar('B flikelihood', tf.reduce_sum(self.Bf))
#         tf.summary.scalar('V * h flikelihood', tf.reduce_sum(self.sum_h_v))
#         tf.summary.histogram('V flikelihood', self.Vf)
#         self.merged = tf.summary.merge_all()

        ''' the optimizer'''
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        
        print("network successfully created!")


    def marker_generation(self, y, h):
        '''
        :param y: the marks in list shape
        :param h: hiddens list
        :return: sum of P(yj_i+1| hj) for every i and j
        '''

        '''exp_offset is for avoiding overflow in tf.exp'''
        exp_offset = 200
        '''the len_seq * batch_size matrix that is denominator witch is the sum of h * Vm + Bm'''
        denominator = tf.reduce_sum(tf.exp(tf.tensordot(h, self.Vm, [[2], [1]], name = "tensor_dot1")[:, :, :, 0]
                                           + tf.reshape(self.Bm, [-1]) - exp_offset), axis=2)

        '''y_max is the len_seq * batch_size that maximizes p(yj_i+1| hj) wich is max of h * Vm + Bm'''
        y_max = tf.argmax(tf.exp(
            tf.tensordot(h, self.Vm, [[2], [1]], name = "tensor_dot2")[:, :, :, 0] + 
            tf.reshape(self.Bm, [-1]) - exp_offset),
                          axis=2)

        '''the len_seq * batch_size matrix that is nominator'''
        '''tf.gather finds the Vm with indices of y-1'''
        nominator = tf.exp(tf.reduce_sum(h * tf.gather(self.Vm, indices=y - 1)[:, :, :, 0], axis=2)
                           + tf.gather(self.Bm, indices=y - 1)[:, :, 0] - exp_offset)

        '''the last result'''
        self.gen_loss = nominator / denominator
        '''accuracy of anticapating the next mark right which is number of ys that are equal to y_max'''
        self.sum_acc = tf.reduce_sum(tf.cast(tf.equal(y_max, y), tf.int32)) 

    def f_likelihood(self, d, h):
        '''
        :param d: gaps in list shape
        :param h: the hiddens list
        :return: sum of f(hj|yj_i+1) for every i and j
        '''
        ''' h * Vf'''
        h_v = tf.tensordot(h, self.Vf, [[2], [0]])
        ''' sum_h_v is saved for summaries for better visualisation'''
        self.sum_h_v = tf.reduce_sum(h_v)
        ''' the loss likelihood function as mentioned in the paper'''
        self.likeli_loss = h_v + d * self.Wf + self.Bf + 1 / self.Wf * tf.exp(h_v + self.Bf) - 1 / self.Wf * tf.exp(h_v + d * self.Wf + self.Bf)



    def get_loss(self, d, y, h, mask):
        h = h[:-1]
        y = y[1:]
        self.f_likelihood(d,h)
        self.marker_generation(y, h)
        '''multiply loss to mask'''
        self.loss = -tf.reduce_sum((self.gen_loss+ self.likeli_loss[:,:,0]) * mask[1:], axis=0)
        '''find the loss mean!'''
        self.loss = self.loss / tf.cast(tf.reduce_sum(mask, axis=0), dtype=tf.float64)
        self.loss = tf.reduce_mean(self.loss)

    def integral_f_likelihood(self,h_v, w, b):
        '''
        :param h_v: h * Vf
        :param w: Wf
        :param b: Bf
        :return: the integration of flikelihood * d from 0 to infinity wich is the predicted gap time
        '''
        '''w_m is for avoiding overflow'''
        w_m = 0
        if np.absolute(w) > self.eps:
            w_m = w
        else:
            w_m = self.eps

        def integrate_func(d):
            return d * np.exp(
                h_v +
                d * w + b +
                1 / w_m * np.exp(h_v + b) -
                1 / w_m * np.exp(h_v + d * w + b)
            )

        result = integrate.quad(integrate_func, 0, np.inf)
        return result[0] - result[1]

    def train_network(self , sess, train_time, train_mark_reshaped, train_mark, X_mask_train, write_summary = False):
        '''

        :param sess: tf.Session
        :param train_time: sessions normalized starting time
        :param train_mark_reshaped: one hot of sessions mark
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

        sess.run(self.init)
        for e in range(self.n_epoch):
            total_loss = 0
            total_acc = 0
            se = 0
            sum_se = 0
            n_batch = math.ceil(len(train_time) / self.batch_size )
            print(n_batch)
            for s in range(1):
                t1 = train_time[s * self.batch_size:(s + 1) * self.batch_size]
                y1 = train_mark_reshaped[s * self.batch_size:(s + 1) * self.batch_size]
                y2 = train_mark[s * self.batch_size:(s + 1) * self.batch_size]
                x_mask = X_mask_train[s * self.batch_size:(s + 1) * self.batch_size]

                if not write_summary:

                    train_loss, acc, v_train, w_train, b_train, o_train, _ = sess.run(
                        [self.loss, self.sum_acc, self.Vf, self.Wf, self.Bf, self.output_list, self.optimizer]
                        , feed_dict={self.t: t1, self.y: y1, self.y_numeric: y2, self.mask: x_mask})
                else:
                    train_loss, acc, v_train, w_train, b_train, o_train, _, summary = sess.run(
                        [self.loss, self.sum_acc, self.Vf, self.Wf, self.Bf, self.output_list, self.optimizer, self.merged]
                        , feed_dict={self.t: t1, self.y: y1, self.y_numeric: y2, self.mask: x_mask})

                    file_writer.add_summary(summary, e * n_batch + s)

#                 if e == self.n_epoch - 1:
#                     '''
#                     computing the se
#                     for time consuming issues we only compute se for last epoch
#                     '''
#                     for j in range(self.len_seq -1):
#                         for i in range(self.batch_size):
#                             h_v = np.matmul(np.reshape(o_train[j][i,:], [1,-1]),v_train)[0,0]
#                             d_max = self.integral_f_likelihood(h_v, w_train[0],b_train[0])

#                             se += np.square(t1[i][j + 1][0] - t1[i][j][0] - d_max) * x_mask[i, j]

#                     sum_se += se / np.sum(x_mask)
                    
#                 total_loss += train_loss
#                 total_acc += acc

#             print("epoch #{}, train_loss = {:.6f} , train_acc = {:.6f} "
#                   .format(e, total_loss / n_batch, float(total_acc) / n_batch / self.batch_size /(self.len_seq - 1)))
#             if e == self.n_epoch - 1:
#                 print("sum_se is: {}".format(sum_se))
                
        print("training finnished!")
    
    def train_network2(self , sess, train_time, train_mark_reshaped, train_mark, X_mask_train, write_summary = False):
        '''

        :param sess: tf.Session
        :param train_time: sessions normalized starting time
        :param train_mark_reshaped: one hot of sessions mark
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

        sess.run(self.init)
        for e in range(self.n_epoch):
            total_loss = 0
            total_acc = 0
            se = 0
            sum_se = 0
            n_batch = len(train_time) // self.batch_size
            
            for s in range(n_batch):
                t1 = train_time[s * self.batch_size:(s + 1) * self.batch_size]
                y1 = train_mark_reshaped[s * self.batch_size:(s + 1) * self.batch_size]
                y2 = train_mark[s * self.batch_size:(s + 1) * self.batch_size]
                x_mask = X_mask_train[s * self.batch_size:(s + 1) * self.batch_size]

                if not write_summary:

                    train_loss, acc, v_train, w_train, b_train, o_train, _ = sess.run(
                        [self.loss, self.sum_acc, self.Vf, self.Wf, self.Bf, self.output_list, self.optimizer]
                        , feed_dict={self.t: t1, self.y: y1, self.y_numeric: y2, self.mask: x_mask})
                else:
                    train_loss, acc, v_train, w_train, b_train, o_train, _, summary = sess.run(
                        [self.loss, self.sum_acc, self.Vf, self.Wf, self.Bf, self.output_list, self.optimizer, self.merged]
                        , feed_dict={self.t: t1, self.y: y1, self.y_numeric: y2, self.mask: x_mask})

                    file_writer.add_summary(summary, e * n_batch + s)

                if e == self.n_epoch - 1:
                    '''
                    computing the se
                    for time consuming issues we only compute se for last epoch
                    '''
                    for j in range(self.len_seq -1):
                        for i in range(self.batch_size):
                            h_v = np.matmul(np.reshape(o_train[j][i,:], [1,-1]),v_train)[0,0]
                            d_max = self.integral_f_likelihood(h_v, w_train[0],b_train[0])

                            se += np.square(t1[i][j + 1][0] - t1[i][j][0] - d_max) * x_mask[i, j]

                    sum_se += se / np.sum(x_mask)
                    
                total_loss += train_loss
                total_acc += acc

            print("epoch #{}, train_loss = {:.6f} , train_acc = {:.6f} "
                  .format(e, total_loss / n_batch, float(total_acc) / n_batch / self.batch_size /(self.len_seq - 1)))
            if e == self.n_epoch - 1:
                print("sum_se is: {}".format(sum_se))
                
        print("training finnished!")

    def test_network(self,sess, test_time, test_mark_reshaped, test_mark, X_mask_test):
        time_mse = {"Train":[], "Test":[]}
        time_mae = {"Train":[], "Test":[]}
        loss = {"Train":[], "Test":[]}
        
        total_acc = 0
        total_loss = 0
        sum_se = 0
        sum_ae = 0
        sum_re = 0
        n_batch = len(test_time) // self.batch_size
        for s in range(n_batch):
            t1 = test_time[s * self.batch_size:(s + 1) * self.batch_size]
            y1= test_mark_reshaped[s * self.batch_size:(s + 1) * self.batch_size] 
            y2 = test_mark[s * self.batch_size:(s + 1) * self.batch_size]
            x_mask = X_mask_test[s * self.batch_size:(s + 1) * self.batch_size]
            se = 0
            ae = 0
            re = 0

            test_loss, acc, v_test, w_test, b_test, o_test = \
                sess.run([self.loss, self.sum_acc, self.Vf, self.Wf, self.Bf, self.output_list]
                                            , feed_dict={self.t: t1, self.y: y1, self.y_numeric: y2,self.mask: x_mask})
            total_loss += test_loss
            total_acc += acc
            for j in range(self.len_seq - 1):
                for i in range(self.batch_size):
                    d_max = self.integral_f_likelihood(np.matmul(np.reshape(o_test[j][i, :], [1, -1]), v_test)[0, 0],
                                                  w_test[0], b_test[0])
                    se += np.square(t1[i][j + 1][0] - t1[i][j][0] - d_max) * x_mask[i, j]
                    ae += np.abs(t1[i][j + 1][0] - t1[i][j][0] - d_max) * x_mask[i, j]
                    

            sum_se += se / np.sum(x_mask)
            sum_ae += ae / np.sum(x_mask)
        
        time_mse["Test"].append(sum_se / n_batch)
        time_mae['Test'].append(sum_ae / n_batch)
        loss['Test'].append(total_loss / n_batch)
        
        result_dict = {}
        result_dict["Time MSE"] = time_mse
        result_dict["Time MAE"] = time_mae
        result_dict["Loss"] = loss

        
        print("test_loss = {:.6f} , test_mse = {},test_acc = {:.6f} "
              .format(total_loss / n_batch, sum_se / n_batch, float(total_acc) / n_batch/ self.batch_size /(self.len_seq - 1)))
        print("mae: {}".format(sum_ae / n_batch))
        
        print("test finished!")
        return result_dict


