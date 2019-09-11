import tensorflow as tf
import numpy as np
from ChurnRNNCell import ChurnRNNCell

# gpu_options = tf.GPUOptions(allow_growth=True)
# config = tf.ConfigProto(gpu_options=gpu_options)


class Network:
    def __str__(self):
        return'\tn_h: {0},\n' \
            '\tn_epoch: {1},\n' \
            '\tlearning_rate:{2},\n' \
            '\tlen_seq: {3},\n' \
            '\tmu0: {4},\n' \
            '\tsigma0: {5},\n' \
            '\tepsilon: {6},\n' \
            '\tuser_prioir: {7},\n' \
            '\tn_phi_prior: {8},\n' \
            '\tn_phi_encoder: {9},\n' \
            '\tn_phi_z_decoder: {10},\n' \
            '\tn_phi_h_decoder: {11},\n'.format(self.n_h,
                                              self.n_epoch,
                                              self.learning_rate,
                                              self.len_seq,
                                              self.mu0,
                                              self.sigma0,
                                              self.epsilon,
                                              self.use_prior,
                                              self.n_phi_prior,
                                              self.n_phi_encoder,
                                              self.n_phi_z_decoder,
                                              self.n_phi_h_decoder)

    def __init__(self, setting):
        self.mu0 = setting['mu0']
        self.sigma0 = setting['sigma0']
        self.epsilon = setting['epsilon']
        self.use_prior = setting['use_prior']
        self.len_seq = setting['len_seq']
        self.n_h = setting['n_h']
        self.n_phi_prior = setting['n_phi_prior']
        self.n_phi_encoder = setting['n_phi_encoder']
        self.n_phi_z_decoder = setting['n_phi_z_decoder']
        self.n_phi_h_decoder = setting['n_phi_h_decoder']
        self.n_epoch = setting['n_epoch']
        self.learning_rate = setting['learning_rate']

    def __call__(self, data, sess, run):
        '''Placeholders for data'''
        X = tf.placeholder(dtype=tf.float32, shape=[None, self.len_seq, 1], name="X")
        F = tf.placeholder(dtype=tf.float32, shape=[None, self.len_seq, 1], name="F")
        Z = tf.zeros(shape=[tf.shape(X)[0], self.len_seq, 1], name="Z")
        mask = tf.placeholder(dtype=tf.int32, shape=[None, self.len_seq])
        YT = tf.placeholder(dtype=tf.float32, shape=[None, self.len_seq, 1], name="YT")
        YC = tf.placeholder(dtype=tf.float32, shape=[None, self.len_seq, 1], name="YC")
        YL = tf.placeholder(dtype=tf.float32, shape=[None, self.len_seq, 1], name="YL")
        XFZ = tf.concat([X, F, Z], axis=2, name="XFZ")

        cell = ChurnRNNCell(self.n_h, self.n_phi_prior, self.n_phi_encoder, self.n_phi_z_decoder, self.n_phi_h_decoder)
        output, last_state = tf.nn.dynamic_rnn(cell, XFZ, sequence_length=length(mask), dtype=tf.float32)
        lmbda_x_pred = output[0]
        lmbda_f_pred = output[1]
        z_pred = output[2]
        mu_encoder_pred = output[3]
        sigma_encoder_pred = output[4]
        h_pred = output[5]
        mu_prior_pred = output[6]
        sigma_prior_pred = output[7]
        x_pred = output[8]
        f_pred = output[9]
        z_pred_sigmoid = tf.nn.sigmoid(z_pred)

        TRL = self.time_reconstruction_loss(lmbda_x_pred, YT, mask)
        TML = self.mark_reconstruction_loss(lmbda_f_pred, YC, mask)
        # TKL = time_KL_loss(z_pred, mu_encoder_pred, sigma_encoder_pred, mu_prior_pred, sigma_prior_pred, mask, use_prior=False)
        TKL = self.time_KL_loss(z_pred, mu_encoder_pred, sigma_encoder_pred, mu_prior_pred, sigma_prior_pred, mask,
                           use_prior=self.use_prior)
        loss = TKL - TRL - TML
        MSE_x = self.prediction_mse(x_pred, YT, mask)
        MSE_x2 = self.prediction_mse2(x_pred, YT, mask)
        MAE_x = self.prediction_mae(x_pred, YT, mask)
        MRE_x = self.prediction_mre(x_pred, YT, mask)
        
        MSE_f = self.prediction_mse(f_pred, YC, mask)
        MSE_f2 = self.prediction_mse2(f_pred, YC, mask)
        MAE_f = self.prediction_mae(f_pred, YC, mask)
        MRE_f = self.prediction_mre(f_pred, YC, mask)

        lr = tf.placeholder_with_default(0.001, shape=[])
        train_operation = tf.train.AdamOptimizer(lr).minimize(loss)
        train_operation_mse = tf.train.AdamOptimizer(lr).minimize(MSE_x)

        sess.run(tf.global_variables_initializer())
        time_mse = {"Train":[], "Test":[]}
        time_mse2 = {"Train":[], "Test":[]}
        time_mae = {"Train":[], "Test":[]}
        time_mre = {"Train":[], "Test":[]}
        total_loss = {"Train":[], "Test":[]}
        mark_mse = {"Train":[], "Test":[]}
        mark_mse2 = {"Train":[], "Test":[]}
        mark_mae = {"Train":[], "Test":[]}
        mark_mre = {"Train":[], "Test":[]}
        kl = {"Train":[], "Test":[]}
        mark_rl = {"Train":[], "Test":[]}
        time_rl = {"Train":[], "Test":[]}
        
        #  Training the network
        for epoch in range(self.n_epoch):
            print(epoch)

#             feed_dict = {X: data['X_T_train'], YT: data['Y_T_train'], YC: data['Y_C_train'], mask: data['X_mask_train'], F: data['X_C_train']}
# #             l, mx, mf, kl_, time_rl_, mark_rl_ = sess.run([loss, MSE_x, MSE_f, TKL, -TRL, -TML], feed_dict=feed_dict)
#             l, mx, mx2, maex, mrx, mf, mf2, maef,mrf, kl_, time_rl_, mark_rl_  = sess.run([loss, MSE_x, MSE_x2, MAE_x, MRE_x, MSE_f, MSE_f2, MAE_f, MRE_f, TKL, -TRL, -TML], feed_dict=feed_dict)
#             time_mse["Train"].append(mx)
#             time_mse2["Train"].append(mx2)
#             time_mae["Train"].append(maex)
#             time_mre["Train"].append(mrx)
    
#             mark_mse["Train"].append(mf)
#             mark_mse2["Train"].append(mf2)
#             mark_mae["Train"].append(maef)
#             mark_mre["Train"].append(mrf)
            
#             total_loss["Train"].append(l)
#             kl["Train"].append(kl_)
#             mark_rl["Train"].append(mark_rl_)
#             time_rl["Train"].append(time_rl_)
#             print("Epoch = {}".format(epoch))
#             print(
#                 "train loss : ",  l, "\n",
#                 "train time mse : ", mx,"\n",
#                 "train time mse2 : ", mx2,"\n",
#                 "train time mae : ", maex, "\n",
#                 "train time mre : ", mrx, "\n",
#                 "train F mse : ", mf,"\n",
#                 "train F mse2 : ", mf2,"\n",
#                 "train F mae : ", maef,"\n",
#                 "train F mre : ", mrf,"\n"
#             )
# #             print("train loss : ", l, "train time mse : ", mx, "train F mse : ", mf)
#             feed_dict = {X: data['X_T_test'], YT: data['Y_T_test'], YC: data['Y_C_test'], mask: data['X_mask_test'], F: data['X_C_test']}

# #             l, mx, mf, kl_, time_rl_, mark_rl_ = sess.run([loss, MSE_x, MSE_f, TKL, -TRL, -TML], feed_dict=feed_dict)
#             l, mx, mx2, maex, mrx, mf, mf2, maef,mrf, kl_, time_rl_, mark_rl_  = sess.run([loss, MSE_x, MSE_x2, MAE_x, MRE_x, MSE_f, MSE_f2, MAE_f, MRE_f, TKL, -TRL, -TML], feed_dict=feed_dict)
#             print(
#                 "test loss : ",  l, "\n",
#                 "test time mse : ", mx, "\n",
#                 "test time mse2 : ", mx2, "\n",
#                 "test time mae : ", maex, "\n",
#                 "test time mre : ", mrx, "\n",
#                 "test F mse : ", mf,"\n",
#                 "test F mse2 : ", mf2,"\n",
#                 "test F mae : ", maef,"\n",
#                 "test F mre : ", mrf,"\n"
#             )
#             time_mse["Test"].append(mx)
#             time_mse2["Test"].append(mx2)
#             time_mae["Test"].append(maex)
#             time_mre["Test"].append(mrx)

#             mark_mse["Test"].append(mf)
#             mark_mse2["Test"].append(mf2)
#             mark_mae["Test"].append(maef)
#             mark_mre["Test"].append(mrf)

#             total_loss["Test"].append(l)
#             kl["Test"].append(kl_)
#             mark_rl["Test"].append(mark_rl_)
#             time_rl["Test"].append(time_rl_)
            print(len(data['X_T_train']))
            feed_dict = {X: data['X_T_train'], YT: data['Y_T_train'], YC: data['Y_C_train'], mask: data['X_mask_train'], F: data['X_C_train'],lr: get_lr(epoch)}
            _ = sess.run(train_operation, feed_dict=feed_dict)

#         # save results
#         session_length = data['session_length']
#         dataset_name = data['dataset_name']
#         result_dict = {}
#         result_dict["Time MSE"] = time_mse
#         result_dict["Time MSE2"] = time_mse2
#         result_dict["Time MAE"] = time_mae
#         result_dict["Time MRE"] = time_mre
        
#         result_dict["Mark MSE"] = mark_mse
#         result_dict["Mark MSE2"] = mark_mse2
#         result_dict["Mark MAE"] = mark_mae
#         result_dict["Mark MRE"] = mark_mre
        
#         result_dict["Time Reconstruction Loss"] = time_rl
#         result_dict["Mark Reconstruction Loss"] = mark_rl
#         result_dict["Loss"] = total_loss
#         result_dict["KL"] = kl
#         result_dict["Session Length"] = session_length
#         result_dict["Dataset Name"] = dataset_name
#         # result_dict["Lower Bound Time MSE"] = lower_bound_time_mse
#         # result_dict["Lower Bound Mark MSE"] = lower_bound_mark_mse
#         result_dict["Use Prior"] = self.use_prior

#         np.save("Result/Gaussian/{0}_{1}_{2}".format(dataset_name, session_length, run), result_dict)

    def prediction_mre(self, x_pred, y, mask):
        mre = tf.abs(x_pred[:, :, 0] - y[:, :, 0])/y[:, :, 0]
        mre = tf.where(tf.is_nan(mre), tf.zeros_like(mre), mre)
        len_mask = tf.cast(length(mask), dtype=tf.float32)
        ll = tf.reduce_sum(len_mask)
        mre = tf.reduce_sum(mre, axis=1)
        mre = tf.reduce_sum(mre, axis=0)
        mre = mre / ll
        return mre
    
    def prediction_mse2(self, x_pred, y, mask):
        mse = tf.square(x_pred[:, :, 0] - y[:, :, 0])
        len_mask = tf.cast(length(mask), dtype=tf.float32)
        ll = tf.reduce_sum(len_mask)
        mse = tf.reduce_sum(mse, axis=1)
        mse = tf.reduce_sum(mse, axis=0)
        mse = mse / ll
#     mse = tf.reduce_mean(mse, axis=0)
        return mse
    
    def prediction_mae(self, x_pred, y, mask):
        mae = tf.abs(x_pred[:, :, 0] - y[:, :, 0])
        len_mask = tf.cast(length(mask), dtype=tf.float32)
        ll = tf.reduce_sum(len_mask)
        mae = tf.reduce_sum(mae, axis=1)
        mae = tf.reduce_sum(mae, axis=0)
        mae = mae / ll
        return mae


    def prediction_mse(self, x_pred, y, mask):
        mse = tf.square(x_pred[:, :, 0] - y[:, :, 0])
        len_mask = tf.cast(length(mask), dtype=tf.float32)
        mse = tf.reduce_sum(mse, axis=1)
        mse = mse / len_mask
        mse = tf.reduce_mean(mse, axis=0)
        return mse

    def time_reconstruction_loss(self, lmbda_x_pred, y, mask):
        lmbda_x_pred_ = lmbda_x_pred[:, :, 0]
        y_ = y[:, :, 0]
        len_mask = tf.cast(length(mask), dtype=tf.float32)
        rl = tf.log(tf.maximum(self.epsilon, lmbda_x_pred_)) - tf.multiply(lmbda_x_pred_, y_)
        rl *= tf.cast(mask, dtype=tf.float32)
        rl = tf.reduce_sum(rl, axis=1)
        rl /= len_mask
        rl = tf.reduce_mean(rl, axis=0)
        return rl

    def mark_reconstruction_loss(self, lmbda_f, f, mask):
        lmbda_f_pred_ = lmbda_f[:, :, 0]
        f_ = f[:, :, 0]
        len_mask = tf.cast(length(mask), dtype=tf.float32)
        rl = tf.multiply(tf.log(tf.maximum(self.epsilon, lmbda_f_pred_)), f_) - lmbda_f_pred_
        rl *= tf.cast(mask, dtype=tf.float32)
        rl = tf.reduce_sum(rl, axis=1)
        rl /= len_mask
        rl = tf.reduce_mean(rl, axis=0)
        return rl

    def time_KL_loss(self, z_pred, mu_encoder_pred, sigma_encoder_pred, mu_prior_pred, sigma_prior_pred, mask,
                     use_prior=False):
        z_pred_ = z_pred[:, :, 0]
        mu_encoder_pred_ = mu_encoder_pred[:, :, 0]
        sigma_encoder_pred_ = sigma_encoder_pred[:, :, 0]
        mu_prior_pred_ = mu_prior_pred[:, :, 0]
        sigma_prior_pred_ = sigma_prior_pred[:, :, 0]
        if (use_prior):
            kl = (tf.log(tf.maximum(self.epsilon, sigma_encoder_pred_)) - tf.log(tf.maximum(self.epsilon, sigma_prior_pred_)) +
                  (tf.square(sigma_encoder_pred_) + tf.square(mu_encoder_pred_ - mu_prior_pred_)) / (
                              2 * tf.square(sigma_prior_pred_)) - 0.5)
            kl = tf.where(tf.is_nan(kl), tf.zeros_like(kl), kl)
        else:
            kl = (tf.log(tf.maximum(self.epsilon, sigma_encoder_pred_)) - tf.log(tf.maximum(self.epsilon, self.sigma0)) +
                  (tf.square(sigma_encoder_pred_) + tf.square(mu_encoder_pred_ - self.mu0)) / (2 * tf.square(self.sigma0)) - 0.5)
        len_mask = tf.cast(length(mask), dtype=tf.float32)
        kl = kl * tf.cast(mask, dtype=tf.float32)
        kl = tf.reduce_sum(kl, axis=1)
        kl /= len_mask
        kl = tf.reduce_mean(kl)
        return kl


def length(mask):
    return tf.reduce_sum(mask, axis=1)


def logit(z, epsilon):
    logitz = tf.log(tf.maximum(epsilon, z)) - tf.log(tf.maximum(epsilon, 1 - z))
    return logitz


def get_lr(epoch):
    if (epoch < 5):
        return 0.1
    if (epoch < 10):
        return 0.05
    if (epoch < 15):
        return 0.025
    if (epoch < 20):
        return 0.01
    if (epoch < 30):
        return 0.005
    return 0.001
