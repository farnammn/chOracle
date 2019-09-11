import numpy as np


class Data_Loader:
    def __init__(self, setting):
        self.data_path = setting["data_path"]
        self.epsilon = 1e-12
        self.scale_param = setting["scale_param"]
        self.g_size = setting["g_size"]
        self.delta_t = setting["delta_t"]
        self.max_predict = setting["max_predict"]
        self.T = 0
        self.max_C = 0

    def load_data_set(self):
        # code to load data
        dataset = np.load(self.data_path).item()
        n_data = 0
        for k in dataset.keys():
            if len(dataset[k][0]) > 10:
                n_data += 1
                self.T = max(self.T, len(dataset[k][0]))
                self.max_C = max(self.max_C, max(dataset[k][2]))
        X_T = np.zeros([n_data, self.T, 1])
        X_L = np.zeros([n_data, self.T, 1])
        X_C = np.zeros([n_data, self.T, 1])
        Y_T = np.zeros([n_data, self.T, 1])
        Y_L = np.zeros([n_data, self.T, 1])
        Y_C = np.zeros([n_data, self.T, 1])
        X_mask = np.zeros([n_data, self.T], dtype=np.int16)

        ind = 0
        for i, user in enumerate(dataset.keys()):
            t, l, c = dataset[user]
            if len(t) <= 10:
                continue
            t = np.array(t)
            # t[1:] = t[1:] - t[:-1]
            # t[0] = 0.0
            l = np.array(l)
            c = np.array(c)
            length = len(t)

            X_T[ind][:length] = (t[:, np.newaxis])
            X_L[ind][:length] = (l[:, np.newaxis])
            X_C[ind][:length] = (c[:, np.newaxis])
            X_mask[ind][:length] = 1
            ind += 1

        X_T /= self.scale_param
        # X_C /= scale_param
        # Y_C /= scale_param

        n_train = n_data * 80 // 100
        n_test = n_data - n_train

        X_T_train = X_T[:n_train]  # time
        X_L_train = X_L[:n_train]  # length
        X_C_train = X_C[:n_train]  # count
        X_mask_train = X_mask[:n_train]
        X_T_test = X_T[n_train:]
        X_L_test = X_L[n_train:]
        X_C_test = X_C[n_train:]
        X_mask_test = X_mask[n_train:]

        train_mark = X_C_train
        test_mark = X_C_test
        train_b = X_T_train
        test_b = X_T_test
        train_length = X_L_train / self.scale_param
        test_length = X_L_test / self.scale_param

        train_e = train_b + train_length
        test_e = test_b + test_length

        ''' this is the gap time '''
        train_g1 = train_b[:, 1:] - train_e[:, :-1]
#         shp = train_g1.shape
#         zeross = np.zeros([shp[0], 1, 1])
#         train_g1 = np.concatenate((zeross, train_g1), axis=0)
        print(train_g1.shape)
        '''
         gap time is -1 in the last b but if we change all minuses to zero every  minus to zero problem will be solved but we should use masked
        '''
        train_g1[train_g1 < 0] = 0
        train_g1 = (train_g1 / self.delta_t).astype(int)
        train_g = self.to_one_hot(train_g1, self.g_size - 1)
        train_g_masked = self.to_mask(train_g1, self.max_predict)

        test_g1 = test_b[:, 1:] - test_e[:, :-1]
        test_g1[test_g1 < 0] = 0
        test_g1 = (test_g1 / self.delta_t).astype(int)
        test_g = self.to_one_hot(test_g1, self.g_size - 1)
        test_g_masked = self.to_mask(test_g1, self.max_predict)
        
        print(train_e[0][0])
        print(train_b[0][1])
        print(train_g1[0][0])
        cnt = 0
        for idx,item in enumerate(train_g_masked[0][0]):
            print("cnt:{0}, mask:{1}".format(idx,item))

        print("the whole data has been loaded and normalized successfuly!")

        return train_e, train_b, train_g, train_g_masked, train_mark, X_mask_train, test_e, test_b, test_g, test_g_masked, test_mark, X_mask_test

    def to_one_hot(self, x, max_num):
        shape = x.shape
        print(shape)
        out = np.zeros((shape[0], shape[1], max_num + 1))
        for i in range(shape[0]):
            for j in range(shape[1]):
                ind = min(int(x[i, j]), max_num)
                out[i, j, ind] = 1
        return out

    def to_mask(self, x, max_num):
        shape = x.shape
        print(shape[0])
        print(shape[1])
        print(max_num)
        out = np.zeros((shape[0], shape[1], max_num))
        for i in range(shape[0]):
            for j in range(shape[1]):
                ind = min(int(x[i, j]), max_num-2)
                out[i, j, :ind + 1] = 1
        return out
