import numpy as np


class DataLoader:
    def __init__(self, setting):
        self.data_path = setting["data_path"]
        self.dataset_name = setting["dataset_name"]
        self.session_length = setting['session_length']
        self.scale_param = setting["scale_param"]
        self.T = 0
        self.n_data = 0

    def load_data(self):
        dataset = np.load(self.data_path).item()
        max_T = 0
        n_data = 0
        for k in dataset.keys():
            if (len(dataset[k][0]) > 10):
                n_data += 1
            max_T = max(max_T, len(dataset[k][0]))

        self.T = max_T

        # code to load data
        X_T = np.zeros([n_data, self.T, 1])
        X_L = np.zeros([n_data, self.T, 1])
        X_C = np.zeros([n_data, self.T, 1])
        Y_T = np.zeros([n_data, self.T, 1])
        Y_L = np.zeros([n_data, self.T, 1])
        Y_C = np.zeros([n_data, self.T, 1])
        X_mask = np.zeros([n_data, self.T], dtype=np.int16)
        i = -1
        for j, user in enumerate(dataset.keys()):
            t, l, c = dataset[user]
            if (len(t) <= 10):
                continue
            i += 1
            t = np.array(t)
            t[1:] = t[1:] - t[:-1]
            t[0] = 0.0
            l = np.array(l)
            c = np.array(c)
            length = len(t)
            X_T[i][:length - 1] = (t[:-1, np.newaxis])
            X_L[i][:length - 1] = (l[:-1, np.newaxis])
            X_C[i][:length - 1] = (c[:-1, np.newaxis])
            Y_T[i][:length - 1] = (t[1:, np.newaxis])
            Y_L[i][:length - 1] = (l[1:, np.newaxis])
            Y_C[i][:length - 1] = (c[1:, np.newaxis])
            X_mask[i][:length - 1] = 1
            
            
        X_T /= self.scale_param
        Y_T /= self.scale_param
        X_C /= self.scale_param
        Y_C /= self.scale_param

        n_train = n_data * 80 // 100
        n_test = n_data - n_train

        X_T_train = X_T[:n_train]
        X_L_train = X_L[:n_train]
        X_C_train = X_C[:n_train]
        X_mask_train = X_mask[:n_train]
        X_T_test = X_T[n_train:]
        X_L_test = X_L[n_train:]
        X_C_test = X_C[n_train:]
        X_mask_test = X_mask[n_train:]

        Y_T_train = Y_T[:n_train]
        Y_L_train = Y_L[:n_train]
        Y_C_train = Y_C[:n_train]
        Y_T_test = Y_T[n_train:]
        Y_L_test = Y_L[n_train:]
        Y_C_test = Y_C[n_train:]

        result = dict()
        result['dataset_name'] = self.dataset_name
        result['session_length'] = self.session_length
        result['T'] = self.T
        result['n_data'] = self.n_data
        result['X_T_train'] = X_T_train
        result['X_L_train'] = X_L_train
        result['X_C_train'] = X_C_train
        result['X_mask_train'] = X_mask_train
        result['X_T_test'] = X_T_test
        result['X_L_test'] = X_L_test
        result['X_C_test'] = X_C_test
        result['X_mask_test'] = X_mask_test
        result['Y_T_train'] = Y_T_train
        result['Y_L_train'] = Y_L_train
        result['Y_C_train'] = Y_C_train
        result['Y_T_test'] = Y_T_test
        result['Y_L_test'] = Y_L_test
        result['Y_C_test'] = Y_C_test

        return result

    def __str__(self):
        return '\tdataset: {0},\n' \
                   '\tsession_length: {1},\n' \
                   '\tn_data: {2},\n' \
                   '\tlen_seq: {3},\n' \
               '\tscale_param: {4}, \n'.format(self.dataset_name, self.session_length, self.n_data, self.T, self.scale_param)

