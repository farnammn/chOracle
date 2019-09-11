import numpy as np
class Data_Loader:

   def __init__(self, setting):
        self.data_path = setting["data_path"]
        self.epsilon = 1e-12
        self.scale_param = setting["scale_param"]


   def load_data_set(self):
       # code to load data
       dataset = np.load(self.data_path).item()

       self.T = 0
       for k in dataset.keys():
           self.T = max(self.T, len(dataset[k][0]))

       self.max_C = 0
       for k in dataset.keys():
           self.max_C = max(self.max_C, max(dataset[k][2]))

       n_data = len(dataset.keys())
       X_T = np.zeros([n_data, self.T, 1])
       X_L = np.zeros([n_data, self.T, 1])
       X_C = np.zeros([n_data, self.T, 1])
       Y_T = np.zeros([n_data, self.T, 1])
       Y_L = np.zeros([n_data, self.T, 1])
       Y_C = np.zeros([n_data, self.T, 1])

       X_mask = np.zeros([n_data, self.T], dtype=np.int16)
       for i, user in enumerate(dataset.keys()):
           t, l, c = dataset[user]
           t = np.array(t)
           # t[1:] = t[1:] - t[:-1]
           # t[0] = 0.0
           l = np.array(l)
           c = np.array(c)
           length = len(t)
           X_T[i][:length] = (t[:, np.newaxis])
           X_L[i][:length] = (l[:, np.newaxis])
           X_C[i][:length] = (c[:, np.newaxis])
           X_mask[i][:length] = 1

       X_T /= self.scale_param
       # X_C /= scale_param
       # Y_C /= scale_param
       n_train = n_data * 70 // 100

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
       train_time = X_T_train
       test_time = X_T_test

       train_mark_reshaped = self.to_one_hot(train_mark, self.max_C)
       test_mark_reshaped = self.to_one_hot(test_mark, self.max_C)

       train_mark = train_mark.reshape(-1, self.T)
       test_mark = test_mark.reshape(-1, self.T)
    
       print("whole data has been loaded")

       return train_time, train_mark, train_mark_reshaped, X_mask_train, test_time, test_mark, test_mark_reshaped, X_mask_test


   def to_one_hot(self, x , max_num):
        shape = x.shape
        out = np.zeros((shape[0], shape[1], max_num + 1))
        for i in range(shape[0]):
            for j in range(shape[1]):
                out[i, j , int(x[i, j])] = 1
        return out