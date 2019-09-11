from network import Network
from load_data import Data_Loader
from config import setting
import tensorflow as tf
import numpy as np
from datetime import datetime
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


'''
This code is implemented from
Recurrent Marked Temporal Point Processes: Embedding Event History to Vector
by
Nan Du
Georgia Tech
dunan@gatech.edu,

Hanjun Dai
Georgia Tech
hanjundai@gatech.edu,

Rakshit Trivedi
Georgia Tech
rstrivedi@gatech.edu,

Utkarsh Upadhyay
MPI-SWS
utkarshu@mpi-sws.org,

Manuel Gomez-Rodriguez
MPI-SWS
manuelgr@mpi-sws.org,

Le Song
Georgia Tech
lsong@cc.gatech.edu

paper: www.kdd.org/kdd2016/papers/files/rpp1081-duA.pdf
'''
gpu_options = tf.GPUOptions(allow_growth=True)
config =  tf.ConfigProto(gpu_options=gpu_options)

# ds_names = ['lastfm', 'tianchi', 'foursquare', 'iptv']
ds_names = ['iptv']
sess_length = '1.0'

for ds_name in ds_names:
    tf.reset_default_graph()
    setting["data_path"] = "../../../data/user_session_dictionary_{0}_{1}.npy".format(ds_name, sess_length)
    data_loader = Data_Loader(setting=setting)
    train_time, train_mark, train_mark_reshaped, X_mask_train, test_time, test_mark, test_mark_reshaped, X_mask_test = data_loader.load_data_set()
    print("Data Loaded ....")
    setting["len_seq"] = data_loader.T
    setting["y_size"] = data_loader.max_C + 1
    
    start=datetime.now()
    network = Network(setting=setting)
    sess = tf.Session(config=config)
    network.create_network()
    network.train_network(sess=sess, train_time = train_time, train_mark=train_mark,train_mark_reshaped=train_mark_reshaped, X_mask_train=X_mask_train)
    res_dict = network.test_network(sess=sess, test_time = test_time, test_mark=test_mark,test_mark_reshaped=test_mark_reshaped, X_mask_test=X_mask_test)
    np.save("Result/{0}_{1}".format(ds_name, sess_length), res_dict)
    end=datetime.now()
    run_dur = end-start
    res_dict['duration']=run_dur
    print("\033[95m DataSet:{0}, Run Duration is: {1}".format(ds_name, run_dur))
    print('\033[92m')
    sess.close()
            
            

