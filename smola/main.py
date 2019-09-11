from network import Network
from load_data import Data_Loader
from config import setting
import tensorflow as tf
from datetime import datetime
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


'''
This code is implemented from:
Neural Survival Recommender, Amazon
'''
gpu_options = tf.GPUOptions(allow_growth=True)
config =  tf.ConfigProto(gpu_options=gpu_options)

ds_names = ['lastfm', 'tianchi', 'foursquare', 'iptv']
sess_length = '1.0'

for ds_name in ds_names:
    tf.reset_default_graph()
    setting["data_path"] = "../../../data/user_session_dictionary_{0}_{1}.npy".format(ds_name, sess_length)
    setting["dataset_name"]=ds_name
    data_loader = Data_Loader(setting=setting)
    train_e, train_b, train_g, train_g_masked, train_mark, X_mask_train, test_e, test_b, test_g, test_g_masked, test_mark, X_mask_test = data_loader.load_data_set()
    setting["len_seq"] = data_loader.T
    
    network = Network(setting=setting)
    sess = tf.Session(config=config)
    start=datetime.now()
    network.create_network()
    network.train_network(sess=sess, train_e=train_e, train_b=train_b, train_g=train_g, train_mark=train_mark, X_mask_train=X_mask_train, train_g_masked=train_g_masked)
    res_dict = network.test_network(sess=sess, test_e=test_e, test_b=test_b, test_g=test_g, test_mark=test_mark, X_mask_test=X_mask_test, test_g_masked=test_g_masked)
    end=datetime.now()
    dur =  end - start
    res_dict['duration'] = dur
    np.save("Result/{0}_{1}".format(setting["dataset_name"],"1.0"), res_dict)
    sess.close()

'''
####################ATTENTION######################
it seems like a good idea to list the problems we still have:
1- we don't have the T in the datasets wich in the paper used it for adding a loss
I commented the part wich include the third loss, wich if we have the T it can be easily added
2- we don't have the time of the week in cleaned data in the paper it was used as a vector, although it 
can be easily added just by concating the input with d vector
'''
