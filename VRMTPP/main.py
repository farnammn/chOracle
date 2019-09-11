from DataLoader import DataLoader
from Network import Network
import tensorflow as tf
import os
from datetime import datetime

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)

'''
Load data
'''
# ds_names = ['lastfm', 'tianchi', 'foursquare', 'iptv']
ds_names = ['tianchi']
# sess_lengths = [1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0]
sess_lengths = [1.0]
for name in ds_names: 
    for length in sess_lengths:
        dataset_name = name
        session_length = length
        data_base_path = "../../../data/"
        data_path = '{0}user_session_dictionary_{1}_{2}.npy'.format(data_base_path, dataset_name, session_length)

        data_settings = {
            "data_path": data_path,
            "dataset_name": dataset_name,
            'session_length': session_length,
            "scale_param": 100,
        }
        dl = DataLoader(data_settings)
        data = dl.load_data()

        print("Data Loaded ....")
        print("Data Statistics are:")
        print(str(dl))

        network_settings = {
            'n_sampling': 90,
            'n_h': 60,
            'n_phi_prior': 10,
            'n_phi_encoder': 10,
            'n_phi_z_decoder': 10,
            'n_phi_h_decoder': 10,
            'mu0': 0,
            'sigma0': 1.8,
            'n_epoch': 10,
            'learning_rate': 0.03,
            'epsilon': 1e-12,
            'use_prior': True,
            'len_seq': data['T']
        }
        run_dur = 0
        for i in range(1):
            tf.reset_default_graph()
            print('\033[92m')
            
            network = Network(network_settings)
            start=datetime.now()
            print("Network Created ....")
            print("Network Info is:")
            print(str(network))
            sess = tf.InteractiveSession(config=config)
            network(data, sess, i)
            end=datetime.now()
            run_dur = end-start
            print("\033[95m RUN:{0}, Prev Run Duration is: {1}".format(i, run_dur))
