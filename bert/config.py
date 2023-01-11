"""
Configuration file for BERT model(small)
"""

# training options
epoch = 100
batch_size = 32
learning_rate = 1e-4
source_path = './data/'

# model options
layer_num = 6
d_model = 768
max_len = 512