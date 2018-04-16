import numpy as numpy
import tensorflow as tf


word2idx={'PAD':0}
weights=[]
def intialize_glove_vectors():
    with open(glove_file_location) as glove_file:
        i=0
        for index,line in enumerate(glove_file):
            line_split=line.split()
            word=line_split[0]
            word_weights=np.asarray(line_split[1:],dtype=np.float32)
            weights.append(word_weights)
            word2idx[word]=index+1
            if index>48000:
                break

class seq2seqModel():
    def __init__(self,rnn_size=128,embedding_dim=300):
        self.rnn_size=rnn_size
        self.x=tf.placeholder(dtype=tf.float32,shape=[None,None,embedding_dim])
        self.y=tf.placeholder(dtype=tf.float32,shape=[None,None,embedding_dim])
        self.encoder_lengths=tf.placeholder(dtype=tf.int32,shape=None)
        self.decoder_lengths=tf.placeholder(dtype=tf.int32,shape=None)
        encoder_lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        decoder_lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
        