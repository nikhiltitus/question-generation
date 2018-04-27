import numpy as np
import tensorflow as tf
import pickle
import Paragraph as p
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential


glove_file_location='../data/glove.6B/glove.6B.300d.txt'
def prepare_glove_vector():
    word2idx={'PAD':0,'STOP':1,'START':2,'UNK':3}
    index2word={0:'PAD',1:'STOP',2:'START',3:'UNK'}
    weights=[np.random.randn(300) for _ in range(4)]
    with open(glove_file_location) as glove_file:
        i=0
        for index,line in enumerate(glove_file):
            line_split=line.split()
            word=line_split[0]
            word_weights=np.asarray(line_split[1:],dtype=np.float32)
            weights.append(word_weights)
            word2idx[word.lower()]=index+4
            index2word[index+4]=word.lower()
            if index>48000:
                break
    weights=np.asarray(weights,dtype=np.float32)
    return weights,word2idx,index2word



pickle_file_location='../data/squad/text_sel_dump'

with open(pickle_file_location) as file:
    processed_data=pickle.load(file)
sample_input= processed_data.data[0].paragraph

weights,word2idx,index2word=prepare_glove_vector()
Vocab_size=len(weights)
Embedding_dimension=300

#keras sequential model
model=Sequential()

glove_weights_initializer=tf.constant_initializer(weights)
embedding_weights=tf.get_variable(name='embedding_weights',
                                  shape=(Vocab_size,Embedding_dimension),
                                  trainable=True,initializer=glove_weights_initializer)
input=tf.placeholder(dtype=tf.int32,shape=None)
embedded_vectors=tf.nn.embedding_lookup(embedding_weights,input)

# we try sample embedding for a paragrph

sample_par_embed=None
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sample_par_embed=sess.run(embedded_vectors,feed_dict={
        input:np.array(sample_input)
    })

para_sentece_embedding=np.zeros((len(processed_data.data[0].sentence_lengths),300))
start=0
for i,length in enumerate(processed_data.data[0].sentence_lengths):
    print length
    para_sentece_embedding[i]=np.sum(sample_par_embed[start:start+length])
    start+=length
model.add(LSTM(128,activation='sigmoid',return_sequences=True,input_shape=[None,300]))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(para_sentece_embedding[np.newaxis,:,:],
          (processed_data.data[0].question_worthiness)[np.newaxis,:,np.newaxis],epochs=10)
    
