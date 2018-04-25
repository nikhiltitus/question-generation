import numpy as np
import tensorflow as tf
import pickle
from QuestionAnswers import QuestionAnswers
from nltk.tokenize import word_tokenize

#File locations for laoding glove and pickle dump of questions and answers

class seq_to_seq():
    def __init__(self):
        self.glove_file_location='../data/glove.6B/glove.6B.300d.txt'
        self.pickle_file_location='../data/squad/qa_dump'
        self.train_size=50
        self.Max_time=80
        self.lstm_size=128
        self.Embedding_dimension=300

        self.word2idx={}
        self.weights=[]
        self.length_questions=[]
        self.length_answers=[]
        self.train_questions=[]
        self.train_answers=[]
        self.train_question_array=np.zeros((self.train_size,self.Max_time))
        self.train_answer_array=np.zeros((self.train_size,self.Max_time))

    def prepare_glove_vector(self):
        self.word2idx={'PAD':0,'STOP':1,'START':2,'UNK':3}
        self.weights=[np.random.randn(300) for _ in range(4)]
        with open(self.glove_file_location) as glove_file:
            i=0
            for index,line in enumerate(glove_file):
                line_split=line.split()
                word=line_split[0]
                word_weights=np.asarray(line_split[1:],dtype=np.float32)
                self.weights.append(word_weights)
                self.word2idx[word.lower()]=index+4
                if index>48000:
                    break
        self.weights=np.asarray(self.weights,dtype=np.float32)
    def prepare_questions_and_answers(self):
        pickle_file_location='../data/squad/qa_dump'
        Question_answers=[]
        self.Vocab_size=len(self.word2idx)
        self.train_answer_array_one_hot=np.zeros(shape=(self.train_size,self.Max_time,self.Vocab_size))
        with open(self.pickle_file_location) as pickle_file:
            Question_answers=pickle.load(pickle_file)[0]
        for i,qa in enumerate(Question_answers[:self.train_size]):
            tokenized_question=word_tokenize((qa.question))
            self.length_questions.append(len(tokenized_question))
            new_question=[]
            for word in tokenized_question:
                new_question.append(self.word2idx.get(word.lower(),self.word2idx['UNK']))
            while len(new_question)<self.Max_time:
                new_question.append(0)
            self.train_questions.append(np.array(new_question))
            self.train_question_array[i]=np.array(new_question[:self.Max_time])
        
            tokenized_answer=word_tokenize(qa.answer)
            self.length_answers.append(len(tokenized_answer))
            new_answer=[]
            new_answer_one_hot=np.zeros(shape=(self.Max_time,self.Vocab_size))
            for j,word in enumerate(tokenized_answer):
                new_answer_one_hot[i,self.word2idx.get(word.lower(),self.word2idx['UNK'])]=1
                new_answer.append(self.word2idx.get(word.lower(),self.word2idx['UNK']))
            new_answer_one_hot[j,self.word2idx['STOP']]=1
            new_answer.append(self.word2idx['STOP'])
            while len(new_answer)<self.Max_time:
                j+=1
                new_answer_one_hot[j,self.word2idx['PAD']]=1
                new_answer.append(0)
            self.train_answers.append(np.array(new_answer))
            self.train_answer_array[i]=np.array(new_answer[:self.Max_time])
            self.train_answer_array_one_hot[i,:,:]=np.array(new_answer_one_hot[:self.Max_time,:])

    def train_model(self):
        self.batch_size=25
        self.prepare_glove_vector()
        self.prepare_questions_and_answers()
        glove_weights_initializer=tf.constant_initializer(self.weights)
        embedding_weights=tf.get_variable(name='embedding_weights',
                                        shape=(self.Vocab_size,self.Embedding_dimension),
                                        trainable=True,initializer=glove_weights_initializer)
        
        
        
        target_out=tf.placeholder(dtype=tf.float32,shape=[None,None,self.Vocab_size])
        encoder_inputs=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.Max_time])
        decoder_inputs=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.Max_time])
        encoder_lengths=tf.placeholder(dtype=tf.int32,shape=None)
        decoder_lengths=tf.placeholder(dtype=tf.int32,shape=None)

        with tf.variable_scope('encoder'):
            encoder_input_embeds=tf.nn.embedding_lookup(embedding_weights,encoder_inputs)
            encoder_lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
            encoder_output,encoder_states=tf.nn.dynamic_rnn(encoder_lstm_cell,encoder_input_embeds,
                                                        sequence_length=encoder_lengths,
                                                        dtype=tf.float32)

        with tf.variable_scope('decoder'):
            decoder_inputs_embeds=tf.nn.embedding_lookup(embedding_weights,decoder_inputs)
            decoder_lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
            decoder_output,decoder_states=tf.nn.dynamic_rnn(decoder_lstm_cell,decoder_inputs_embeds,
                                                        initial_state=encoder_states,
                                                        sequence_length=decoder_lengths,
                                                        dtype=tf.float32)

        weights=tf.Variable(tf.random_normal([self.lstm_size,self.Vocab_size]))
        biases=tf.Variable(tf.random_normal([self.Vocab_size]))
        sess = tf.Session()

        scores=tf.einsum('ijk,kl -> ijl',decoder_output,weights)+biases
        #training_loss=tf.losses.mean_squared_error(target_out,scores)
        training_loss=tf.losses.softmax_cross_entropy(target_out,scores)
        params=tf.trainable_variables()
        gradients=tf.gradients(training_loss,params)
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-2)
        
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())  
        update_step=optimizer.apply_gradients(zip(gradients,params))

        
        sess.run(tf.global_variables_initializer())
        epoch_number=10
        
        for j in range(epoch_number):
            for i in range(0,self.train_size,self.batch_size):
                print "Epoch : {0}, Sample :{1} of {2}".format(j,i,self.train_size)
                feed_dict={
                        encoder_inputs: self.train_question_array[:self.batch_size,:],
                        decoder_inputs: self.train_answer_array[:self.batch_size,:],
                        target_out: self.train_answer_array_one_hot[:self.batch_size,:,:],
                        encoder_lengths: self.length_questions[:self.batch_size],
                        decoder_lengths: self.length_answers[:self.batch_size]
                }
                loss = sess.run(training_loss, feed_dict=feed_dict)
                print loss
                sess.run(update_step,feed_dict=feed_dict)



        







#encoder_emb_inp=tf.nn.embedding_lookup(embedding_weights,encoder_inputs)
    
        

if __name__=="__main__":
    model=seq_to_seq()
    model.train_model()

