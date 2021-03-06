import numpy as np
import tensorflow as tf
import pickle
from QuestionAnswers import QuestionAnswers
from nltk.tokenize import word_tokenize
from tensorflow.python.layers import core as layers_core

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
        self.train_question_array=np.zeros((self.train_size,self.Max_time+1),dtype=np.int32)
        self.train_answer_array=np.zeros((self.train_size,self.Max_time),dtype=np.int32)
        

    def prepare_glove_vector(self):
        self.word2idx={'PAD':0,'STOP':1,'START':2,'UNK':3}
        self.index2word={0:'PAD',1:'STOP',2:'START',3:'UNK'}
        self.weights=[np.random.randn(300) for _ in range(4)]
        with open(self.glove_file_location) as glove_file:
            i=0
            for index,line in enumerate(glove_file):
                line_split=line.split()
                word=line_split[0]
                word_weights=np.asarray(line_split[1:],dtype=np.float32)
                self.weights.append(word_weights)
                self.word2idx[word.lower()]=index+4
                self.index2word[index+4]=word.lower()
                if index>48000:
                    break
        self.weights=np.asarray(self.weights,dtype=np.float32)

    def prepare_questions_and_answers(self):
        pickle_file_location='../data/squad/qa_dump'
        Question_answers=[]
        self.Vocab_size=len(self.word2idx)
        self.train_question_array_one_hot=np.zeros(shape=(self.train_size,self.Max_time+1,self.Vocab_size))
        with open(self.pickle_file_location) as pickle_file:
            Question_answers=pickle.load(pickle_file)[0]
        for i,qa in enumerate(Question_answers[:self.train_size]):
            tokenized_question=word_tokenize((qa.question))
            self.length_questions.append(len(tokenized_question)+1)
            new_question=[self.word2idx['START']]
            new_question_one_hot=np.zeros(shape=(self.Max_time+1,self.Vocab_size))
            new_question_one_hot[0,self.word2idx['START']]=1
            for j,word in enumerate(tokenized_question):
                new_question_one_hot[j+1,self.word2idx.get(word.lower(),self.word2idx['UNK'])]=1
                new_question.append(self.word2idx.get(word.lower(),self.word2idx['UNK']))
            j+=1
            new_question_one_hot[j+1,self.word2idx['STOP']]=1
            new_question.append(self.word2idx['STOP'])
            while len(new_question)<self.Max_time+1:
                j+=1
                new_question.append(0)
                new_question_one_hot[j+1,self.word2idx['PAD']]=1
            self.train_questions.append(np.array(new_question))
            self.train_question_array[i]=np.array(new_question[:self.Max_time+1])
        
            tokenized_answer=word_tokenize(qa.answer)
            self.length_answers.append(len(tokenized_answer))
            new_answer=[]
            for j,word in enumerate(tokenized_answer):
                new_answer.append(self.word2idx.get(word.lower(),self.word2idx['UNK']))
            
            while len(new_answer)<self.Max_time:
                j+=1
                new_answer.append(0)
            self.train_answers.append(np.array(new_answer))
            self.train_answer_array[i]=np.array(new_answer[:self.Max_time])
            self.train_question_array_one_hot[i,:,:]=np.array(new_question_one_hot[:self.Max_time+1,:])
            self.length_answers_array=np.array(self.length_answers)
            self.length_questions_array=np.array(self.length_questions)
        self.generate_sentences(self.train_question_array_one_hot)

    def train_model(self):
        self.batch_size=2
        self.prepare_glove_vector()
        self.prepare_questions_and_answers()
        glove_weights_initializer=tf.constant_initializer(self.weights)
        embedding_weights=tf.get_variable(name='embedding_weights',
                                        shape=(self.Vocab_size,self.Embedding_dimension),
                                        trainable=True,initializer=glove_weights_initializer)
        
        
        
        target_out=tf.placeholder(dtype=tf.float32,shape=[None,None,self.Vocab_size])
        # target_out=tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.Max_time])
        encoder_inputs=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.Max_time])
        decoder_inputs=tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.Max_time])
        #target_out=tf.placeholder(dtype=tf.int32,shape=[self.Max_time,self.batch_size])
        # encoder_inputs=tf.placeholder(dtype=tf.int32,shape=[self.Max_time,self.batch_size])
        # decoder_inputs=tf.placeholder(dtype=tf.int32,shape=[self.Max_time,self.batch_size])
        encoder_lengths=tf.placeholder(dtype=tf.int32,shape=self.batch_size)
        decoder_lengths=tf.placeholder(dtype=tf.int32,shape=self.batch_size)
        scores=layers_core.Dense(self.Vocab_size, use_bias=False)

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
        # with tf.variable_scope('decoder'):
            
        #     decoder_inputs_embeds=tf.nn.embedding_lookup(embedding_weights,decoder_inputs)
            
        #     print "Decoder input shape",decoder_inputs.shape
        #     print "Decoder Lengths : ",decoder_lengths
        #     decoder_lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size)
        #     helper = tf.contrib.seq2seq.TrainingHelper(
        #             decoder_inputs_embeds, decoder_lengths, time_major=True)
        #     decoder = tf.contrib.seq2seq.BasicDecoder(
        #                     decoder_lstm_cell, helper, initial_state=encoder_states,
        #                     output_layer=scores)
        #     decoder_output,final_context,_=tf.contrib.seq2seq.dynamic_decode(decoder,
        #                         output_time_major=False,swap_memory=True,
        #                         maximum_iterations=self.Max_time)
        #     logits=decoder_output.rnn_output

        weights=tf.Variable(tf.random_normal([self.lstm_size,self.Vocab_size]))
        biases=tf.Variable(tf.random_normal([self.Vocab_size]))
        sess = tf.Session()
        
        scores=tf.einsum('ijk,kl -> ijl',decoder_output,weights)+biases
        #training_loss=tf.losses.mean_squared_error(target_out,scores)
        #print logits.shape
        #print target_out.shape
        cross_ent=tf.nn.softmax_cross_entropy_with_logits(labels=target_out,logits=scores)
        training_loss=(tf.reduce_sum(cross_ent*tf.cast(tf.sequence_mask(decoder_lengths,
                                        self.Max_time),tf.float32))/self.batch_size)
        params=tf.trainable_variables()
        gradients=tf.gradients(training_loss,params)
        #clipped_gradients,_=tf.clip_by_global_norm(gradients,1)
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-2)
        
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())  
        #update_step=optimizer.apply_gradients(zip(clipped_gradients,params))
        update_step=optimizer.apply_gradients(zip(gradients,params))
        
        sess.run(tf.global_variables_initializer())
        epoch_number=20
        
        for j in range(epoch_number):
            for i in range(0,self.train_size,self.batch_size):
                print "Epoch : {0}, Sample :{1} of {2}".format(j,i,self.train_size)
                print self.train_answer_array.shape
                print self.length_answers_array[:10]
                feed_dict={
                        encoder_inputs: (self.train_answer_array[:self.batch_size,:]),
                        decoder_inputs: (self.train_question_array[:self.batch_size,:self.Max_time]),
                        target_out: self.train_question_array_one_hot[:self.batch_size,1:,:],
                        #target_out:(self.train_question_array[:self.batch_size,:self.Max_time]).T,
                        encoder_lengths: self.length_answers_array[:self.batch_size],
                        decoder_lengths: self.length_questions_array[:self.batch_size]
                }
                loss = sess.run(training_loss, feed_dict=feed_dict)
                print np.mean(loss)
                sess.run(update_step,feed_dict=feed_dict)
                sampl_scores=sess.run(scores,feed_dict=feed_dict)
                #self.generate_sentences(sampl_scores[:5,:,:])
                #print "EXPECTED"
                #self.generate_sentences(self.train_question_array_one_hot[:self.batch_size,:self.Max_time,:])
                print "GOT"
                self.generate_sentences(sampl_scores[:5,:,:])

    
    def generate_sentences(self,word_vector):
        decompiled_sentences=[]
        max_vectors=np.argmax(word_vector,axis=2)
        for sentence in max_vectors:
            current_sentence=[]
            for word in sentence:
                if self.index2word[word]!='STOP':
                    current_sentence.append(self.index2word[word]) 
                else:
                    break
            print (' '.join(current_sentence))
            decompiled_sentences.append(current_sentence)
        return decompiled_sentences


        







#encoder_emb_inp=tf.nn.embedding_lookup(embedding_weights,encoder_inputs)
    
        

if __name__=="__main__":
    model=seq_to_seq()
    model.train_model()
    #model.prepare_glove_vector()
    #model.prepare_questions_and_answers()

