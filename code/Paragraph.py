# from nltk.tokenize import sent_tokenize
# from nltk.tokenize import word_tokenize
# para=sent_tokenize(inp)
# for i in range(0,len(para)):
#     para[i]=word_tokenize(para[i])
import numpy as np

class Paragraph():
    def __init__(self):
        self.paragraph=[]
        self.sentence_lengths=[]
        self.question_worthiness=[]
    
    

class squad_data():
    def __init__(self):
        # Length of longest paragraph in number of lines
        self.max_sent_length=0
        # Length of longest paragraph in number of words
        self.max_par_length=0
        #data - list of paragraphs
        self.data=[]#list of paragraphs
        self.glove_file_location='../data/glove.6B/glove.6B.300d.txt'
        #weights for glove vector
        self.weights=None
        self.prepare_glove_vector()


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
    
    
    def add_paragrap_information(self,para_tokenized,para_sent_lengths,question_worthiness):
        para=Paragraph()
        para.paragraph=self.convert_to_index(para_tokenized)
        para.sentence_lengths=para_sent_lengths
        para.question_worthiness=question_worthiness
        self.data.append(para)

    def convert_to_index(self,list_of_words):
        return_list=[]
        for word in list_of_words:
            return_list.append(self.word2idx.get(word.lower(),self.word2idx['UNK']))
        return return_list

    def create_train_val_test(self):
        len_para=len(self.data)
        marker=int(0.1*len_para)
        self.train_data=self.data[:8*marker]
        self.val_data=self.data[8*marker:9*marker]
        self.test_data=self.data[9*marker:]

        
