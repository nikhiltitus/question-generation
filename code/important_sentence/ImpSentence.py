import torch.nn as nn
import pdb
import torch
import torch.autograd as autograd
from Paragraph import Paragraph
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#Before running export PYTHONPATH=/Users/nikhiltitus/acads/anlp/project/question-generation/code:/Users/nikhiltitus/acads/anlp/project/question-generation/code/important_sentence
class ImpSentenceModel(nn.Module):
    def __init__(self,mini_batch_size,embedding_dim,vocab_size,hidden_dim):
        super(ImpSentenceModel,self).__init__()
        self.hidden_dim=hidden_dim
        self.mini_batch_size=mini_batch_size
        self.embedding_dim=embedding_dim
        self.embedding_layer=nn.Embedding(vocab_size,embedding_dim)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim)
        self.hidden=self.init_hidden(mini_batch_size)
    
    def init_hidden(self,mini_batch_length):
        self.hidden=(autograd.Variable(torch.zeros(1, mini_batch_length, self.hidden_dim)),autograd.Variable(torch.zeros(1, mini_batch_length, self.hidden_dim)))

    def forward(self,paragraph_variable,sentence_length_list,paragh_length_list,max_no_lines):
        pdb.set_trace()
        embedding=self.embedding_layer(paragraph_variable)
        line_embedding=autograd.Variable(torch.zeros(self.mini_batch_size,max_no_lines,self.embedding_dim))
        for i in range(0,self.mini_batch_size):
            counter=0
            previous=0
            for j in sentence_length_list[i]:
                line_embedding[i,counter]=embedding[i,previous: previous + j].sum(dim=0)
                counter+=1
                previous+=j
        line_embedding=line_embedding.transpose(0,1)
        line_embedding=pack_padded_sequence(line_embedding,paragh_length_list)
        packed_lstm_out,self.hidden=self.lstm(line_embedding,self.hidden)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out)
        pdb.set_trace()

def main():
    vocab_map={'i':0,'am':1,'nikhil':2,'an':3,'amateur':4,'guitar':5,'player':6,'stop':7}
    paragraph_list=[]
    paragraph=Paragraph()
    paragraph.set_sentence_list([['i','am','nikhil'],['i','am','an','amateur','guitar','player']],vocab_map)
    paragraph_list.append(paragraph)
    impModel=ImpSentenceModel(3,50,20,128)
    inp_tensor=torch.LongTensor([ [1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18] ])
    impModel(autograd.Variable(inp_tensor),[[2,4],[5,1],[1,5]],[2,2,2],2)

main()