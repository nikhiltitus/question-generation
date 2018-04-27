import torch.nn as nn
import pdb
import torch
import torch.autograd as autograd
import torch.optim as optim
from Paragraph import Paragraph
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#Before running export PYTHONPATH=/Users/nikhiltitus/acads/anlp/project/question-generation/code:/Users/nikhiltitus/acads/anlp/project/question-generation/code/important_sentence
class ImpSentenceModel(nn.Module):
    def __init__(self,mini_batch_size,embedding_dim,vocab_size,hidden_dim,max_no_lines):
        super(ImpSentenceModel,self).__init__()
        self.max_no_lines=max_no_lines
        self.hidden_dim=hidden_dim
        self.mini_batch_size=mini_batch_size
        self.embedding_dim=embedding_dim
        self.embedding_layer=nn.Embedding(vocab_size,embedding_dim)
        self.lstm=nn.LSTM(embedding_dim,hidden_dim)
        self.hidden=self.init_hidden()
        self.linear=nn.Linear(hidden_dim,2)
    
    def init_hidden(self):
        self.hidden=(autograd.Variable(torch.zeros(1, self.mini_batch_size, self.hidden_dim)),autograd.Variable(torch.zeros(1, self.mini_batch_size, self.hidden_dim)))

    def forward(self,paragraph_variable,sentence_length_list,paragh_length_list):
        # pdb.set_trace()
        no_of_sentence=0
        embedding=self.embedding_layer(paragraph_variable)
        line_embedding=autograd.Variable(torch.zeros(self.mini_batch_size,self.max_no_lines,self.embedding_dim))
        for i in range(0,self.mini_batch_size):
            counter=0
            previous=0
            for j in sentence_length_list[i]:
                no_of_sentence+=1
                line_embedding[i,counter]=embedding[i,previous: previous + j].sum(dim=0)
                counter+=1
                previous+=j
        line_embedding=line_embedding.transpose(0,1)
        line_embedding=pack_padded_sequence(line_embedding,paragh_length_list)
        packed_lstm_out,self.hidden=self.lstm(line_embedding,self.hidden)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out)
        lstm_out=lstm_out.transpose(0,1)
        sentence_lstm=autograd.Variable(torch.zeros(no_of_sentence,lstm_out.shape[2]))
        counter=0
        for i in range(0,self.mini_batch_size):
            for j,element in enumerate(sentence_length_list[i]):
                sentence_lstm[counter]=lstm_out[i,j]
                counter+=1
        output=self.linear(sentence_lstm)
        return output
        # pdb.set_trace()

def main():
    loss_function=nn.CrossEntropyLoss()
    target=torch.LongTensor([1,0,1,0,1,1])
    impModel=ImpSentenceModel(3,50,20,128,2)
    optimizer = optim.SGD(impModel.parameters(), lr=0.1)
    inp_tensor=torch.LongTensor([ [1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18] ])
    # impModel(autograd.Variable(inp_tensor),[[2,4],[5,1],[1,5]],[2,2,2])
    print '-------IN--------------'
    for epoch in range(10):
        impModel.zero_grad()
        impModel.init_hidden()
        out_scores=impModel(autograd.Variable(inp_tensor),[[2,4],[5,1],[1,5]],[2,2,2])
        # pdb.set_trace()
        loss=loss_function(out_scores, autograd.Variable(target))
        # pdb.set_trace()
        loss.backward()
        # pdb.set_trace()
        optimizer.step()
        # pdb.set_trace()
        print('Loss: ',loss.data)

main()