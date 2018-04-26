import torch.nn as nn
import pdb
import torch
import torch.autograd as autograd
from Paragraph import Paragraph

#Before running export PYTHONPATH=/Users/nikhiltitus/acads/anlp/project/question-generation/code:/Users/nikhiltitus/acads/anlp/project/question-generation/code/important_sentence
class ImpSentenceModel(nn.Module):
    def __init__(self,mini_batch_size,embedding_dim,vocab_size):
        super(ImpSentenceModel,self).__init__()
        self.mini_batch_size=mini_batch_size
        self.embedding_dim=embedding_dim
        self.embedding_layer=nn.Embedding(vocab_size,embedding_dim)

    def forward(self,paragraph_variable,sentence_length_list,max_no_lines):
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
        pdb.set_trace()

def main():
    vocab_map={'i':0,'am':1,'nikhil':2,'an':3,'amateur':4,'guitar':5,'player':6,'stop':7}
    paragraph_list=[]
    paragraph=Paragraph()
    paragraph.set_sentence_list([['i','am','nikhil'],['i','am','an','amateur','guitar','player']],vocab_map)
    paragraph_list.append(paragraph)
    impModel=ImpSentenceModel(2,50,20)
    inp_tensor=torch.LongTensor([ [1,2,3,4,5,6],[7,8,9,10,11,12] ])
    impModel(autograd.Variable(inp_tensor),[[2,4],[5,1]],2)

main()