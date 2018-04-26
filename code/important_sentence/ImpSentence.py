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

    def forward(self,paragraph_variable):
        pdb.set_trace()
        embedding=self.embedding_layer(paragraph_variable)
        pdb.set_trace()

def convert_paralists_to_ids(paragraph_list):
    output_paragraph=[]
    for paragraph in paragraph_list:
        output_paragraph.append(sentence_list_ids)
def main():
    vocab_map={'i':0,'am':1,'nikhil':2,'an':3,'amateur':4,'guitar':5,'player':6,'stop':7}
    paragraph_list=[]
    paragraph=Paragraph()
    paragraph.set_sentence_list([['i','am','nikhil'],['i','am','an','amateur','guitar','player']],vocab_map)
    paragraph_list.append(paragraph)
    impModel=ImpSentenceModel(2,50,20)
    inp_tensor=torch.LongTensor([ [[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]] ])
    inp_tensor=inp_tensor.view(inp_tensor.shape[0],inp_tensor.shape[1]*inp_tensor.shape[2])
    impModel(autograd.Variable(inp_tensor))

main()