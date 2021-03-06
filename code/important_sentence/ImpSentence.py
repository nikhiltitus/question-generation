import torch.nn as nn
import pdb
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import sys
import pickle
import numpy as np
import torch
sys.path.append( '/media/albert/Albert Bonu/Studies/CS 690N/Project/work/question-generation/code')
sys.path.append( '../')
sys.path.append('/home/nikhilgeorge/question-generation/code')


from Paragraph import squad_data

#Global variables
batch_count=0
processed_data=None
data_size=0
#input_file_path=None
Model_save_path=None
enable_cuda=None
index_to_word=None


def retrieve_data(file_location):
    global processed_data
    with open(file_location) as file:
        processed_data=pickle.load(file)

def create_batches(batch_size,mode='train'):
    global processed_data,batch_count,data_size,index_to_word
    if not processed_data:
        print "Retrieving"
        retrieve_data(input_file_path)
        data_size=len(processed_data.train_data)
    max_count=data_size//batch_size
    if not batch_count<max_count:
        batch_count=0
    # we take a batch of data
    index_to_word=processed_data.index2word
    if mode=='train':
        return_data=processed_data.train_data[batch_count*batch_size:(batch_count+1)*batch_size]
    elif mode=='val':
        if batch_size==-1:
            return_data=processed_data.val_data
        else :
            return_data=np.random.choice(processed_data.val_data,batch_size,replace=False)
    else:
        if batch_size==-1:
            return_data=processed_data.test_data
        else:
            return_data=processed_data.test_data[:batch_size]

    data_batch=return_data
    sorted_data_batch=[]
    for i,para in enumerate(data_batch):
        
        sorted_data_batch.append((para.paragraph,para.sentence_lengths,
                            para.question_worthiness,len(para.sentence_lengths)))

    sorted_data_batch=sorted(sorted_data_batch,key=lambda tup:-tup[3])
    paragraph_list=[]
    para_sentence_lengths=[]
    para_question_worthiness=[]
    paragraph_line_length=[]
    for element in sorted_data_batch:
        para_element=element[0]
        while len(para_element)<processed_data.max_par_length:
            para_element.append(0)
        paragraph_list.append(element[0])
        para_sentence_lengths.append(element[1])
        para_question_worthiness+=list(element[2])
        paragraph_line_length.append(element[3])
    print "MAX lengths : ",processed_data.max_par_length,processed_data.max_sent_length
    if mode=='train':
        batch_count+=1
    return paragraph_list,para_sentence_lengths,para_question_worthiness,paragraph_line_length
        




class ImpSentenceModel(nn.Module):
    def __init__(self,mini_batch_size,embedding_dim,vocab_size,hidden_dim,max_no_lines):
        global enable_cuda
        super(ImpSentenceModel,self).__init__()
        self.max_no_lines=max_no_lines
        self.hidden_dim=hidden_dim
        self.embedding_dim=embedding_dim
        self.embedding_layer=nn.Embedding(vocab_size,embedding_dim)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(processed_data.weights))
        self.lstm=nn.LSTM(embedding_dim,hidden_dim,bidirectional=True)
        self.hidden=self.init_hidden(mini_batch_size)
        self.relu_layer=nn.ReLU()
        self.linear=nn.Linear(2*hidden_dim,100)
        self.linear_2=nn.Linear(100,50)
        self.relu_layer_2=nn.ReLU()
        self.linear_3=nn.Linear(50,2)
    
    def init_hidden(self,mini_batch_size):
        global enable_cuda
        if enable_cuda:
            print 'In init hidden'
            self.hidden=(autograd.Variable(torch.cuda.FloatTensor(2, mini_batch_size, self.hidden_dim).fill_(0)),autograd.Variable(torch.cuda.FloatTensor(2, mini_batch_size, self.hidden_dim).fill_(0)))
        else:
            self.hidden=(autograd.Variable(torch.zeros(2, mini_batch_size, self.hidden_dim)),autograd.Variable(torch.zeros(2, mini_batch_size, self.hidden_dim)))

    def forward(self,mini_batch_size,paragraph_variable,sentence_lens,no_of_lines):
        # pdb.set_trace()
        self.init_hidden(mini_batch_size)
        global enable_cuda
        no_of_sentence=0
        embedding=self.embedding_layer(paragraph_variable)
        if enable_cuda:
            line_embedding=autograd.Variable(torch.cuda.FloatTensor(mini_batch_size,self.max_no_lines,self.embedding_dim).fill_(0))
        else:
            line_embedding=autograd.Variable(torch.zeros(mini_batch_size,self.max_no_lines,self.embedding_dim))
        for i in range(0,mini_batch_size):
            counter=0
            previous=0
            for j in sentence_lens[i]:
                no_of_sentence+=1
                line_embedding[i,counter]=embedding[i,previous: previous + j].sum(dim=0)
                counter+=1
                previous+=j
        line_embedding=line_embedding.transpose(0,1)
        line_embedding=pack_padded_sequence(line_embedding,no_of_lines)
        packed_lstm_out,self.hidden=self.lstm(line_embedding,self.hidden)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out)
        lstm_out=lstm_out.transpose(0,1)
        if enable_cuda:
            sentence_lstm=autograd.Variable(torch.cuda.FloatTensor(no_of_sentence,lstm_out.shape[2]).fill_(0))
        else:
            sentence_lstm=autograd.Variable(torch.zeros(no_of_sentence,lstm_out.shape[2]))
        counter=0
        for i in range(0,mini_batch_size):
            for j,element in enumerate(sentence_lens[i]):
                sentence_lstm[counter]=lstm_out[i,j]
                counter+=1
        output_1=self.relu_layer(self.linear(sentence_lstm))
        output_2=self.relu_layer_2(self.linear_2(output_1))
        output=self.linear_3(output_2)
        return output
        # pdb.set_trace()


def get_accuracy(out_scores,target_scores):
    return np.mean(np.argmax(out_scores.data.cpu().numpy(),axis=1) == target_scores.data.cpu().numpy())

def get_val_accuracy(model):
    global enable_cuda
    val_p_list,val_sentence_lens,val_ques_worthy,val_n_line=create_batches(128,'val')
    if enable_cuda:
        paragraph_input=autograd.Variable(torch.cuda.LongTensor(val_p_list))
    else:
        paragraph_input=autograd.Variable(torch.LongTensor(val_p_list))
    # paragraph_input=autograd.Variable(torch.LongTensor(p_list))
    if enable_cuda:
        target_scores=autograd.Variable(torch.cuda.LongTensor(val_ques_worthy))
    else:
        target_scores=autograd.Variable(torch.LongTensor(val_ques_worthy))
    model.zero_grad()
    out_scores=model(128,paragraph_input,val_sentence_lens,val_n_line)
    accuracy=get_accuracy(out_scores,target_scores)
    return accuracy

def write_results_as_text(p_list,sentence_lens,no_of_lines,out,output_location):
    global index_to_word
    text_par=[]
    text_lines=[]
    par_lines=[]
    count=0
    for i in range(0,len(p_list)):
        index=0
        par_lines=[]
        for j in sentence_lens[i]:
            line=p_list[i][index:index+j]
            text_lines=[index_to_word[it] for it in line]
            par_lines.append(text_lines)
            index+=j
            count+=1
        text_par.append(par_lines)
    print count
    #Questions mapped to corresponding paragraphs
    counter=0
    ques_par=[]
    for i in range(0,len(no_of_lines)):
        questions=[]
        for j in range(0,no_of_lines[i]):
            print counter
            if out[counter]==1:
                questions.append(text_par[i][j])
            counter+=1
        ques_par.append(questions)
    with open(output_location+'paragraphs','w') as parafile:
        with open(output_location+'worthy_sentences','w') as worthyfile:
            counter=0
            for i in range (0,len(text_par)):
                sentence=' '
                for j in range(0,len(text_par[i])):
                    sentence=sentence+' '.join(text_par[i][j])
                parafile.write(str(counter)+'|'+str(len(ques_par[i]))+'|'+sentence+'\n')
                counter+=1
            for i in range(0,len(ques_par)):
                for j in range(0,len(ques_par[i])):
                    sentence=' '.join(ques_par[i][j])
                    worthyfile.write(sentence+'\n')
    print 'File writes complete'
    
def main3():
    MINI_BATCH_SIZE=128
    print '----------PROGRAM STARTING------------------------'
    global input_file_path,Model_save_path,enable_cuda
    enable_cuda=sys.argv[3] == 'TRUE'
    if enable_cuda:
        print 'CUDA enabled'
    input_file_path=sys.argv[1]
    Model_save_path=sys.argv[2]
    running_accuracy,running_loss=[],[]
    no_of_epochs,epoch_count=20,0
    loss_function=nn.CrossEntropyLoss()
    p_list,sentence_lens,ques_worthy,n_line=create_batches(128)
    max_no_sentences,max_no_of_words=processed_data.max_sent_length,processed_data.max_par_length
    if enable_cuda:
        impModel=ImpSentenceModel(MINI_BATCH_SIZE,300,48006,128,max_no_sentences).cuda()
    else:
        impModel=ImpSentenceModel(MINI_BATCH_SIZE,300,48006,128,max_no_sentences)
    optimizer = optim.SGD(impModel.parameters(), lr=0.1)
    while True:
        print batch_count
        if (len(p_list) !=128 ):
            print 'Batch size issue'
            continue
        if batch_count == 1 and len(running_loss) != 0:
            torch.save(impModel, Model_save_path)
            epoch_count+=1
            print 'No of epoch: ',epoch_count
            print 'Running training accuracy %f'%(sum(running_accuracy)/len(running_accuracy))
            print 'Running Loss %f'%(sum(running_loss)/len(running_loss))
            val_acc=get_val_accuracy(impModel)
            # pdb.set_trace()
            print 'Validation accuracy: %f'%(val_acc)
            running_accuracy=[]
            running_loss=[]
        if epoch_count == no_of_epochs:
            print 'Max epochs reached'
            break
        print ('Batch count is: %d of %d'%(batch_count,data_size//128))
        if enable_cuda:
            paragraph_input=autograd.Variable(torch.cuda.LongTensor(p_list))
            target_scores=autograd.Variable(torch.cuda.LongTensor(ques_worthy))
        else:
            paragraph_input=autograd.Variable(torch.LongTensor(p_list))
            target_scores=autograd.Variable(torch.LongTensor(ques_worthy))
        impModel.zero_grad()
        out_scores=impModel(MINI_BATCH_SIZE,paragraph_input,sentence_lens,n_line)
        accuracy=get_accuracy(out_scores,target_scores)
        loss=loss_function(out_scores, autograd.Variable(target_scores))
        loss.backward()
        optimizer.step()
        p_list,sentence_lens,ques_worthy,n_line=create_batches(128)
        print 'Loss: ',loss.data
        print 'accuracy: ',accuracy
        running_loss.append(float(loss.data.cpu()))
        running_accuracy.append(accuracy)

def test_inputs():
    global batch_count
    print len(create_batches(100,'train')[0])
    print len(create_batches(100,'test')[0])
    print len(create_batches(100,'val')[0])
    print len(processed_data.data)
    print len(processed_data.train_data)
    print len(processed_data.test_data)
    print len(processed_data.val_data)
    #checking if all baches work well
    prev_count=0
    while  batch_count>prev_count:
        print len(create_batches(100,'train')[0])
        print batch_count
        prev_count+=1

def main4():
    global input_file_path,enable_cuda
    input_file_path=sys.argv[1]
    model_load_path=sys.argv[2]
    enable_cuda=sys.argv[3]
    output_location=sys.argv[4]
    p_list,sentence_lens,ques_worthy,n_line=create_batches(-1,'val')
    impModel=torch.load(model_load_path, map_location=lambda storage, loc: storage)
    impModel=impModel.cuda()
    impModel.zero_grad()
    if enable_cuda:
        paragraph_input=autograd.Variable(torch.cuda.LongTensor(p_list))
        target_scores=autograd.Variable(torch.cuda.LongTensor(ques_worthy))
    else:
        paragraph_input=autograd.Variable(torch.LongTensor(p_list))
        target_scores=autograd.Variable(torch.LongTensor(ques_worthy))
    out_scores=impModel(len(p_list),paragraph_input,sentence_lens,n_line)
    accuracy=get_accuracy(out_scores,target_scores)
    print accuracy
    output=np.argmax(out_scores.data.cpu().numpy(),axis=1)
    pdb.set_trace()
    write_results_as_text(p_list,sentence_lens,n_line,output,output_location)

# main4()
main3()
