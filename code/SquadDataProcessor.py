from QuestionAnswers import QuestionAnswers
import json
import pdb
import pickle
import numpy as np
from utils import get_answer_sentence,check_overlap,sentence_selection_processing
from Paragraph import squad_data
import utils as utils
from optparse import OptionParser

class SquadDataProcessor:
    def __init__(self):
        question_answer_list=[]
    
    def process_data(self,squad_json,remove_nonoverlap=False,**kwargs):
        qans_list=[]
        dataset_json=squad_json['data']
        encode_format=kwargs.get('encoding_format','utf-8')
        i=0
        for paragraph in dataset_json:
            paragraph_json=paragraph['paragraphs']
            for context in paragraph_json:
                context_json=context['context']
                qas_json=context['qas']
                for quesans in qas_json:
                    question=quesans['question']
                    answer=quesans['answers'][0]
                    answer_sentence=get_answer_sentence(context_json,answer['answer_start'],answer['text'])
                    if i%1000==0:
                        print i
                    i+=1
                    if remove_nonoverlap and check_overlap(answer_sentence,question)==0:
                        continue
                    question=question.replace('\n',' ').replace('\r',' ').encode(encode_format).strip()
                    answer_sentence=answer_sentence.replace('\n',' ').replace('\r',' ').encode(encode_format).strip()
                    qans_list.append(qans)
        offset=int(len(qans_list)*0.8)
        test_offset=offset+int(len(qans_list)*0.1)
        train_list=qans_list[:offset]
        val_list=qans_list[offset:test_offset]
        test_list=qans_list[test_offset:]            
        return (train_list,val_list,test_list)

    def important_text_selection(self,squad_json):
        squad_data_return=squad_data()
        dataset_json=squad_json['data']
        i=0
        max_length_para_length=0
        max_para_length=0
        for paragraph in dataset_json:
            paragraph_json=paragraph['paragraphs']
            for context in paragraph_json:
                context_json=context['context']
                tokenized_context=utils.tokenize_para(context_json)
                para_sent_lengths=utils.find_para_lengths(context_json)
                qas_json=context['qas']
                
                if len(tokenized_context)>max_para_length:
                    max_para_length=len(tokenized_context)
                    print max_para_length

                if len(para_sent_lengths)>max_length_para_length:
                    max_length_para_length=len(para_sent_lengths)
                    print max_length_para_length

                question_worthiness=np.zeros((len(qas_json),len(para_sent_lengths)),dtype=np.int32)
                for j,quesans in enumerate(qas_json):
                    i+=1
                    if i%1000==0:
                        print i
                    question=quesans['question']
                    answer=quesans['answers'][0]
                    answer_sentence=get_answer_sentence(context_json,answer['answer_start'],answer['text'])
                    #if  check_overlap(answer_sentence,question)==0:
                    #    continue
                    question_worthiness[j]=sentence_selection_processing(context_json,
                                                    answer['answer_start'],
                                                    answer['text'])
                worthiness=np.array((np.sum(question_worthiness,axis=0)>0)*1,dtype=np.int32)
                squad_data_return.add_paragrap_information(tokenized_context,
                                                        para_sent_lengths,worthiness)
                
        squad_data_return.max_sent_length=max_length_para_length
        squad_data_return.max_par_length=max_para_length

        squad_data_return.create_train_val_test()
        return squad_data_return

    def read_squad(self,file_location):
        with open(file_location) as squad_file:
            squad_json=json.load(squad_file)
        return squad_json

def preprocess_and_save(file_location,dump_file_location,remove_nonoverlap=False):
    sqad_preprocessor=SquadDataProcessor()
    squad_data=sqad_preprocessor.read_squad(file_location)
    train,val,test=sqad_preprocessor.process_data(squad_data,
                            remove_nonoverlap=remove_nonoverlap,encoding_format='utf-8')
    with open(dump_file_location,'wb') as dump_file:
        pickle.dump((train,val,test),dump_file)

def preprocess_and_save_text_selection(file_location,dump_file_location,remove_nonoverlap=False):
    dump_file_location=dump_file_location+'qa_dump'
    sqad_preprocessor=SquadDataProcessor()
    squad_data=sqad_preprocessor.read_squad(file_location)
    text_selection_data=sqad_preprocessor.important_text_selection(squad_data)
    with open(dump_file_location,'wb') as dump_file:
        pickle.dump(text_selection_data,dump_file)

def process_and_save_for_nmt(file_location,dest_folder):
    sqad_preprocessor=SquadDataProcessor()
    squad_data=sqad_preprocessor.read_squad(file_location)
    train_list,val_list,test_list=sqad_preprocessor.process_data(squad_data,remove_nonoverlap=True)
    # pdb.set_trace()
    with open(dest_folder+'train-src.txt','w') as trainsrc:
        with open(dest_folder+'train-tgt.txt','w') as traintgt:
            for qans in train_list:
                trainsrc.write(qans.answer+'\n')
                traintgt.write(qans.question+'\n')
    with open(dest_folder+'val-src.txt','w') as valsrc:
        with open(dest_folder+'val-tgt.txt','w') as valtgt:
            for qans in val_list:
                valsrc.write(qans.answer+'\n')
                valtgt.write(qans.question+'\n')
    with open(dest_folder+'test-src.txt','w') as testsrc:
        with open(dest_folder+'test-tgt.txt','w')as testgt:
            for qans in test_list:
                testsrc.write(qans.answer+'\n')
                testgt.write(qans.question+'\n')


if __name__=='__main__':
        parser=OptionParser()
        parser.add_option('-i','--input',default='../data/squad/train-v1.1.json',dest='input_location',help='input squad json location')
        parser.add_option('-o','--output',default='../out/',dest='output_location',help='output folder location')
        parser.add_option("-m", "--mode", default='2',dest='mode' ,help="mode 1 for training nmt model, mode 2 for text selection model [default: %default]")
        options, args = parser.parse_args()
        if options.mode == '1':
            print 'Preprocessing squad for Open NMT'
            process_and_save_for_nmt(options.input_location,options.output_location)
        elif options.mode == '2':
            print 'Preprocessing for the text selection model'
            preprocess_and_save_text_selection(options.input_location,options.output_location,remove_nonoverlap=True)

