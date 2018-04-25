from QuestionAnswers import QuestionAnswers
import json
import pdb
import pickle
from utils import get_answer_sentence,check_overlap


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
                    # pdb.set_trace()
                    answer_sentence=get_answer_sentence(context_json,answer['answer_start'],answer['text'])
                    if i%1000==0:
                        print i
                    i+=1
                    if remove_nonoverlap and check_overlap(answer_sentence,question)==0:
                        #print question
                        #print answer_sentence,answer['text']
                        continue
                    question=question.replace('\n',' ').replace('\r',' ').encode(encode_format).strip()
                    answer_sentence=answer_sentence.replace('\n',' ').replace('\r',' ').encode(encode_format).strip()
                    # question=question.replace('\n',' ').replace('\r',' ').strip()
                    # answer_sentence=answer_sentence.replace('\n',' ').replace('\r',' ').strip()
                    qans=QuestionAnswers(question,answer_sentence,context_json)
                    qans_list.append(qans)
        offset=int(len(qans_list)*0.8)
        test_offset=offset+int(len(qans_list)*0.1)
        train_list=qans_list[:offset]
        val_list=qans_list[offset:test_offset]
        test_list=qans_list[test_offset:]            
        return (train_list,val_list,test_list)

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
         preprocess_and_save('../data/squad/train-v1.1.json','../data/squad/qa_dump',
                             remove_nonoverlap=True)
        # process_and_save_for_nmt('../data/squad/train-v1.1.json','../out/')
        #process_and_save_for_nmt('../data/squad/train-v1.1.json','../out/')
