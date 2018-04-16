from QuestionAnswers import QuestionAnswers
import json
import pdb
import pickle
from utils import get_answer_sentence,check_overlap


class SquadDataProcessor:
    def __init__(self):
        question_answer_list=[]
    
    def process_data(self,squad_json,remove_nonoverlap=False):
        qans_list=[]
        dataset_json=squad_json['data']
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
                    qans=QuestionAnswers(question,answer_sentence,context_json)
                    qans_list.append(qans)
                    
        return qans_list

    def read_squad(self,file_location):
        with open(file_location) as squad_file:
            squad_json=json.load(squad_file)
        return squad_json

def preprocess_and_save(file_location,dump_file_location,remove_nonoverlap=False):
    sqad_preprocessor=SquadDataProcessor()
    #checking=open(dump_file_location,'wb')
    squad_data=sqad_preprocessor.read_squad(file_location)
    data=sqad_preprocessor.process_data(squad_data,remove_nonoverlap=remove_nonoverlap)
    with open(dump_file_location,'wb') as dump_file:
        pickle.dump(data,dump_file)

if __name__=='__main__':
        preprocess_and_save('../data/squad/train-v1.1.json','../data/squad/qa_dump',
                            remove_nonoverlap=True)
