from QuestionAnswers import QuestionAnswers
import json
import pdb
from utils import get_answer_sentence

class SquadDataProcessor:
    def __init__(self):
        question_answer_list=[]
    
    def process_data(self,squad_json):
        qans_list=[]
        dataset_json=squad_json['data']
        for paragraph in dataset_json:
            paragraph_json=paragraph['paragraphs']
            for context in paragraph_json:
                context_json=context['context']
                qas_json=context['qas']
                for quesans in qas_json:
                    question=quesans['question']
                    answer=quesans['answers'][0]
                    # pdb.set_trace()
                    answer=get_answer_sentence(context_json,answer['answer_start'],answer['text'])
                    qans=QuestionAnswers(question,answer)
                    qans_list.append(qans)
                    print question
                    print answer
        return qans_list

    def read_squad(self,file_location):
        with open(file_location) as squad_file:
            squad_json=json.load(squad_file)
        return squad_json

def main():
    preprocessor=SquadDataProcessor()
    squad_json = preprocessor.read_squad('../data/squad/train-v1.1.json')
    preprocessor.process_data(squad_json)
main()