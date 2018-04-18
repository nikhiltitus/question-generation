import numpy as np 
import nltk 
import pickle
import re
from QuestionAnswers import QuestionAnswers 
import score_metrics as sm

file_location='../data/squad/qa_dump'



class DirectIn():
    def __init__(self,file_location):
        with open(file_location) as dump_file:
            data=pickle.load(dump_file)
            print len(data)
            self.train,self.val,self.test=data
    def train_model(self,verbose=False):
        questions=[]
        answers=[]
        total_length=len(self.train)
        i=0
        for i,qa in enumerate(self.train):
            if i%1000==0 and verbose:
                print "processed :{0} remaining :{1}".format(i,total_length-i)
            split_answer=max(re.split("[',','.','!','?']",qa.answer),key=len)
            #print split_answer
            questions.append(qa.question)
            answers.append(split_answer)
        print "calculating_bleu_scores"
        return sm.bleu_score_estimator(input_list=questions,candidate_list=answers,verbose=True)
            


if __name__=="__main__":
    baseline=DirectIn(file_location)
    print baseline.train_model(verbose=True)