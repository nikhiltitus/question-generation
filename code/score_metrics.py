from __future__ import division
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


def compute_bleu(string_a,string_b):
    bleu_1=sentence_bleu(string_a,string_b,weights=(1,0,0,0))
    bleu_2=sentence_bleu(string_a,string_b,weights=(0.5,0.5,0,0))
    bleu_3=sentence_bleu(string_a,string_b,weights=(1/3,1/3,1/3,0))
    bleu_4=sentence_bleu(string_a,string_b,weights=(0.25,0.25,0.25,0.25))
    bleu_overall=sentence_bleu(string_a,string_b)
    return bleu_1,bleu_2,bleu_3,bleu_4,bleu_overall

def bleu_score_estimator(**kwargs):
    """
    This computes the bleu score. You can give the following inputs to this
    input_string= source string
    candidate_string=candidate string
    input_list=source list of strings
    candidate_list=candidate list of strings
    verbose=True or False
    """
    input_string=kwargs.get('input_string',None)
    candidate_string=kwargs.get('candidate_string',None)
    input_list=kwargs.get('input_list',None)
    candidate_list=kwargs.get('candidate_list',None)
    verbose=kwargs.get('verbose',False)
    if input_string and candidate_string:
        return compute_bleu(input_string,candidate_string)
    if input_list and candidate_list:
        total_length=len(candidate_list)
        bleu_scores_iterationwise=[]
        for i,string in enumerate(input_list):
            if i%100==0 and verbose:
                print "Bleu Computation :: processed :{0} remaining :{1}".format(i,total_length-i)
            bleu_1,bleu_2,bleu_3,bleu_4,bleu_overall=compute_bleu(string,candidate_list[i])
            bleu_scores_iterationwise.append((bleu_1,bleu_2,bleu_3,bleu_4,bleu_overall))
        return np.mean(bleu_scores_iterationwise,axis=0)