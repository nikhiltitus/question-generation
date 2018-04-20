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
    bleu_scores_iterationwise=[]
    if input_string and candidate_string:
        return compute_bleu(input_string,candidate_string)
    if input_list and candidate_list:
        total_length=len(candidate_list)
        bleu_total_score_1=0
        bleu_total_score_2=0
        bleu_total_score_3=0
        bleu_total_score_4=0
        bleu_total_score_default=0
        for i,string in enumerate(input_list):
            if i%100==0 and verbose:
                print "Bleu Computation :: processed :{0} remaining :{1}".format(i,total_length-i)
                print "Current Bleu Scores : {0}".format(np.mean(bleu_scores_iterationwise,axis=0))
            bleu_1,bleu_2,bleu_3,bleu_4,bleu_overall=compute_bleu(string,candidate_list[i])
            bleu_total_score_1+=bleu_1
            bleu_total_score_2+=bleu_2
            bleu_total_score_3+=bleu_3
            bleu_total_score_4+=bleu_4
            bleu_total_score_default+=bleu_overall
            bleu_scores_iterationwise.append((bleu_1,bleu_2,bleu_3,bleu_4,bleu_overall))
        return (bleu_total_score_1/total_length,bleu_total_score_2/total_length,
                bleu_total_score_3/total_length,bleu_total_score_4/total_length,
                bleu_total_score_default/total_length)
    