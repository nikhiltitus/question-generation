import pdb
from optparse import OptionParser

def get_no_questions(para_file_location):
    with open(para_file_location) as para_file:
        paragraphs=para_file.readlines()
    no_of_questions=[int(par.split('|')[1]) for par in paragraphs]
    return no_of_questions

def get_questions(question_file_location):
    with open(question_file_location) as ques_file:
        questions=ques_file.readlines()
    return questions

def main():
    parser=OptionParser()
    parser.add_option('--para_file',dest='para_file',help='The paragraph file')
    parser.add_option('--question_file',dest='question_file',help='The question file')
    options,args=parser.parse_args()
    pdb.set_trace()
    no_of_questions=get_no_questions(options.para_file)#List of numbers
    questions=get_questions(options.question_file)#List of questions
    para_question_list=[]
    index=0
    for i in range(0,len(no_of_questions)):
        para_question_list.append(questions[index:index+no_of_questions[i]])
        index+=no_of_questions[i]
    pdb.set_trace()