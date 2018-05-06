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

def write_output(para_question_list,output_location):
    with open(output_location,'w') as output_file:
        for i in range(0,len(para_question_list)):
            output_file.write('Paragraph: '+str(i)+'\n')
            for question in para_question_list[i]:
                output_file.write(question)+'\n'

def main():
    parser=OptionParser()
    parser.add_option('--para_file',dest='para_file',help='paragraph file location')
    parser.add_option('--question_file',dest='question_file',help='question file location')
    parser.add_option('--output_file',dest='output_file',help='output file location')
    options,args=parser.parse_args()
    no_of_questions=get_no_questions(options.para_file)#List of numbers
    questions=get_questions(options.question_file)#List of questions
    para_question_list=[]
    index=0
    for i in range(0,len(no_of_questions)):
        para_question_list.append(questions[index:index+no_of_questions[i]])
        index+=no_of_questions[i]
    write_output(para_question_list,options.output_file)
    

main()