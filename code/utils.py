import nltk
import numpy as np

def get_answer_sentence(context,ans_start_index,answer):
    sentence_list=[]
    running_count=0
    end_index=ans_start_index+len(answer)
    tok_context=nltk.sent_tokenize(context)
    for i in range(0,len(tok_context)):
        current_count=running_count+len(tok_context[i])
        if current_count >= ans_start_index:
            sentence_list.append(tok_context[i])
        if current_count > end_index:
            break
        running_count=current_count
    return ' '.join(sentence_list)

def sentence_selection_processing(context,ans_start_index,answer):
    sentence_list=[]
    running_count=0
    end_index=ans_start_index+len(answer)
    tok_context=nltk.sent_tokenize(context)
    question_worthiness=[]
    word_count=0
    for i in range(0,len(tok_context)):
        current_count=running_count+len(tok_context[i])
        if current_count >= ans_start_index and current_count<end_index:
            sentence_list.append(tok_context[i])
            question_worthiness.append(1)
        else:
            question_worthiness.append(0)
        running_count=current_count
    return np.array(question_worthiness)

def find_para_lengths(context):
    para_lengths=[]
    tok_context=nltk.sent_tokenize(context)
    for i,sent in enumerate(tok_context):
        para_lengths.append(len(nltk.word_tokenize(sent)))
    return para_lengths

def tokenize_para(context):
    return nltk.word_tokenize(context)

def check_overlap(sentence_1,sentence_2):
    bow_sent1=set(sentence_1.split())
    bow_sent2=set(sentence_2.split())
    return len(bow_sent1.intersection(bow_sent2))


def main():
    context='Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'
    ans_start_index=515
    answer='Saint Bernadette Soubirous'
    print(get_answer_sentence(context,ans_start_index,answer))

