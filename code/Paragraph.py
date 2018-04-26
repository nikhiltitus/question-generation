# from nltk.tokenize import sent_tokenize
# from nltk.tokenize import word_tokenize
# para=sent_tokenize(inp)
# for i in range(0,len(para)):
#     para[i]=word_tokenize(para[i])


class Paragraph:
    def __init__(self):
        self.sentence_list=[]
        self.sentence_list_ids=[]

    def set_sentence_list(self,sentence_list,vocab_map):
        self.sentence_list=sentence_list
        self.sentence_list_ids=self.get_as_ids(vocab_map)
    
    def get_as_ids(self,vocab_map):
        paragraph_ids=[]
        for sentence in self.sentence_list:
            sentence_ids=[]
            for word in sentence:
                sentence_ids.append(vocab_map[word])
            paragraph_ids.append(sentence_ids)
        return paragraph_ids
