import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

#Model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question = '''Why was the student group called "the Methodists?"'''

paragraph = ''' The movement which would become The United Methodist Church began in the mid-18th century within the Church of England.
            A small group of students, including John Wesley, Charles Wesley and George Whitefield, met on the Oxford University campus.
            They focused on Bible study, methodical study of scripture and living a holy life.
            Other students mocked them, saying they were the "Holy Club" and "the Methodists", being methodical and exceptionally detailed in their Bible study, opinions and disciplined lifestyle.
            Eventually, the so-called Methodists started individual societies or classes for members of the Church of England who wanted to live a more religious life. '''
            
encoding = tokenizer.encode_plus(text=question,text_pair=paragraph, add_special=True)