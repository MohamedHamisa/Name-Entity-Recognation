import nltk
import pandas as pd

text = "Apple acquired Zoom in China on Wednesday 6th May 2020.\
This news has made Apple and Google stock jump by 5% on Dow Jones Index in the \
United States of America"

#tokenize to words
words = nltk.word_tokenize(text)
words

#Part of speech tagging
pos_tags = nltk.pos_tag(words)
pos_tags

#check nltk help for description of the tag
nltk.help.upenn_tagset('NNP')

chunks = nltk.ne_chunk(pos_tags, binary=True) #either NE or not NE
for chunk in chunks:
    print(chunk)

entities =[]
labels =[]
for chunk in chunks:
    if hasattr(chunk,'label'):  #hasattr() function returns True if the specified object has the specified attribute, otherwise False
        #print(chunk)
        entities.append(' '.join(c[0] for c in chunk))
        labels.append(chunk.label())
        
entities_labels = list(set(zip(entities, labels)))
entities_df = pd.DataFrame(entities_labels)
entities_df.columns = ["Entities","Labels"]
entities_df

chunks = nltk.ne_chunk(pos_tags, binary=False) #either NE or not NE
for chunk in chunks:
    print(chunk)
    
entities =[]
labels =[]
for chunk in chunks:
    if hasattr(chunk,'label'):
        #print(chunk)
        entities.append(' '.join(c[0] for c in chunk))
        labels.append(chunk.label())
        
entities_labels = list(set(zip(entities, labels)))
entities_df = pd.DataFrame(entities_labels)
entities_df.columns = ["Entities","Labels"]
entities_df

entities = []
labels = []

sentence = nltk.sent_tokenize(text)
for sent in sentence:
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)),binary=False):
        if hasattr(chunk,'label'):
            entities.append(' '.join(c[0] for c in chunk))
            labels.append(chunk.label())
            
entities_labels = list(set(zip(entities,labels)))

entities_df = pd.DataFrame(entities_labels)
entities_df.columns = ["Entities","Labels"]
entities_df

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import os
model = 'C:/StanfordNER_Tagger/stanford-ner-2018-10-16/classifiers/english.all.3class.distsim.crf.ser.gz'
jar = 'C:/StanfordNER_Tagger/stanford-ner-2018-10-16/stanford-ner.jar'



st = StanfordNERTagger(model, jar,encoding='utf-8')
tokenized_text = nltk.word_tokenize(text)
classified_text = st.tag(tokenized_text)

classified_text_df = pd.DataFrame(classified_text)

classified_text_df.drop_duplicates(keep='first', inplace=True)
classified_text_df.reset_index(drop=True, inplace=True)
classified_text_df.columns = ["Entities", "Labels"]
classified_text_df

tokenized_text = nltk.word_tokenize(text)
classified_text = st.tag(tokenized_text)

netagged_words = classified_text

entities = []
labels = []

from itertools import groupby
for tag, chunk in groupby(classified_text, lambda x:x[1]):
    if tag != "O":
        entities.append(' '.join(w for w, t in chunk))
        labels.append(tag)
        
        
entities_all = list(zip(entities, labels))
entities_unique = list(set(zip(entities, labels))) #unique entities   
entities_df = pd.DataFrame(entities_unique)
entities_df.columns = ["Entities", "Labels"]
entities_df

import spacy 
from spacy import displacy
#SpaCy 2.x brough significant speed and accuracy improvements
spacy.__version__

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")
#nlp = spacy.load("en_core_web_md")
#nlp = spacy.load("en_core_web_lg")
doc = nlp(text)

entities = []
labels = []
position_start = []
position_end = []

for ent in doc.ents:
    entities.append(ent)
    labels.append(ent.label_)
    position_start.append(ent.start_char)
    position_end.append(ent.end_char)
    
df = pd.DataFrame({'Entities':entities,'Labels':labels,'Position_Start':position_start, 'Position_End':position_end})

df

spacy.explain("ORG")
