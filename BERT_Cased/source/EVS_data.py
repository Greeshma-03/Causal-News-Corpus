

import xml.etree.ElementTree as Xet
import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string
pd.options.mode.chained_assignment = None

# Number of Columns in the data

cols = ["_t_id", "_sentence", "_number", "__text"]
rows = []


# Converting the XML Data  to the CSV format

xmlparse = Xet.parse('data.xml')
root = xmlparse.getroot()
for i in root:
	name = i.find("_t_id").text
	phone = i.find("_sentence").text
    sen = i.find("_number").text
    name1 = i.find("__text").text
	rows.append({"_t_id": name,"_sentence": phone,"_number":sen,"__text":name1})

df = pd.DataFrame(rows, columns=cols)

# Writing dataframe to csv
df.to_csv('output2.csv')

temp=len("output2.csv")

import nltk
from nltk.corpus import stopwords
print(stopwords.words('english'))


''' {‘ourselves’, ‘hers’, ‘between’, ‘yourself’, ‘but’, ‘again’, 
‘there’, ‘about’, ‘once’, ‘during’, ‘out’, ‘very’, ‘having’, 
‘with’, ‘they’, ‘own’, ‘an’, ‘be’, ‘some’, ‘for’, ‘do’, ‘its’, 
‘yours’, ‘such’, ‘into’, ‘of’, ‘most’, ‘itself’, ‘other’, ‘off’, 
‘is’, ‘s’, ‘am’, ‘or’, ‘who’, ‘as’, ‘from’, ‘him’, ‘each’, ‘the’, 
‘themselves’, ‘until’, ‘below’, ‘are’, ‘we’, ‘these’, ‘your’, ‘his’, 
‘through’, ‘don’, ‘nor’, ‘me’, ‘were’, ‘her’, ‘more’, ‘himself’, ‘this’, 
‘down’, ‘should’, ‘our’, ‘their’, ‘while’, ‘above’, ‘both’, ‘up’, ‘to’, 
‘ours’, ‘had’, ‘she’, ‘all’, ‘no’, ‘when’, ‘at’, ‘any’, ‘before’, ‘them’, 
‘same’, ‘and’, ‘been’, ‘have’, ‘in’, ‘will’, ‘on’, ‘does’, ‘yourselves’, 
‘then’, ‘that’, ‘because’, ‘what’, ‘over’, ‘why’, ‘so’, ‘can’, ‘did’, ‘not’, 
‘now’, ‘under’, ‘he’, ‘you’, ‘herself’, ‘has’, ‘just’, ‘where’, ‘too’, ‘only’, 
‘myself’, ‘which’, ‘those’, ‘i’, ‘after’, ‘few’, ‘whom’, ‘t’, ‘being’, ‘if’, 
‘theirs’, ‘my’, ‘against’, ‘a’, ‘by’, ‘doing’, ‘it’, ‘how’, ‘further’, ‘was’, ‘here’, ‘than’} '''


full_df = pd.read_csv("output2.csv", nrows=temp)
df = full_df[["__text"]]
df["text"] = df["text"].astype(str)
full_df.head()


# making all text to lower case for better understanding

df["loweredtext"] = df["text"].str.lower()
df.head()

df.drop(['_t_id'], axis=1)
df.drop(['_sentence'], axis=1)


# Punctuation removing function for the given text

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["textpunct"] = df["text"].apply(lambda text: remove_punctuation(text))
df.head()

# Stop words in english to remove in preprocessing

STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
   
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df["textstop"] = df["textpunct"].apply(lambda text: remove_stopwords(text))
df.head()



from collections import Counter
cnt = Counter()
for text in df["textstop"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)


FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):

    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

df["textstopfreq"] = df["textstop"].apply(lambda text: remove_freqwords(text))
df.head()

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Doing lemmatization that is changing the word to its root

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

df["text_lemmatized"] = df["text"].apply(lambda text: lemmatize_words(text))
df.head()

# Removing the urls and tags from the text

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)



def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)



df["remove_url"]=remove_urls(df["text_lemmatized"])
df["remove_tags"]=remove_html(df["remove_url"])

