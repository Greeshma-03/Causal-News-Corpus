import xml.etree.ElementTree as Xet
import numpy as np
import pandas as pd
import re
import nltk
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import spacy
import string
pd.options.mode.chained_assignment = None
model_name ='bert-base-cased'

tokenizer = BertTokenizer.from_pretrained(model_name)

cols = ["index", "text"]
rows = []


xmlparse = Xet.parse('data.xml')
root = xmlparse.getroot()
for i in root:
	name = i.find("index").text
	phone = i.find("text").text
    
	rows.append({"_id": name,
				"text": phone})

df = pd.DataFrame(rows, columns=cols)

# Writing dataframe to csv
df.to_csv('output.csv')

temp=len("output.csv")

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


full_df = pd.read_csv("output.csv", nrows=temp)
df = full_df[["text"]]
df["text"] = df["text"].astype(str)
full_df.head()


df["loweredtext"] = df["text"].str.lower()
df.head()



PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["textpunct"] = df["text"].apply(lambda text: remove_punctuation(text))
df.head()



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

lemmatizer = WordNetLemmatizer()
wordnet_map = {"N":wordnet.NOUN, "V":wordnet.VERB, "J":wordnet.ADJ, "R":wordnet.ADV}
def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])

df["text_lemmatized"] = df["text"].apply(lambda text: lemmatize_words(text))
df.head()



def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)



def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

class datloader(df["remove_tags"]):

  def __init__(self,data):
    self.text = data
    
  def __len__(self):
    return len(self._id)
  
  def __getitem__(self, item):
    text = str(self.text[item])
    target = self._id[item]

    encoding = self.tokenizer.encode_plus(
      self.text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'text': self.text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }


df["remove_url"]=remove_urls(df["text_lemmatized"])
df["remove_tags"]=remove_html(df["remove_url"])

origindata=datloader(df["remove_tags"])


def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = datloader(
    text=df.content.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return datloader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )


data=create_data_loader(origindata,tokenizer,120,16)
