
# coding: utf-8

# In[1]:



from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Word2VecUtility import Word2VecUtility
import pickle
import pandas as pd
import numpy as np
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)


# In[2]:


data = pd.read_csv('review.tsv', header=0, delimiter="\t", quoting=3)
print ('\nThe first review is:\n')
print((data["text"][0], '\n'))
print((data.shape))
print((data.columns))


# In[30]:


print((data['stars'][:3]))
print()
print((data.ix[:2]['text']))


# In[3]:



size = 1000000 #80000
subdata = data.sample(n = size, random_state=520)
subdata = subdata[pd.notnull(subdata['text'])]
print((subdata.index))
subdata.to_csv('review_sub_399850.tsv', index=False, quoting=3, sep='\t', encoding='utf-8')


# In[4]:


del(data)
data = subdata
del(subdata)


# In[6]:


data = pd.read_csv('review_sub_399850.tsv', header=0, delimiter="\t", quoting=3, encoding='utf-8')


# In[5]:


print((data.shape))
print((data.columns))
print((data.index))
# only after to_csv without index, and read_csv back to data, can you use ix[5]
# print data.ix[:5]['text']
# if you want to index dataframe directly after sampled it. use iloc
print((data.iloc[:5]))


# In[6]:


import nltk.data
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# In[7]:


# print data.ix[0:10]
print((data.iloc[:10]['text']))
# print data['text'][2]


# In[8]:


review_sents = []
print ("Cleaning and parsing the reviews...\n")
for i in range( 0, len(data["text"])):
    # sent_reviews += Word2VecUtility.review_to_sentences(data["text"][i], tokenizer)
    review_sents += Word2VecUtility.review_to_sentences(data.iloc[i]["text"], tokenizer)
    


# In[53]:


out = open('review_sents_1859888.pkl', 'wb')
pickle.dump(review_sents, out)
out.close()


# In[11]:


# review_sents = pickle.load(open('review_sents_1859888.pkl', 'rb'))
print((len(review_sents)))
print((review_sents[:5]))


# In[57]:


# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print ("Training model...")
model = word2vec.Word2Vec(review_sents, workers=num_workers,             size=num_features, min_count = min_word_count,             window = context, sample = downsampling)


# In[58]:


# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "300features_40minwords_10context"
model.save(model_name)


# In[ ]:


model = word2vec.Word2Vec.load("300features_40minwords_10context")


# In[71]:


model.doesnt_match("man woman child kitchen".split())


# In[70]:


model.doesnt_match("coffee tea juice restaurant".split())


# In[4]:


model.most_similar("delicious")


# In[5]:


model.most_similar("chinese")


# In[6]:


print((model["chinese"]))
print((model.syn0.shape))


# In[15]:


review_words = []
print((type(model.index2word)))
print((len(model.index2word)))
print((model.index2word[:100]))
index2word_set = set(model.index2word)
print((len(index2word_set)))


# In[16]:


words = Word2VecUtility.review_to_wordlist(data.iloc[0]['text'])
print(words)
for word in words:
    print((word in index2word_set))


# In[8]:


clean_labels = np.array(data["stars"])
print((clean_labels[:10], clean_labels.shape))
clean_labels[clean_labels <= 3] = 0
clean_labels[clean_labels > 3] = 1
print((clean_labels[:10]))
# num of positive reviews
print(((clean_labels == 1).sum()))

