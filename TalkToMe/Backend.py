# %%
import nltk
import string
import pandas as pd
import nlp_utils as nu
import matplotlib.pyplot as plt
# Loading necessary libraries

# %%
f = open("dialogs.txt", "r")
#print(f.read())
# reading the data 

# %%
df=pd.read_csv('dialogs.txt',names=('Query','Response'),sep=('\t'))
# Reading the data

# %%
#df
# loading the data

# %%
"""
## Data Understanding
"""

# %%
#df.shape
# There are 3724 rows and 2 columns in our dataset

# %%
#df.columns
# Displaying the names of columns present in the dataset

# %%
#df.info()
# Checking information of the data

# %%
#df.describe()
# Describe function shows us the frequency,unique and counts of all columns

# %%
#df.nunique()
# nunique() function return number of unique elements in the object. 

# %%
#df.isnull().sum()
# Checking for the presence of null values in the data. As we can see there are no null values present in the data

# %%
#df['Query'].value_counts()
# Checking the counts of the values present in the column 'Query'

# %%
#df['Response'].value_counts()
# Checking the counts of the values present in the column 'Response'

# %%
"""
## Data Visualization
"""

# %%
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# %%
Text=df['Query']

# %%
sid = SentimentIntensityAnalyzer()
for sentence in Text:
#     print(sentence)
        
     ss = sid.polarity_scores(sentence)
#     for k in ss:
#         print('{0}: {1}, ' .format(k, ss[k]), end='')
#     print()

# %%
analyzer = SentimentIntensityAnalyzer()
df['rating'] = Text.apply(analyzer.polarity_scores)
df=pd.concat([df.drop(['rating'], axis=1), df['rating'].apply(pd.Series)], axis=1)
### Creating a dataframe.

# %%
#df

# %%
from wordcloud import WordCloud
# importing word cloud

# %%
def wordcloud(df, label):
    
    subset=df[df[label]==1]
    text=df.Query.values
    wc= WordCloud(background_color="black",max_words=1000)

    wc.generate(" ".join(text))

    plt.figure(figsize=(20,20))
    plt.subplot(221)
    plt.axis("off")
    plt.title("Words frequented in {}".format(label), fontsize=20)
    plt.imshow(wc.recolor(colormap= 'gist_earth' , random_state=244), alpha=0.98)
# visualising wordcloud    

# %%
wordcloud(df,'Query')
# top words in the query column

# %%
wordcloud(df,'Response')
# top words in the response column

# %%
"""
## Text-Normalization
"""

# %%
# Removing special characters

# %%
import re
# importing regular expressions

# %%
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
# Lower case conversion

# %%
remove_n = lambda x: re.sub("\n", " ", x)
# removing \n and replacing them with empty value

# %%
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)
# removing non ascii characters

# %%
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
# removing alpha numeric values

# %%
df['Query'] = df['Query'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)
# using map function and applying the function on query column

# %%
df['Response'] = df['Response'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)
# using map function and applying the function on response column

# %%
#df
# final cleaned dataset

# %%
pd.set_option('display.max_rows',3800)
# Displaying all rows in the dataset

# %%
#df

# %%
"""
### Important Sentence
"""

# %%
imp_sent=df.sort_values(by='compound', ascending=False)
# arranging the compound column in descending order to find the best sentence. 

# %%
#imp_sent.head(5)
# printing the first 5 rows

# %%
"""
### Top Positive Sentence
"""

# %%
pos_sent=df.sort_values(by='pos', ascending=False)
# Arranging the dataframe by positive column in descending order to find the best postive sentence.

# %%
#pos_sent.head(5)
# printing the first 5 rows

# %%
"""
### Top Negative Sentence
"""

# %%
neg_sent=df.sort_values(by='neg', ascending=False)
# Arranging the dataframe by negative column in descending order to find the best negative sentence.

# %%
#neg_sent.head(5)
# printing the first 5 rows

# %%
"""
### Top Neutral Sentence
"""

# %%
neu_sent=df.sort_values(by='neu', ascending=False)
# Arranging the dataframe by negative column in descending order to find the best neutral sentence.

# %%
#neu_sent.head(5)
# printing the first 5 rows

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
# importing tfidf vectorizer

# %%
tfidf = TfidfVectorizer()
# Word Embedding - TF-IDF

# %%
factors = tfidf.fit_transform(df['Query']).toarray()
#print(factors)
# changing column into array

# %%
tfidf.get_feature_names_out()
# displaying feature names

# %%
"""
# Application
"""

# %%
from sklearn.metrics.pairwise import cosine_distances

# %%
query = 'who are you ?'
def chatbot(query):
    # step:-1 clean
    query = nu.lemmatization_sentence(query)
    # step:-2 word embedding - transform
    query_vector = tfidf.transform([query]).toarray()
    # step-3: cosine similarity
    similar_score = 1 -cosine_distances(factors,query_vector)
    index = similar_score.argmax() # take max index position
    # searching or matching question
    matching_question = df.loc[index]['Query']
    response = df.loc[index]['Response']
    pos_score = df.loc[index]['pos']
    neg_score = df.loc[index]['neg']
    neu_score = df.loc[index]['neu']
    confidence = similar_score[index][0]
    chat_dict = {'match':matching_question,
                'response':response,
                'score':confidence,
                'pos':pos_score,
                'neg':neg_score,
                'neu':neu_score}
    return chat_dict

