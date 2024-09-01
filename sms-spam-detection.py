#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[7]:


df = pd.read_csv('spam.csv', encoding='ISO-8859-1' )  # Ignores invalid characters


# In[8]:


df.sample(5)


# In[9]:


df.shape


# In[10]:


# 1. Data cleaning
# 2. EDA
# 3. Text Preprocessing
# 4. Model building
# 5. Evaluation
# 6. Improvement
# 7. Website
# 8. Deploy


# ## 1. Data Cleaning

# In[11]:


df.info()


# In[12]:


# drop last 3 cols
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[13]:


df.sample(5)


# In[14]:


# renaming the cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)


# In[15]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[16]:


df['target'] = encoder.fit_transform(df['target'])


# In[17]:


df.head()


# In[18]:


# missing values
df.isnull().sum()


# In[19]:


# check for duplicate values
df.duplicated().sum()


# In[20]:


# remove duplicates
df = df.drop_duplicates(keep='first')


# In[21]:


df.duplicated().sum()


# In[22]:


df.shape


# ## 2.EDA

# In[23]:


df.head()


# In[24]:


df['target'].value_counts()


# In[25]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[26]:


# Data is imbalanced


# In[28]:


get_ipython().system('pip install nltk')


# In[48]:


import nltk


# In[53]:


nltk.download('punkt_tab')


# In[54]:


df['num_characters'] = df['text'].apply(len)


# In[55]:


df.head()


# In[56]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[57]:


df.head()


# In[58]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[59]:


df.head()


# In[60]:


df[['num_characters','num_words','num_sentences']].describe()


# In[61]:


# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[62]:


#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[63]:


import seaborn as sns


# In[64]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'],color='red')


# In[65]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'],color='red')


# In[66]:


sns.pairplot(df,hue='target')


# ## 3. Data Preprocessing
# - Lower case
# - Tokenization
# - Removing special characters
# - Removing stop words and punctuation
# - Stemming

# In[73]:


nltk.download('stopwords')


# In[76]:


import string


# In[79]:


from nltk.stem import PorterStemmer

# Initialize the PorterStemmer
ps = PorterStemmer()


# In[80]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)


# In[81]:


transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")


# In[82]:


df['text'][10]


# In[83]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')


# In[84]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[85]:


df.head()


# In[87]:


get_ipython().system('pip install wordcloud')


# In[88]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[89]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[90]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[91]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[92]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[93]:


df.head()


# In[94]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
        


# In[95]:


len(spam_corpus)


# In[97]:


from collections import Counter
sns.barplot(x = pd.DataFrame(Counter(spam_corpus).most_common(30))[0], y = pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[98]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[99]:


len(ham_corpus)


# In[100]:


from collections import Counter
sns.barplot(x = pd.DataFrame(Counter(ham_corpus).most_common(30))[0], y = pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[101]:


# Text Vectorization
# using Bag of Words
df.head()


# ## 4. Model Building

# In[104]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[105]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[106]:


#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)


# In[107]:


# appending the num_character col to X
#X = np.hstack((X,df['num_characters'].values.reshape(-1,1)))


# In[108]:


X.shape


# In[109]:


y = df['target'].values


# In[110]:


from sklearn.model_selection import train_test_split


# In[111]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[112]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[113]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[114]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[115]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[116]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[117]:


# tfidf --> MNB


# In[119]:


get_ipython().system('pip install xgboost')


# In[120]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[121]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[122]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[123]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[124]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[125]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[126]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[127]:


performance_df


# In[128]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[129]:


performance_df1


# In[131]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[132]:


# model improve
# 1. Change the max_features parameter of TfIdf


# In[133]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending=False)


# In[134]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[135]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[136]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[137]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending=False)


# In[138]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[139]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[140]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[141]:


voting.fit(X_train,y_train)


# In[142]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[143]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[144]:


from sklearn.ensemble import StackingClassifier


# In[145]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[146]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[147]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




