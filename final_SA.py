import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix
from nltk.corpus import stopwords
from textblob import Word

text1 = pd.read_csv(r"C:\Users\girievasan\Desktop\sentiment.csv",encoding=('latin'))
#converting to tweets to lower case 
for i in range(len(text1)):
    text1.apply(lambda x: x.astype(str).str.lower())
   
text1= text1.apply(lambda x: x.astype(str).str.lower())
#Removing punctuations
text1['TWEETS']=text1['TWEETS'].str.replace('[^\w\s]','')

#removing stopwords
stop=stopwords.words('english')
text1['TWEETS']=text1['TWEETS'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#Stemming words
text1['TWEETS']=text1['TWEETS'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#Labelling X and Y variables & vectorizing the tweets  
X= text1['TWEETS']                                      
tfidf=TfidfVectorizer(max_features=(13),ngram_range=(1,2))
X=tfidf.fit_transform(X)
y=text1['LABELS']
#Train and test split and apply 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
clf=LinearSVC()
clf.fit(X_train ,y_train)

y_pred=clf.predict(X_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
                    