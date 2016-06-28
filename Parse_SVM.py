# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 13:37:56 2016

@author: U505118
"""

from sklearn import svm
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
import pylab as pl
from sklearn.metrics import roc_curve, roc_auc_score
import re
 
df = pd.read_csv('C:/Users/U505118/Desktop/P/Final_SVM.csv')
df_train = df[:1874]
df_test = df[1874:]
df_model = df_test



model = svm.SVC(kernel = 'linear', C = 100, gamma = 2, probability = True)

#--------------------Training the model---------------------#

'''Declaring required variables'''

vocab = ['revenue', 'revenue months', 'revenues', 'revenues months', 'sales', 'sales months',
          'net income', 'net revenue', 'net revenues', 'net sales', 'no', 'no customer', 'no customer accounted', 'cost revenue', 'fiscal', 'following', 'following table',
          'follows', 'gross', 'table', 'accounted', 'accounted approximately', 'accounted total',
          'customers', 'customers accounted', 'represented', 'percent', 'foreign', 'country', 'product service',
          'escrow', 'contracts', 'contract', 'properties', 'equity', 'cost']
          
words = df_train['Phrase'].tolist()
words_test = df_test['Phrase'].tolist()
indArray = np.asarray(df_train['Indicator'].tolist())
for i in range(len(words)):
    words[i] = re.sub('\d+', '', words[i])
    
for i in range(len(words_test)):
    words_test[i] = re.sub('\d+', '', words_test[i])

stopwords = ['a',
           'about',
           'above',
           'across',
           'after',
           'afterwards',
           'again',
           'against',
           'all',
           'almost',
           'alone',
           'along',
           'already',
           'also',
           'although',
           'always',
           'am',
           'among',
           'amongst',
           'amoungst',
           'amount',
           'an',
           'and',
           'another',
           'any',
           'anyhow',
           'anyone',
           'anything',
           'anyway',
           'anywhere',
           'are',
           'around',
           'as',
           'at',
           'back',
           'be',
           'became',
           'because',
           'become',
           'becomes',
           'becoming',
           'been',
           'before',
           'beforehand',
           'behind',
           'being',
           'below',
           'beside',
           'besides',
           'between',
           'beyond',
           'bill',
           'both',
           'bottom',
           'but',
           'by',
           'call',
           'can',
           'cannot',
           'cant',
           'co',
           'con',
           'could',
           'couldnt',
           'cry',
           'de',
           'describe',
           'detail',
           'do',
           'done',
           'down',
           'due',
           'during',
           'each',
           'eg',
           'eight',
           'either',
           'eleven',
           'else',
           'elsewhere',
           'empty',
           'enough',
           'etc',
           'even',
           'ever',
           'every',
           'everyone',
           'everything',
           'everywhere',
           'except',
           'few',
           'fifteen',
           'fify',
           'fill',
           'find',
           'fire',
           'first',
           'five',
           'for',
           'former',
           'formerly',
           'forty',
           'found',
           'four',
           'from',
           'front',
           'full',
           'further',
           'get',
           'give',
           'go',
           'had',
           'has',
           'hasnt',
           'have',
           'he',
           'hence',
           'her',
           'here',
           'hereafter',
           'hereby',
           'herein',
           'hereupon',
           'hers',
           'herself',
           'him',
           'himself',
           'his',
           'how',
           'however',
           'hundred',
           'i',
           'ie',
           'if',
           'in',
           'inc',
           'indeed',
           'interest',
           'into',
           'is',
           'it',
           'its',
           'itself',
           'keep',
           'last',
           'latter',
           'latterly',
           'least',
           'less',
           'ltd',
           'made',
           'many',
           'may',
           'me',
           'meanwhile',
           'might',
           'mill',
           'mine',
           'more',
           'moreover',
           'most',
           'mostly',
           'move',
           'much',
           'must',
           'my',
           'myself',
           'name',
           'namely',
           'neither',
           'never',
           'nevertheless',
           'next',
           'nine',
           'nobody',
           'none',
           'noone',
           'nor',
           'not',
           'non',
           'nothing',
           'now',
           'nowhere',
           'of',
           'off',
           'often',
           'on',
           'once',
           'one',
           'only',
           'onto',
           'or',
           'other',
           'others',
           'otherwise',
           'our',
           'ours',
           'ourselves',
           'out',
           'over',
           'own',
           'part',
           'per',
           'perhaps',
           'please',
           'put',
           'rather',
           're',
           'same',
           'see',
           'seem',
           'seemed',
           'seeming',
           'seems',
           'serious',
           'several',
           'she',
           'should',
           'show',
           'side',
           'since',
           'sincere',
           'six',
           'sixty',
           'so',
           'some',
           'somehow',
           'someone',
           'something',
           'sometime',
           'sometimes',
           'somewhere',
           'still',
           'such',
           'system',
           'take',
           'ten',
           'than',
           'that',
           'the',
           'their',
           'them',
           'themselves',
           'then',
           'thence',
           'there',
           'thereafter',
           'thereby',
           'therefore',
           'therein',
           'thereupon',
           'these',
           'they',
           'thick',
           'thin',
           'third',
           'this',
           'those',
           'though',
           'three',
           'through',
           'throughout',
           'thru',
           'thus',
           'to',
           'together',
           'too',
           'top',
           'toward',
           'towards',
           'twelve',
           'twenty',
           'two',
           'un',
           'under',
           'until',
           'up',
           'upon',
           'us',
           'very',
           'via',
           'was',
           'we',
           'well',
           'were',
           'what',
           'whatever',
           'when',
           'whence',
           'whenever',
           'where',
           'whereafter',
           'whereas',
           'whereby',
           'wherein',
           'whereupon',
           'wherever',
           'whether',
           'which',
           'while',
           'whither',
           'who',
           'whoever',
           'whole',
           'whom',
           'whose',
           'why',
           'will',
           'with',
           'within',
           'without',
           'would',
           'yet',
           'you',
           'your',
           'yours',
           'yourself',
           'yourselves',
           '000',
           'year',
           'states',
           'tax',
           'taxes',
           '100',
           'january',
           'jan',
           'february',
           'feb',
           'march',
           'april',
           'may',
           'june',
           'jun',
           'july',
           'jul',
           'august',
           'september',
           'sept',
           'october',
           'oct',
           'november',
           'nov',
           'december',
           'dec',
           'equity',
           'ended',
           'accounts',
           'cash',
           'consolidated',
           'period',
           'periods',
           'liabilities',
           'customer']
'''
           '11'
           '12',
           '13',
           '14',
           '15',
           '16',
           '17',
           '18',
           '19',
           '20',
           '2013',
           '2014',
           '2015',
           '23',
           '24',
           '25',
           '26',
           '27',
           '28',
           '30',
           '31']'''
        
'''vect = TfidfVectorizer(stop_words = stopwords, token_pattern = '[a-z]+')

idfArray = vect.fit_transform(words).toarray()

netscore = vect.idf_
wordname = vect.get_feature_names()
wordscore = []
featurename = []

for i in range(len(netscore)):
    if netscore[i] <= 5.00 and len(wordname[i]) > 1:
        wordscore.append(netscore[i])
        featurename.append(wordname[i])'''
    
vect = CountVectorizer(stop_words = stopwords, ngram_range = (1, 2), vocabulary = vocab, max_features = 100)
idfArray = vect.fit_transform(words).toarray()

vect_test = CountVectorizer(stop_words = stopwords, ngram_range = (1, 2), vocabulary = vocab)
testArray = vect_test.fit_transform(words_test).toarray()

model.fit(idfArray, indArray)
testInd = model.predict(testArray)

df_model['SVM Indicator'] = testInd.tolist()
df_model = df_model.reset_index(drop = True)
df_model.to_csv('C:/Users/U505118/Desktop/P/SVMTestData.csv', index = False)
prob = model.predict_proba(testArray).tolist()
prob_temp = []
for x in prob:
    prob_temp.append(x[1])
    
prob = np.asarray(prob_temp)
del prob_temp


#----------------Confusion Matrix---------------------------------#

TP = 0.0
TN = 0.0
FP = 0.0
FN = 0.0
sensitivity = 0.0
specificity = 0.0
precision = 0.0
NPV = 0.0

for i in range(len(df_model)):
    if df_model.loc[i, 'Indicator'] == 'Yes':
        if df_model.loc[i, 'SVM Indicator'] == 'Yes':
            TP = TP + 1
        elif df_model.loc[i, 'SVM Indicator'] == 'No':
            FN = FN + 1
    elif df_model.loc[i, 'Indicator'] == 'No':
        if df_model.loc[i, 'SVM Indicator'] == 'Yes':
            FP = FP + 1
        elif df_model.loc[i, 'SVM Indicator'] == 'No':
            TN = TN + 1
            
sensitivity = (TP/(TP + FN))*100.0
specificity = (TN/(TN + FP))*100.0
precision = (TP/(TP + FP))*100.0
NPV = (TN/(TN + FN))*100.0

 
print '\n\n\n\t\t\t---------Confusion Matrix---------\n'
print '\t\t\t\t     Predicted\n'
print '\t\t\t\tYes\t\tNo'
print '\t\t\t---------------------------------'
print '\t\t\t|\t\t|\t\t|' 
print '\t\tYes\t|\t' + str(TP) + '\t|\t' + str(FN) + '\t|\tPrecision : %f' %precision
print '\t\t\t|\t\t|\t\t|'
print '      Actual\t\t---------------------------------'
print '\t\t\t|\t\t|\t\t|'       
print '\t\tNo\t|\t'+ str(FP)+ '\t|\t' + str(TN) + '\t|\tNPV : %f' %NPV
print '\t\t\t|\t\t|\t\t|'
print '\t\t\t---------------------------------\n\n'
print ('\t\tSensitivity : %f\t\tSpecificity : %f\n\n' %(sensitivity, specificity))



#-----------------ROC Curve-------------------------#

actual = []
for x in df_test['Indicator']:
    if x == 'Yes':
        actual.append(1)
    elif x == 'No':
        actual.append(0)
        
actualvalArray = np.asarray(actual)
del actual
        
FPR, TPR, thresholds = roc_curve(actualvalArray, prob)
roc_auc = roc_auc_score(actualvalArray, prob)

print("Area under the ROC curve : %f" % roc_auc)

'''
pl.clf()
pl.plot(FPR, TPR, label='ROC curve')
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc='lower right')
pl.show()'''

#---------------Gain and Lift Chart------------------#

df_model['Probability'] = prob
df_model.sort(columns = 'Probability', inplace = True, axis = 1, ascending = False)


'''
model = svm.SVC(kernel = 'linear', C = 1, gamma = 1)

df = pd.read_csv('C:/Users/U505118/Desktop/P/Final.csv')
frame = pd.DataFrame([],columns  = ['Indicator', 'Phrase'])
k = 0

for i in range(len(df)):
    if df.loc[i, 'Indicator'] == 'Yes':
        frame.loc[k] = [df.loc[i, 'Indicator'], df.loc[i, 'Phrase']]
        k = k + 1
    
frame = frame.reset_index(drop = True)
words = []
sen = ''

for x in frame['Phrase']:
    sen = sen + x

sen = sen.lower()
t = nltk.tokenize.RegexpTokenizer(r'[a-z]+')
y = t.tokenize(sen)
y = list(set(y))
words = [w for w in y if not w in stopwords.words('english')]'''

