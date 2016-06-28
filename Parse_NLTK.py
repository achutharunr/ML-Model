# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 09:41:46 2016

@author: U505118
"""

import pandas as pd
import re
import unidecode
import nltk


from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize.punkt import PunktParameters
punkt = PunktParameters()
punkt.abbrev_types = ['u.s.a', 'ltd', 'inc', 'no']
sen = PunktSentenceTokenizer(punkt)

for k in range(21, 87, 3):
    df = pd.read_csv('C:/Users/U505118/Desktop/P/10_K/outt'+str(k)+'01.csv')
    
    textl = []
    i = 0
    index = []
    
    for x in df['fact']:
        if type(x) is str:
            if len(x) > 100:
                #x = unidecode.unidecode(x)
                x = re.sub(r'<.*?>', ' ', x)
                x = re.sub(r'\xe2', '', x)
                x = re.sub(r'\xc2', '', x)
                x = re.sub(r'\x93', '', x)
                x = re.sub(r'\x94', '', x)
                x = re.sub(r'\x92', '', x)
                x = re.sub(r'\x96', '', x)
                x = re.sub(r'\x99', '', x)
                x = re.sub(r'\x80', '', x)
                x = re.sub(r'\x9c', '', x)
                x = re.sub(r'\x9d', '', x)
                x = re.sub(r'&.*?;', ' ', x)
                x = re.sub(r'\s{2,10}?\d\s{2, 10}?', r'', x)
                x = re.sub(r'\s+', ' ', x)
                x = re.sub(r'\t+?', r'\n', x)
                a = nltk.sent_tokenize(x)
                textl = textl + a
                for text in a:
                    if len(text) > 100:
                        index.append(i)
                    
        i = i + 1
    
    l1 = re.compile('sales|revenue')
    l2 = re.compile('contribute|attribute|represent|account')
    l3 = re.compile('%|percent')
    textl = set(list(textl))
    
    frame = pd.DataFrame([],columns  = ['qtr', 'Form_Type', 'Company_Name', 'CIK', 'Date', 'Path', 'Indicator', 'Phrase'])
    
    ans1 = []
    in1 = []
    ans2 = []
    in2 = []
    ans3 = []
    in3 = []
    
    i = 0
    
    for x in textl:
        i = i + 1
        if l1.search(x) != None:
            ans1.append(x)
            in1.append(i)
    
    i = 0
            
    for x in ans1:
        i = i + 1
        if l2.search(x) != None:
            ans2.append(x)
            in2.append(i)
    
    i = 0
    
    for x in ans2:
        i = i + 1
        if l3.search(x) != None:
            ans3.append(x)
            in3.append(i)
        
    ans1 = list(set(ans1))
    ans2 = list(set(ans2))
    ans3 = list(set(ans3))
        
    
    for i in range(0, len(ans3
    )):
        frame.loc[i] = [df.loc[i, 'qtr'], df.loc[i, 'form_type'], df.loc[i, 'Company_Name'], df.loc[i, 'CIK'], df.loc[i, 'Date.Filed'], df.loc[i, 'Path'], 'No', ans3[i]]
    
    '''with open('C:/Users/U505118/Desktop/P/Final.csv', mode = 'ab') as f:        
        frame.to_csv(f, index = False)'''