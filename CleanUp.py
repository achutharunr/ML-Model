# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 13:30:06 2016

@author: U505118
"""

import pandas as pd
import re
path = 'C:/Users/U505118/Desktop/P/outt2101.csv'

df = pd.read_csv(path)
textl = []
CIK = []
qtr = []
form_type = []
company_name = []
date_filed = []
index = []
i = 0

for x in df['fact']:
    x = str(x).lstrip(' ')
    i = i + 1
    if len(x) > 50:
        x = re.sub(r'<.*?>', ' ', x)
        x = re.sub(r'&.*?;', ' ', x)
        x = re.sub(r'\s{2,10}?\d\s{2, 10}?', r'', x)
        x = re.sub(r'\t+?', r'\n', x)
        '''x = re.sub(r'\s', r' ', x)'''
        textl.append(x)
        CIK.append(df.loc[i, 'CIK'])
        qtr.append(df.loc[i, 'qtr'])
        form_type.append(df.loc[i, 'form_type'])
        company_name.append(df.loc[i, 'Company_Name'])
        date_filed.append(df.loc[i, 'Date.Filed'])
        index.append(i)
        
        
p = re.compile('[A-Z].+?[.] ')     
sen = []
CIK1 = []
qtr1 = []
form_typ1e = []
company_name1 = []
date_filed1 = []
i = 0

sen = []
        
for x in textl:
    a = (p.findall(x))
    i = i+1
    for y in a:
        y = str(y)
        i = i + 1
        if len(y) > 100:
            sen.append(y)


ans = []
ans1 = []
ans2 = []            
sales = re.compile('sale|revenue')
percent = re.compile('%')
accounted = re.compile('accounted|represent')
for x in sen:
    if sales.search(x) != None:
        ans.append(x)
        
for x in ans:
    if accounted.search(x) != None:
        ans1.append(x)

for x in ans1:
    if percent.search(x) != None:
        ans2.append(x)        
print count

ans2 = list(set(ans2))
ans = list(set(ans))
ans1= list(set(ans1))



