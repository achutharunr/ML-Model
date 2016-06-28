# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 09:48:09 2016

@author: U505118
"""

import pandas as pd
import re
path = 'C:/Users/U505118/Desktop/P/outt2101.csv'

df = pd.read_csv(path)
textl = []

for text in df['fact']:
    #text = str( df.loc[i, 'fact'])
    text = str(text).lstrip(' ')
    if len(text) > 50:
        text = re.sub(r'<.*?>', '#', text)
        text = re.sub(r'&.*?;', ' ', text)
        text = re.sub(r'#\s*?#', r'#', text)
        text = re.sub(r'\s+', r' ', text)
        #text = re.sub(r'\n+', r'', text)
        #text = re.sub(r')
        textl.append(text)
        
p = re.compile('sales')

ans = []

for i in range(0, len(textl)):
    if re.search(r'[A-Z][\w ]*[0-9]*%[\w ]*\.', textl[i]) != None:
        if len(re.findall(r'[A-Z][\w ]*[0-9]*%[\w ]*\.', textl[i])) > 0:
            ans.append(re.findall(r'[A-Z][\w ]*[0-9]*%[\w ]*\.', textl[i]))