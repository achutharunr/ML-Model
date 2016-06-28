# -*- coding: utf-8 -*-
"""
Created on Fri Jun 03 13:42:19 2016

@author: U505118
"""
from bs4 import BeautifulSoup
import pandas as pd
import re
from unidecode import unidecode



def BasicParse(path):

    df = pd.DataFrame([],columns  = ['elementId', 'contextId', 'unitId', 'fact', 'decimals', 'scale', 'sign', 'factid', 'ns'])
    
    handle = open(path)
    
    soup = BeautifulSoup(handle)
    
    result = soup.findAll()
    
    y = soup.find('xmlns')

    '''if not y:
	    y = soup.find('xbrli:xbrl').attrs
      else:
         y = y.attr'''

    for i in xrange(0, result.__len__()):
        tag = result[i].attrs
        
        if tag.has_key('contextref'):
            
            sen1 = result[i].name
            if tag.has_key('contextref'):
                sen2 = tag['contextref']
            else:
                sen2 = None
            if tag.has_key('unitref'):
                sen3 = tag['unitref']
            else:
                sen3 = None
            if result[i].text is not None:
                t = result[i].get_text()
                te = ''.join(t)
                sen4 = te
                
                '''soup2 = BeautifulSoup(result[i].text)
                if soup2 is not None:
                    soup2.prettify()
                    string  = soup2.get_text('|')
                    string = re.sub('[\n\t\r]', '', string)
                    string = re.sub('[\xa0]', ' ', string)
                    sen4 = string'''
                    
                '''bar = soup2.findAll()
                if bar != []:
                    string = bar[0].text
                    #string = re.sub('[\n\t\r]', '', string)
                    string = re.sub('[\xa0]', ' ', string)
                    sen4 = string'''
              
                                
                """soup1 = BeautifulSoup(result[i].text)
                if soup1 != None:
                    result1 = soup1.findAll()
                    for j in xrange(0, result1.__len__()):
                        tag1 = result1[j].attrs
                        if tag1 != None:
                            if '<' not in result1[j].text:
                                sen4 = result1[j].text"""
                    
                 
            else:
                sen4 = None
            if tag.has_key('decimals'):
                sen5 = tag['decimals']
            else:
                sen5 = None
                    
            if tag.has_key('scale'):
                sen6 = tag['scale']
            else:
                sen6 = None
                    
            if tag.has_key('sign'):
                sen7 = tag['sign']
            else:
                sen7 = None
                    
            if tag.has_key('factid'):
                sen8 = tag['factid']
            else:
                sen8 = None
                
            '''if result[i].name != None:
                SplitName = result[i].name.split(':')
                if any(SplitName[0] in x for x in y.keys()):
                    key = 'xmlns:' + SplitName[0]
                    sen9 = y[key]
            else:'''
            sen9 = None
            
            df.loc[i] = [sen1,sen2,sen3,sen4,sen5,sen6,sen7,sen8,sen9]
        
    return df
    
df = BasicParse('C:/Users/U505118/Desktop/P/Hormel/10k/hershey_10k.xml')
df = df.reset_index(drop = True)

s = []
count = 0

for i in xrange(0, len(df)):
    if len(df.loc[i, 'fact']) > 50:
        soup = BeautifulSoup(df.loc[i, 'fact'])
        result = soup.findAll('font')
        for j in xrange(0, len(result)):
            st = unidecode(result[j].text)
            st = st.strip()
            if st != '':
                if st[0].isupper() == False:
                    st = s[count-1] + unidecode(result[j].text)
                    del s[count-1]
                    count = count - 1
                s.append(st)
                count = count + 1
                
p = re.compile('sales')

parse = []

for i in xrange(0, len(s)):
    if p.search(s[i]) != None:
        parse.append(s[i])
        
s1 = parse

p = re.compile('%')

parse = []

for i in xrange(0, len(s1)):
    if p.search(s1[i]) != None:
        parse.append(s1[i])