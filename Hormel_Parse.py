# -*- coding: utf-8 -*-
"""
Created on Mon May 23 16:08:01 2016

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
    
    y = soup.find('xbrl')

    if not y:
	y = soup.find('xbrli:xbrl').attrs
    else:
        y = y.attrs

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
                
            if result[i].name != None:
                SplitName = result[i].name.split(':')
                if any(SplitName[0] in x for x in y.keys()):
                    key = 'xmlns:' + SplitName[0]
                    sen9 = y[key]
            else:
                sen9 = None
            
            df.loc[i] = [sen1,sen2,sen3,sen4,sen5,sen6,sen7,sen8,sen9]
        
    return df
    
df = BasicParse('C:/Users/U505118/Desktop/P/Hormel/hormel_10k.xml')
df = df.reset_index(drop = True)

##################
            
"""p = re.compile('value')

parse = []

for i in xrange(0, df.__len__()):
    df.loc[i, 'fact'] =  re.sub(r'<.*?>', '', df.loc[i, 'fact'])
    df.loc[i, 'fact'] =  re.sub('[\n\t\r]', '', df.loc[i, 'fact'])
    df.loc[i, 'fact'] =  re.sub('&#x2019;', "'", df.loc[i, 'fact'])
    df.loc[i, 'fact'] = re.sub('&nbsp;', ' ', df.loc[i, 'fact'])
    df.loc[i, 'fact'] = re.sub('&#x2013;', ' ', df.loc[i, 'fact'])
    if df.loc[i,'fact'].__len__() > 50:
        if p.search(df.loc[i, 'fact']) != None:
            print 'Exists' + str(i)
            l = p.search(df.ix[i, 'fact']).span()
            print l
            for j in xrange(0, 1000):
                if df.loc[i, 'fact'][l[0]-j - 2:l[0]-j] == '. ':
                    print 'Entered j!'
                    startl = j
                    break
            for k in xrange(0, 5000):
                if df.loc[i, 'fact'][l[1]+k : l[1]+k + 2] == '. ':
                    print 'Entered k!\n'
                    endl = k + 1
                    break
                
            sen = df.loc[i, 'fact'][l[0]-startl:l[1]+endl].lstrip(' ')
            parse.append(sen)
    parse = filter(None, parse)"""
                        
#CIK = str(df[df['elementId'] == 'dei:entitycentralindexkey'].reset_index().ix[0,'fact'].lstrip('0'))
#year = str(df[df['elementId'] == 'dei:documentfiscalyearfocus'].reset_index().ix[0,'fact'])

#doc = df[df['elementId'] == 'dei:documenttype'].reset_index().ix[0,'fact']

#df.to_csv('C:/Users/U505118/Desktop/P/Hormel/Test_New.csv', sep = '|', index = False)