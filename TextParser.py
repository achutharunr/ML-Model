# -*- coding: utf-8 -*-
"""
Created on Fri May 27 10:23:25 2016

@author: U505118
"""
    
#df = BasicParse('C:/Users/U505118/Desktop/P/Hormel/hormel_10k.xml')

CIK = df[df['elementId'] == 'identifier'].reset_index().ix[0,'fact'].lstrip('0')
Cname = df[df['elementId'] == 'dei:entityregistrantname'].reset_index().ix[0,'fact']
doc = df[df['elementId'] == 'dei:documenttype'].reset_index().ix[0,'fact']

p = re.compile('Wal-Mart')

parse = []

for i in xrange(0, df.__len__()):
    if df.loc[i,'fact'].__len__() >50:
        if p.search(df.loc[i, 'fact']) != None:
            print 'Exists' + str(i)
            l = p.search(df.ix[i, 'fact']).span()
            print l
            for j in xrange(0, 1000):

                if df.loc[i, 'fact'][l[0]-j-2:l[0]-j] == '.':
                    print 'Entered j!'
                    startl = j+1
                    break
            for k in xrange(0, 5000):
                if df.loc[i, 'fact'][l[1]+k] == '|'  or df.loc[i, 'fact'][l[1]+k : l[1]+k+2] == '.<':
                    print 'Entered k!\n'
                    endl = k
                    break
                
            sen = df.loc[i, 'fact'][l[0]-startl:l[1]+endl].lstrip(' ')
            sen = df.loc[i, 'fact'][l[0]-startl:l[1]+endl].lstrip('>')
            sen = re.sub(r'<.*?>', '', sen)
            sen = re.sub('[\n\t\r]', '', sen)
            sen = re.sub('[\u2019]', "'", sen)
            sen = re.sub('&nbsp;', ' ', sen)
            parse.append(sen)
            
            '''df.loc[i, 'fact'][l[1]+k : l[1]+k+2] == '.' or''' 