# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:34:23 2016

@author: U505118
"""
import urllib
import zipfile
import pandas as pd
from bs4 import BeautifulSoup

"""BasicParse parses an xml file and extracts the xml tag attributes and the information contained within the tags."""

def BasicParse(path):
    
    """Declaring a pandas DataFrame with the required fields"""

    df1 = pd.DataFrame([],columns  = ['elementId', 'contextId', 'unitId', 'fact', 'decimals', 'scale', 'sign', 'factid', 'ns'])
    
    handle = open(path)
    
    """Creating a BeautifulSoup object with the xml file of a given company"""
    
    soup = BeautifulSoup(handle)
    
    """Parsing the xml file for xml tags"""
    
    result = soup.findAll()
    

    if y == None:
        y = soup.findAll('xbrli:xbrl')[0].attrs
    
    """Extracting necessary info"""
    
    for i in xrange(0, result.__len__()):
        
        """Assigning attributes of tag"""        
        
        tag = result[i].attrs
        
        """If condition to only extract those tags that have the contextRef attribute"""
        
        if tag.has_key('contextref'):
            
            elementId = result[i].name    #Assigning name of the tag to elementId 
             
            if tag.has_key('contextref'):
                contextRef = tag['contextref']    #Assigning contextref of the tag to contextRef, if it exists
            else:
                contextRef = None
                
            if tag.has_key('unitref'):
                unitRef = tag['unitref']    #Assigning unitref of the tag to unitRef, if it exists
            else:
                unitRef = None
                
            if result[i].text.encode('ascii','ignore') != None:
                t = result[i].text.encode('ascii','ignore').split('\n')
                te = ''.join(t)
                fact = te    #Assigning text contained within the tag to fact, if it exists 
            else:
                fact = None
                
            if tag.has_key('decimals'):
                decimals = tag['decimals']    #Assigning decimals attribute of the tag to decimals, if it exists
            else:
                decimals = None
                    
            if tag.has_key('scale'):
                scale = tag['scale']    #Assigning scale of the tag to scale, if it exists
            else:
                scale = None
                    
            if tag.has_key('sign'):
                sign = tag['sign']    #Assigning sign of the tag to sign, if it exists
            else:
                sign = None
                    
            if tag.has_key('factid'):
                factid = tag['factid']    #Assigning factid of the tag to factid, if it exists
            else:
                factid = None
    
            SplitName = result[i].name.split(':')    #Code that automatically assigns the corrosponding ns values, if it exists
            if any(SplitName[0] in x for x in y.keys()):
                key = 'xmlns:' + SplitName[0]
                ns = y[key]
            else:
                ns = None
            
            """Writing to DataFrame"""
            
            df1.loc[i] = [elementId, contextRef, unitRef, fact, decimals, scale, sign, factid, ns]    #Writing the tag attributes collected to a dataframe
        
    return df1    #Returning the dataframe with all the tag info from the given XML file 




yearlist = ['2011', '2012', '2013', '2014', '2015']
qtrlist = ('QTR1', 'QTR2', 'QTR3', 'QTR4')

basedir = 'C:/Users/U505118/Desktop/p/Hormel/full-index/'

for i in yearlist:
    for j in qtrlist:
        
        """Extracting the xbrl.idx file from the xbrl.zip archive located in the edgar FTP server"""
        
        urllib.urlretrieve('ftp://ftp.sec.gov/edgar/full-index/'+i+'/'+j+'/xbrl.zip', basedir + i + '/' + j + '/Testxbrl.zip')
            
        with zipfile.ZipFile( basedir + i + '/'+ j + '/Testxbrl.zip') as zp:
            zp.extractall(path = basedir + i + '/' + j + '/')
            
        """Writing contents of the xbrl.idx file to a pandas DataFrame"""
        
        df = pd.read_table(basedir + i + '/' + j + '/xbrl.idx', sep = '|', header = 4)
        
        """Dropping the first row which has only --------- """
        
        df = df.drop(0).reset_index(drop = True)
        
        """Formatting the URL for accessing individual company records in the edgar FTP server"""
        
        for k in xrange(0, df.__len__()):
            df.ix[k, 'Filename'] = df.ix[k, 'Filename'].replace('.txt','')
            temp = df.ix[k, 'Filename'].split('/')
            df.ix[k, 'Filename'] = df.ix[k, 'Filename'].replace('-','')
            df.ix[k, 'Filename'] = df.ix[k, 'Filename'] + '/' +  temp[-1] + '-xbrl.zip'
        df['Filename'] = 'ftp://ftp.sec.gov/' + df['Filename']
        
        
        """Extracting each company's individual xml files to be parsed for information"""
        
        for l in xrange(0, df.__len__()):
            
            """Extracting the xml file in the individual compnay zip archive in the edgar FTP server"""
            
            urllib.urlretrieve(df.ix[l, 'Filename'], basedir + i + '/' + j + '/TestNick.zip')
            with zipfile.ZipFile(basedir + i + '/' + j + '/TestNick.zip') as zp:
                nl = zp.nameslist()
                
                """Searching to find the file name of the exact xml file to be parsed from the company ZIP archive"""
                
                len0 = nl[0].__len__()
                file_parse = []
                for m in nl:    #Loop to loop through the mutliple files in each company's ZIP directory
                    if ('.xml' in m) and (m.__len__() <= len0):    #Checking for the appropriate XML file
                        filem = m
                        len0 = m.__len__()
                zp.extract(filem, path = basedir + i + '/' + j + '/')
                file_parse = file_parse + [filem]        #Adding the name of each company's XML file to be parsed to the list
                
        """Declaring count and FileCount variables for writing the parsed info of 300 companies to each CSV file"""
        
        FileCount = 0
        count = 0   
        
        for n in xrange(0, df.__len__()):
            
            """Opening a CSV file to write the parsed data of 300 companies"""
            
            with open(basedir + i + '/' + j + '/data' + str(FileCount), mode = 'ab') as f:
                
                """Calling the BasicParse function to parse through the xml file of the company and to return a DataFrame"""
                
                frame = BasicParse(basedir + i + '/' + j + '/' + file_parse[n])
                if count == 0:                    #If statement to prevent multiple headers in the same file
                    frame.to_csv(f, sep = '|', index = False)
                else:
                    frame.to_csv(f, header = False, sep = '|', index = False)
                    
                count = count + 1
                
                if count == 300:        #If statement to change file after XML files of 300 companies have been parsed 
                    FileCount = FileCount + 1
                    f.write('\n\n\n\n\n----------END---------\n\n\n\n\n')
                    count = 0

                
                
                
                

        
        
        
        
