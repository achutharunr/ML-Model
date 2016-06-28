# -*- coding: utf-8 -*-
"""
Created on Wed May 25 16:34:23 2016

@author: U505118
"""
import urllib
import zipfile
import pandas as pd
from bs4 import BeautifulSoup
import os
import unidecode

"""BasicParse parses an xml file and extracts the xml tag attributes and the information contained within the tags."""

def BasicParse(path):
    
    """Declaring a pandas DataFrame with the required fields"""

    df1 = pd.DataFrame([],columns  = ['elementId', 'contextId', 'unitId', 'fact', 'decimals', 'scale', 'sign', 'factid', 'ns'])
    
    handle = open(path)
    
    """Creating a BeautifulSoup object with the xml file of a given company"""
    
    soup = BeautifulSoup(handle, "lxml")
    
    """Parsing the xml file for xml tags"""
    
    result = soup.findAll()

    y = soup.find('xbrl')    #Finding the xbrl tag to extract values for the ns field

    if not y:
	y = soup.find('xbrli:xbrl').attrs    #Accounting for variance in xbrl tag in the various documents
    else:
        y = y.attrs
    
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
                
            if unidecode.unidecode(result[i].text) != None:
                t = unidecode.unidecode(result[i].text).split('\n')
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


#########################################################################
#PROGRAM STARTS HERE


yearlist = []
flag = 1    #Flag variable used to check if the user has enter the info correctly

while flag:    
    startyear = raw_input('\n Please enter the start year : ')
    endyear = raw_input('\n Please enter the end year : ') 
    if startyear > endyear:
        startyear, endyear = endyear, startyear
        
    for x in xrange(int(startyear), int(endyear) + 1):    #Creating the list of years to be passed based on user input
        yearlist.append(str(x))
        
    startqtr = raw_input('\n Please enter the start quarter : ')
    endqtr = raw_input('\n Please enter the end quarter : ')
    
    #Creating dictionaries to act as switch case conditions to decide qtrlist
    
    qtrdicts = {'QTR1': ['QTR1', 'QTR2', 'QTR3', 'QTR4'], 'QTR2': ['QTR2', 'QTR3', 'QTR4'], 'QTR3' : ['QTR3', 'QTR4'], 'QTR4' : ['QTR4']}
    qtrdicte = {'QTR4': ['QTR1', 'QTR2', 'QTR3', 'QTR4'], 'QTR3': ['QTR1', 'QTR2', 'QTR3'], 'QTR2' : ['QTR1', 'QTR2'], 'QTR1' : ['QTR1']}
    
    if startyear == '2016' or endyear == '2016':
        if startqtr not in qtrdicte['QTR2'] or endqtr not in qtrdicte['QTR2']:
            print '\n\n Please enter a valid year and quarter!\n\n'
        else:
            flag = 0    #Flag set to 0 indicating that the user has correctly supplied the required info


start = 1    #Initialising start to 1
end = 0    #Initialising end to 0

basedir = 'C:/Users/U505118/Desktop/p/Hormel/full-index/'    #Setting up the base directory

for i in yearlist:
    
    if i == endyear:    #Initialising end to 1 if i is equal to endyear
        end = 1
        
    if start == 1:    #Initialising qtrlist based on start and end conditions
        qtrlist = qtrdicts[startqtr]
    elif end == 1:
        qtrlist = qtrdicte[endqtr]
    else:
        qtrlist = ['QTR1', 'QTR2', 'QTR3', 'QTR4']
        
    for j in qtrlist:
        
        start = 0
        
        """Checking if the directory exists and if not, making it"""
        
        path = basedir + i + '/' + j + '/' + '/parsed_data/'
        
        directory = os.path.dirname(path)
        
        if not os.path.exists(directory):
    	      os.makedirs(directory)

        """Extracting the xbrl.idx file from the xbrl.zip archive located in the edgar FTP server, if it doesn't exist locally"""
        
        if not os.path.exists(basedir + i + '/' + j + '/Testxbrl.zip'):
            urllib.urlretrieve('ftp://ftp.sec.gov/edgar/full-index/'+i+'/'+j+'/xbrl.zip', basedir + i + '/' + j + '/Testxbrl.zip')
            
        with zipfile.ZipFile( basedir + i + '/'+ j + '/Testxbrl.zip') as zp:    #Extracting xbrl.idx from xbrl.zip
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
        
        file_parse = []
        
        for l in xrange(0, 5):		#Set to 5 just for testing
            
            """Extracting the xml file in the individual company zip archive in the edgar FTP server"""
        if not os.path.exists(basedir + i + '/' + j + '/TestNick' + str(l) + '.zip'):
                urllib.urlretrieve(df.ix[l, 'Filename'], basedir + i + '/' + j + '/TestNick' + str(l) + '.zip')
                with zipfile.ZipFile(basedir + i + '/' + j + '/TestNick' + str(l) + '.zip') as zp:
                    
                    nl = zp.namelist()    #Obtaining a list of files in the archive
                    
                    """Searching to find the file name of the exact xml file to be parsed from the company ZIP archive"""
                    
                    len0 = nl[0].__len__()
                    for m in nl:    #Loop to loop through the mutliple files in each company's ZIP directory
                        if ('.xml' in m) and (m.__len__() <= len0):    #Checking for the appropriate XML file
                            filem = m
                            len0 = m.__len__()
                    
                    if not os.path.exists(basedir + i + '/' + j + '/' + filem + '/'):
                        zp.extract(filem, path = basedir + i + '/' + j + '/')
                        file_parse.append(filem)        #Adding the name of each company's XML file to be parsed to the list
                    
        """Declaring count and FileCount variables for writing the parsed info of 300 companies to each CSV file"""
        
        FileCount = 0
        count = 0   
        
        for n in xrange(0, 5):			#Set to 5 for testing
            
            """Opening a CSV file to write the parsed data of 300 companies"""
            
            with open(basedir + i + '/' + j + '/parsed_data/data' + str(FileCount)+'.csv', mode = 'ab') as f:
                
                """Calling the BasicParse function to parse through the xml file of the company and to return a DataFrame"""
                
                frame = BasicParse(basedir + i + '/' + j + '/' + file_parse[n])
                if count == 0:                    #If statement to prevent multiple headers in the same file
                    frame.to_csv(f, sep = '|', index = False)
                else:
                    frame.to_csv(f, header = False, sep = '|', index = False)
                    
                count = count + 1
                f.write('\n\n\n\n\n----------END---------\n\n\n\n\n')
                
                if count == 300:        #If statement to change file after XML files of 300 companies have been parsed 
                    FileCount = FileCount + 1
                    count = 0
                    
        os.remove(basedir + i + '/' + j + '/Testxbrl.zip')    #Removing to xbrl file after each quarter
        