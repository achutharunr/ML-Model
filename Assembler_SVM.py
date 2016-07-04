# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 09:50:55 2016

@author: U505118
"""


def SVM_Parse(phraseList, actualInd):
    
    from sklearn import svm
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    import pylab as pl
    from sklearn.metrics import roc_curve, roc_auc_score
    import re
    import pickle

    #-----------------Declaring necessary variables------------#
              
    with open('C:/Users/U505118/Desktop/P/Hormel/stopwords_SVM.txt','rb') as f:
        stopwords = f.read().split(',')
    f.close()
    
    #--------------------Cleaning the data---------------------#

        
    for i in range(len(phraseList)):
        phraseList[i] = re.sub('\d+', '', phraseList[i])
    
    
    #-------------------Training the model--------------------#        
    
    with open('C:/Users/U505118/Desktop/P/Hormel/SVMmodel.pickle', 'rb') as filehandler:
       model = pickle.load(filehandler)
    
    '''Vectorizing the test data'''
    
    vect_test = CountVectorizer(stop_words = stopwords, ngram_range = (1, 2), max_features = 105)
    testArray = vect_test.fit_transform(phraseList).toarray()
    
    '''Testing the model'''
    
    testInd = model.predict(testArray)
    ind_SVM = testInd.tolist()
    
    indnum_SVM = []
    for x in ind_SVM:
        if x == 'Yes':
            indnum_SVM.append(1)
        elif x == 'No':
            indnum_SVM.append(0)
    
    #--------------Updating the test data------------------------------#
    
    df_model = pd.DataFrame()
    df_model['SVM Indicator'] = ind_SVM
    df_model['Indicator'] = actualInd
    
    '''Calculating proabilities of desired event fro AUCROC'''
    
    prob = model.predict_proba(testArray).tolist()
    prob_list = []
    for x in prob:
        prob_list.append(x[1])
        
    prob = np.asarray(prob_list)
    
    metrics = 0
    
    if metrics:
    
        ####################### Model Evaluation ###########################
        
        
        #----------------Confusion Matrix---------------------------------#
        
        '''Declaring necessary variables'''
        
        TP = 0.0    #True Positive
        TN = 0.0    #True Negative
        FP = 0.0    #False Positive
        FN = 0.0    #False Negative
        sensitivity = 0.0    #Also known as Recall
        specificity = 0.0    #Also known as Recall
        precision = 0.0    #Also known as Postive Predictive Value
        NPV = 0.0    #Negative Predictive Value
        accuracy = 0.0
        
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
                       
        '''Calculating metrics'''
                
        sensitivity = (TP/(TP + FN))*100.0
        specificity = (TN/(TN + FP))*100.0
        precision = (TP/(TP + FP))*100.0
        NPV = (TN/(TN + FN))*100.0
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        
        '''Printing the Confusion Matrix'''
        
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
        print'\t\t\t\tAccuracy : %f' %accuracy
    
        #-----------------ROC Curve-------------------------#
        
        '''Calculating actual number of positive cases'''
        
        actual = []
        for x in df_model['Indicator']:
            if x == 'Yes':
                actual.append(1)
            elif x == 'No':
                actual.append(0)
                
        actualvalArray = np.asarray(actual)
        del actual
        
        '''Calculating FPR, TPR and AUCROC'''
        
        FPR, TPR, thresholds = roc_curve(actualvalArray, prob)
        roc_auc = roc_auc_score(actualvalArray, prob)
        
        
        print("Area under the ROC curve : %f" % roc_auc)
        
        '''Plotting the AUCROC'''
        
        pl.clf()
        pl.plot(FPR, TPR, label='ROC curve')
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic')
        pl.legend(loc='lower right')
        pl.show()
        
        #---------------Gain and Lift Chart------------------#
        
        '''Declaring necessary variables'''    
        
        gain_model = [0]
        gain_random = [0]
        lift = []
        random_cumul = 0.0
        model_cumul = 0.0
        dec = len(df_model)/10
        df_model['Probability'] = prob
        df_GLsort = df_model.sort(columns = 'Probability', inplace = False, ascending = False)
        df_GLsort = df_GLsort.reset_index(drop = True)
        df_model = df_model.sample(n = len(df_model)).reset_index(drop = True)
        df_GL = pd.DataFrame([],columns  = ['Decile', 'Random', 'Model'])
        
        '''Dividing the data set into deciles and calculating gain'''    
        
        for i in range(10):
            df_GL.loc[i, 'Decile'] = str(i + 1)
            random = 0.0
            model = 0.0
            for k in range(i*dec, (i*dec) + dec):
                if df_GLsort.loc[k, 'Indicator'] == 'Yes':
                    model = model + 1
                if df_model.loc[k, 'Indicator'] == 'Yes':
                    random = random + 1
            random_cumul = random_cumul + random
            model_cumul = model_cumul + model
            lift.append((model-random)/dec)
            df_GL.loc[i, 'Random'] = str(random_cumul)
            df_GL.loc[i, 'Model'] = str(model_cumul)
            gain_model.append(((model_cumul)/(TP + FN))*100.00)
            gain_random.append(((random_cumul)/(TP + FN))*100.00)
        
        '''Plotting the cumulative gain chart'''
        
        percent_pop = range(0, 110, 10)
        
        pl.clf()
        pl.plot(percent_pop, gain_model, label='Model')
        pl.plot(percent_pop, gain_random, label='Random')
        pl.xlim([0.0, 100.0])
        pl.ylim([0.0, 100.0])
        pl.xlabel('Percentage of Data Set')
        pl.ylabel('Percentage of Positive Cases')
        pl.title('Cumulative Gain Chart')
        pl.legend(loc='lower right')
        pl.show()
        
        #-------------Sensitivity vs Specificity Curve---------#
        
        '''Calculating specificity from FPR calculated in AUROC'''    
        
        ideal_threshold = 0.0
        TNR = FPR
        for i in range(len(FPR)):    
           TNR[i] = 1 - FPR[i]
           if abs((TNR[i] - TPR[i])) <= 0.0001:
               ideal_threshold = thresholds[i]
        
        '''Plotting sensitivity vs specificity curve'''    
        print '\nIdeal threshold : %f\n' %ideal_threshold
        pl.clf()
        pl.plot(thresholds, TPR, label='Sensitivity')
        pl.plot(thresholds, TNR, label='Specificity')
        pl.xlim([-0.025, 1.0])
        pl.xlabel('Probability')
        pl.title('Sensitivity vs Specificity')
        pl.legend(loc='lower right')
        pl.show()
    
   
    return ind_SVM, prob_list