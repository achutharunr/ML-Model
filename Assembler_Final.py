# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 08:32:05 2016

@author: U505118
"""
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
import re
import pickle
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cPickle
import sklearn.metrics
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import sklearn.metrics
import pandas as pd
import numpy as np
import re
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

#*****************************************************************************
##############################  RF  ##########################################
#*****************************************************************************
class randomforest:
    def rf(self,df):
        #Libraries used
        '''from sklearn import metrics
        from sklearn.ensemble import RandomForestClassifier
        import pandas as pd
        from pandas import DataFrame
        from nltk import word_tokenize
        from sklearn import preprocessing
        from sklearn.metrics import roc_auc_score
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        import numpy as np
        import random
        import cPickle'''
        #READING 10-K FILES
        tr=1857
        
        df = pd.read_csv('C:/Users/U505118/Desktop/FinalReview/Final.csv')
        self.df1 = pd.concat([df['Indicator'], df['Phrase']], axis=1)
        df_train = df[:tr]        
        self.df_test = df[tr:]
        
        #CLASSIFYING INTO TRAIN AND TEST DATA
        words = df_train['Phrase'].tolist()
        wt = self.df_test['Phrase'].tolist()
        
        sw_file = open('C:/Users/U505118/Desktop/FinalReview/RandomForest/stopwords.txt') 
        stopwords = sw_file.readlines()
        for i in range(0,len(stopwords)):
            stopwords[i] = stopwords[i].strip()
        sw_file.close()
        
        #CONVERTING SENTENCES TO VECTORS SO THAT THEY CAN BE INPUT IN RF AND YES,NO TO NUMBERS       
        vect = CountVectorizer(stop_words = stopwords, token_pattern = '[a-z]+', min_df=5,max_features = 400)
        idf = vect.fit_transform(words).toarray()
        number = preprocessing.LabelEncoder()
        df['Indicator'] = number.fit_transform(df.Indicator)
        
        #IMPLEMENTING RANDOM FOREST
        rf = RandomForestClassifier(n_estimators=1000,min_samples_leaf=5)
        rf.fit(idf, df['Indicator'][:tr])
        
                
        wo_test = self.df_test['Phrase'].tolist()
       
       #CONVERTING SENTENCES TO VECTORS SO THAT THEY CAN BE INPUT IN RF AND YES,NO TO NUMBERS       
        test = vect.fit_transform(wo_test).toarray()
        number = preprocessing.LabelEncoder()
        df['Indicator'] = number.fit_transform(df.Indicator)
        t = []
        self.a = 0
        self. b = 0
        self.c=0
        self.d=0
        self.co=0
        
        #PREDICTING DATA USING RF MODEL
        t = rf.predict_proba(test)
        u = []  
        u = t.tolist()    
        
        t = []
        for i in range(0,len(u)):
            if u[i][1]>0.61:
                t.append(1)
            else:
                t.append(2)
                
        #COMBINING PREDICTED VALUE AND ACTUAL VALUE       
        k = []
        for i in range(0,len(u)):
            k.append(u[i][2])
        self.df1 = pd.DataFrame({'true':df['Indicator'][tr:],'model':t,'prob':k})
        
        #CONFUSION MATRIX
        df2 = pd.DataFrame(index = {'Model +ve','Model -ve'},columns = {'Target +ve','Target -ve'})
        for i in range(tr,2674):
            if self.df1['true'][i] == self.df1['model'][i]:
                self.co = self.co+1 
            if self.df1['model'][i] == 2 and self.df1['true'][i] == 2:
                self.a = self.a+1
            elif self.df1['model'][i] == 2 and self.df1['true'][i] == 1:
                self.b = self.b+1
            elif self.df1['model'][i] == 1 and self.df1['true'][i] == 2:
                self.c = self.c+1
            else:
                self.d = self.d+1
        
        df2['Target +ve']['Model +ve'] = self.a
        df2['Target +ve']['Model -ve'] = self.c
        df2['Target -ve']['Model +ve'] = self.b
        df2['Target -ve']['Model -ve'] = self.d 
        self.model = self.df1['model']-1
        self.prob = self.df1['prob']
        self.model = self.model.reset_index()
        self.prob = self.prob.reset_index()
        return list(self.model.model),list(self.prob.prob)
    def met(self):
            #CALCULATING SENSITIVITY,PRECISION,SPECIFICITY,ACCURACY,FALSE POSITIVE RATE
            self.pp = float(self.a)/(self.a + self.b)
            self.np = float(self.d)/(self.c + self.d)
            self.sen = float(self.a)/(self.a + self.c)
            self.spe = float(self.d)/(self.b + self.d)
            self.acc = (self.a+self.d)/float(self.a+self.b+self.c+self.d)
            print "Sensitivity and Accuracy =" 
            return self.sen,self.acc
            
    def ROC(self):  
             #PLOTTING ROC CURVE
            self.df1 = self.df1[self.df1.true!=0]
            self.df1 = self.df1.reindex()
            self.fpr, self.tpr,self.thresholds = metrics.roc_curve(self.df1['true']-1,self.df1['prob'])
            roc_auc = auc(self.fpr, self.tpr)
            print roc_auc
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            ax1.plot(self.fpr, self.tpr, 'b')
            plt.legend(shadow=True, fancybox=True,loc=1) 
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.title("ROC curve")
            plt.show()
                     
            #PLOTTING SENSITIVITY VS. SPECIFICITY CURVE
            fig2 = plt.figure()
            ax2 = fig2.add_subplot(111)
            ax2.plot(self.thresholds, self.tpr, 'b',label='Sensitivity')
            ax2.plot(self.thresholds,1-self.fpr,'r',label='1-Specificity')
            plt.legend(shadow=True, fancybox=True,loc=1) 
            plt.ylabel('Sensitivity and Specificity')
            plt.xlabel('Probability')
            plt.title("Sensitivity vs. Specificity curve")
            plt.show()
            
    def gain(self):
            #PLOTTING GAINS CHART
            TP = self.a
            FN = self.c
            self.df_test = self.df_test.reset_index()
            gain_model = [0]
            model_cumul = 0.0
            dec = len(self.df_test)/10
            dec1 = len(self.df_test)/10
            self.df_test = self.df_test.sample(n = len(self.df_test))
            self.df1.sort(columns = 'prob', inplace = True, ascending = False)
            self.df1 = self.df1.reset_index(drop = True)
            df_GL = pd.DataFrame([],columns  = ['Decile', 'Random', 'Model'])
            
            for i in range(10):
                df_GL.loc[i, 'Decile'] = str(i + 1)
                model = 0.0
                if (i==9):
                    dec1=len(self.df_test)%10
                for k in range(i*dec, (i*dec) + dec1):
                        if self.df1['true'][k] == 2:
                            model = model + 1
                model_cumul = model_cumul + model
                df_GL.loc[i, 'Model'] = str(model_cumul)
                gain_model.append(((model_cumul)/(TP + FN))*100.00)
            
            
            x=[0,10,20,30,40,50,70,90,100]            
            percent_pop = range(0, 110, 10)
            plt.plot(percent_pop, gain_model, 'b', label = "Model")
            plt.plot(x,x,'r',label="Random")
            plt.legend(shadow=True, fancybox=True,loc=1) 
            plt.ylabel('Cummulative Gain')
            plt.xlabel('Population Percentage')
            plt.title("Gain Chart")
            plt.show()  

    

#*****************************************************************************
##############################  ANN  #########################################
#*****************************************************************************
'''
This is the ANN class which implements the Artificial Neural Network.
For training the data it takes following inputs:
    1. X - Feature set
    2. y - Labelled Data
    3. alpha - This is the learning rate, default is 0.1
    4. max_iter - Maximum iterations to see the error minimized, default = 5000
For testing, it just takes the input data abd the calculated weights

Two methods of this calss are to calculate the activation function used in the NN.
1 - sigmoid -> the sigmoid function
2 - sigmoid_output_to derivative - calculates the derivative of the sigmoid function and returns it.
'''
class ANN(object):
    
    # compute sigmoid nonlinearity
    def sigmoid(self,x):
        output = 1/(1+np.exp(-x))
        return output
    
    # convert output of sigmoid function to its derivative
    def sigmoid_output_to_derivative(self,output):
        return output*(1-output)    
    
    #Train the data
    def trainNN(self,X,y,alpha=0.1,max_iter=5000):
            
        dim = len(X[0])
        #print "\nTraining With Alpha:" + str(alpha)
        np.random.seed(1)
    
        # randomly initialize our weights with mean 0
        synapse_0 = 2*np.random.random((dim,1)) - 1

        j=-1
        while j<max_iter:
            j+=1

            #Feed forward through layers 0 and 1
            layer_0 = X
            layer_1 = self.sigmoid(np.dot(layer_0,synapse_0))
    
            #Calculate the error in the layer
            layer_1_error = layer_1 - y
    
            #Absolute error
            err = np.mean(np.abs(layer_1_error))
            
            #Check if error is minimizing after every 100 iterations            
            #if not j%100:
            #    print "Error after "+str(j)+" iterations:" + str(err)
            
            #if error goes below 0.1, break            
            if err < 0.1:
                break
    
            #In what direction is the target value?
            #Check delta
            layer_1_delta = layer_1_error*self.sigmoid_output_to_derivative(layer_1)
    
            #Find the final weights
            synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))  
                   
        #print "Total Iterations =",j
        return synapse_0
    
    #Test the data
    def testNN(self,X,w):
        layer_0 = X
        layer_1 = self.sigmoid(np.dot(layer_0,w))
        return layer_1

'''
This is the WF class, to run the training and testing datasets and to get the required output.
Input while instantiation of the WF object:
    1 - A file/panda Dataframe: You can either input the path of the file or the dataframe itself. 
        Please refer readme for the structure of the Dataframs.
    2 - feature count: The default feature count is 208, user can enter required feature count.
    3 - sw: This is the stop_words text file. Default sw file is in the project folder.

'''

class WF(object):   
    
    #instantiate the WF object
    def __init__(self,file_or_df,feat=208,sw="C:\\Users\\U505118\\Desktop\\FinalReview\\ANN\\stopwords.txt"):

        self.feat = feat
        self.sw_path = sw        

        if type(file_or_df) is str:
            self.file_is = file_or_df
            self.inp_df = pd.read_csv(file_or_df)
        else:
            self.inp_df = file_or_df
        
        #Divide the training and testing data. 70% = Training, 30% = Testing.
        self.train_data = self.clean(self.inp_df[0:int(0.7*len(self.inp_df))])
        self.test_data = self.clean(self.inp_df[int(0.7*len(self.inp_df)):])

        sw_file = open(self.sw_path)    
        self.stop_words = sw_file.readlines()
        
        for i in range(0,len(self.stop_words)):
            self.stop_words[i] = self.stop_words[i].strip()

        sw_file.close()
    
        #This is the count vectorizer to convert the sentence data to features matrix
        self.cv = CountVectorizer(stop_words=self.stop_words,max_features=self.feat)    
    
    #cleanPhrse removes all the unnessary data from the sentences.
    def cleanPhrase(self,s):
        s = re.sub(r"[^A-Za-z\']", " ", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = re.sub(r" [A-Za-z] ", " ", s)
        return s.lower().strip()
    
    #Returns a random dataframe after cleaning the data
    def clean(self,data):
        phrases = data["Phrase"]
        new_phrases = []    
        for i in phrases:
            x = self.cleanPhrase(i)
            new_phrases.append(x)
        data["Phrase"] = new_phrases
        return data
    
    #train the given data with the ANN
    def trainData(self,y):
        nn_inp = self.cv.fit_transform(self.train_data["Phrase"].values).toarray()
        neural_net = ANN()
        ans = neural_net.trainNN(nn_inp,y)
        return ans
        
    #Test the given data with the ANN
    def testData(self,ans):
        test_sample = self.cv.fit_transform(self.test_data["Phrase"].values).toarray()
        neural_net = ANN()        
        test_out = neural_net.testNN(test_sample,ans)    
        return test_out
    
    #Plot the various modelling curves
    def plotCurves(self,fpr,tpr,thresholds):
        
        #ROC Curve
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
        plt.legend(shadow=True, fancybox=True, loc=4)
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.xlabel('False Positive Rate (1-Specificity)')
        plt.title("ROC Curve")
        plt.show()
        
        #Sensitivity vs. Specificity
        fig, ax1 = plt.subplots()
        ax1.plot(tpr,thresholds, 'b')
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Sensitivity', color='b')
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        ax2 = ax1.twinx()
        ax2.plot(1-fpr,thresholds,'r')
        ax2.set_ylabel('Specificity', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        plt.title("Specificity-Sensitivity Curve")
        plt.show()

        #Gain Curve
        plot_test_data=self.test_data.reset_index()
        del plot_test_data['index']
        plot_test_data['pred_scores'] = [i for j in self.predicted_prob.tolist() for i in j]
        plot_test_data = plot_test_data.sort(columns=["pred_scores"],ascending=False)
        plot_test_data=plot_test_data.reset_index()
        del plot_test_data['index']
        
        dec = len(plot_test_data)/10
        vals = [dec for i in range(1,10)]
        dec1 = len(plot_test_data)%10
        vals.append(dec1)
        responses = []        
        
        res_count=0 
        total_yes_count = 0
        for i in range(0,len(plot_test_data)):
            if not i%dec and i>0:
                responses.append(res_count)
                res_count=0
            if plot_test_data['pred_scores'][i]>=self.optimal_thresh:
                res_count+=1
                total_yes_count+=1
        
        cumm_response = [responses[0]]
        for i in range(1,len(responses)):
            cumm_response.append(responses[i]+cumm_response[i-1])
                    
        percent = []
        for i in responses:
            per = (float(i)/total_yes_count)*100
            percent.append(float("%.2f"%per))
        
        self.gains = []
        for i in cumm_response:
            per = (float(i)/total_yes_count)*100
            self.gains.append(float("%.2f" % per))

        percent_pop = range(10, 110, 10)        
        
        self.lift = []
        for i in range(0,10):
            x = self.gains[i]/percent_pop[i]
            self.lift.append(x)        
        x = [10,20,30,40,50,60,70,90,100]

        plt.plot(percent_pop, self.gains, 'b', label="Model")
        plt.plot(x,x,'r',label="Random")
        plt.legend(shadow=True, fancybox=True,loc=4)        
        plt.ylabel('Target Percentage')
        plt.xlabel('Population Percentage')
        plt.title("Gains Chart")
        plt.show()
        
        
        plt.plot(percent_pop,self.lift, 'y', label="Lift")
        plt.legend(shadow=True, fancybox=True,loc=1)        
        plt.ylabel('Cummulative Lift')
        plt.xlabel('Population Percentage')
        plt.title("Lift Chart")
        plt.show()        
        
    #Couple of getters
    def print_confusion_matrix(self):
        print "----------------------------------------------"
        print "|                       |       Model        |"
        print "----------------------------------------------"
        print "----------------------------------------------"
        print "|                       |True     |  False   |"
        print "----------------------------------------------"
        print "|                   |Yes|  %s    |    %s    |"%(self.ranks['tp'],self.ranks['fp'])
        print "  Actual            |---|---------|-----------"
        print "|                   |No |  %s     |    %s   |"%(self.ranks['fn'],self.ranks['tn'])
        print "----------------------------------------------"
        
    def get_confusion_matrix(self):
        return self.ranks
    
    #This is the run function.
    #Arguments are whether to print the curves or not. Default is False
    def run(self,print_curves=False):
        
        #Labelled O/P        
        self.y_train_scores = []
        self.y_test_scores = []        
        for i in self.train_data["Indicator"]:
            if i=="Yes":
               self.y_train_scores.append(1)
            else:
                self.y_train_scores.append(0)
        
        for i in self.test_data["Indicator"]:
            if i=="Yes":
                self.y_test_scores.append(1)
            else:
                self.y_test_scores.append(0)        
        
        #Train the data and get the weight matrix
        valid_scores = np.array([self.y_train_scores]).T
        ans = self.trainData(valid_scores)
        
        #Get the probabilities after testing the data on the obtained weights
        self.predicted_prob = self.testData(ans)
        
        #calculate the false positive rate, true positive rate and threshold matrix
        self.fpr,self.tpr,self.thresholds = sklearn.metrics.roc_curve(self.y_test_scores,self.predicted_prob)
        self.thresholds[0]=1.0
        
        #Get the optimal threshold
        max_sum = 0
        self.__optimal_thresh = 0.5
        for i in range(0,len(self.tpr)):
            temp = self.tpr[i]+(1.0-self.fpr[i])
            if temp>max_sum:
                max_sum = temp
                self.optimal_thresh = self.thresholds[i]
        

        #Keep track of all the false positives:fp, true positives:tp, false negatives:fn, true negatives:tn
        
        self.ranks = {'tp':0,'fp':0,'fn':0,'tn':0}
        ind=0
        self.final_scores = []
        for i in self.y_test_scores:
            if i==1:
                if self.predicted_prob[ind]>self.optimal_thresh:
                    self.ranks['tp'] = self.ranks['tp']+1
                    self.final_scores.append(1)
                else:
                    self.ranks['fn'] = self.ranks['fn']+1
                    self.final_scores.append(0)
            else:
                if self.predicted_prob[ind]>self.optimal_thresh:
                    self.ranks['fp'] = self.ranks['fp']+1
                    self.final_scores.append(1)
                else:
                    self.ranks['tn'] = self.ranks['tn']+1
                    self.final_scores.append(0)
            ind+=1
        
        #Return the precision, sensitivity and accuracy            
        self.precision = float(self.ranks['tp'])/(self.ranks['tp']+self.ranks['fp'])
        self.sensitivity = float(self.ranks['tp'])/(self.ranks['tp']+self.ranks['fn'])    
        self.accuracy = float(self.ranks['tp']+self.ranks['tn'])/(self.ranks['tp']+self.ranks['tn']+self.ranks['fp']+self.ranks['fn'])
        
        if print_curves:
            self.plotCurves(self.fpr,self.tpr,self.thresholds)        
        
        return ["Precision = %s"%(self.precision), "Sensitivity = %s"%(self.sensitivity), "Accuracy = %s"%(self.accuracy)]        
        
#*****************************************************************************
##############################  Maxent  ######################################
#*****************************************************************************


class MaxEnt:
    def LiftGainTable(self):
        # DF: Dataframe read from CSV
        # T: Total number of Actual Rights 
        # F: Total number of Actual Wrongs    
        # frame,yT,fT
        deciles=self.AnswerFrame.sort('y', ascending=0)   # Sorting table based on Yes Probability in descending order
        deciles=deciles.reset_index(drop=True)  # Reseting indexes to traverse by deciles
        length=deciles.__len__()    # Total Population
        sub=int(length/10)  # Population of each decile
        rem=int(length%10)  # Number of deciles to increse Population by 1 (as total population may not necessarily be divisible by 10)
        popu=[sub for item in xrange(10) if 1]  # Making list of population per decile
        popu[:rem]=[(sub+1) for item in xrange(rem) if 1] # Adding 1 to deciles to remove remainders
        last=0 # Population Traveresed until now
        self.lift=pd.DataFrame(columns=['0','1','GT','PercentRights','PercentWrongs','PercentPopulation','CumPercentRight','CumPercentPop','LiftAtdecile','TotalLift'])
        self.lift.loc[0]=[0,0,0,0,0,0,0,0,0,0]   #A dded to make graphs look pretty    
        cumr=0  # Cumulative rights
        cumw=0  # Cumulative Wrongs
        for cin in xrange(10): # As decile
            t0=0    # Number of right per decile, used to calculate decile gain, needed for decile lift
            t1=0    # Number of wrongs per decile
            end=last+popu[cin]  # Decile's ending index
            for i in xrange(last,end):
                if deciles.loc[i,'actual']=='Yes':
                    t1+=1
                else:
                    t0+=1
            t=t0+t1     # Poulation per decile same as popu[cin]
            cumr+=t1    # Cumulative Rights
            cumw+=t0    # Cumulative Wrongs
            last=end    # Shifting start point for next decile
            pr=(t1*100)/self.yT   # Percentage right wrt to total rights that are present in current decile
            pw=(t0*100)/self.fT   # Percentage wrong wrt to total wrongs that are present in current decile
            pp=(t*100)/length   # Percentage on population in current decile
            pcr=(cumr*100)/self.yT    # Perentage of cumulative rights wrt to total rights up to the current decile
            pcp=(end*100)/length    # Percentage of Population traversed till now 
            ld=(pr*100)/pp  # Lift at current decile
            cld=(pcr*100)/pcp   # Total lift 
            self.lift.loc[cin+1]=[t0,t1,t,pr,pw,pp,pcr,pcp,ld,cld]   # Adding entry of decile to dataframe. Index+1 because we filled first row with 0's
            # lift Dataframe is in the same format as lift Gain Chart at http://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/
            return self.lift   
            
    def Lift_Chart(self):   # Displays the lift Chart: either per Decile or Total depending on the value assigned to select
        # l: Dataframe output from  LiftGainTable()
        # select: Selects the chart to display: 'decile' or 'total' lift
        plt.figure()
        plt.plot(self.lift.CumPercentPop,self.lift.TotalLift, label='Lift curve' )
        plt.plot([0, 100], [100, 100], 'k--')   # Graph of random pick model, used as a refence to see how much better our model does than randomly selecting
        plt.xlim([10.0, 100.0])     # Starting from 10% as 0% population lift cannot be calculated (Div by zero)
        plt.ylim([0.0, max(self.lift.TotalLift)])    
        plt.xlabel('Cumulative % of Polulation')
        plt.ylabel('Total % Lift')
        plt.title('Total Lift Chart')
        plt.legend(loc="upper right")   # Upper right as normally the graph has a downward slope near the right end
        plt.show()
        # Sample graph:
        # Decile: http://www.analyticsvidhya.com/wp-content/uploads/2015/01/Liftdecile.png
        # Total: http://www.analyticsvidhya.com/wp-content/uploads/2015/01/Lift.png
    
    def Gain_Chart(self):    # Displays the Total Gain Chart
        # g: Dataframe output from  LiftGainTable()
        plt.figure()
        plt.plot(self.lift.CumPercentPop, self.lift.CumPercentRight, label='Gain curve' )
        plt.plot([0, 100], [0, 100], 'k--')     # Graph of random pick model, used as a refence to see how much better our model does than randomly selecting
        plt.xlim([0.0, 100.0])  # Population percentage always between 0 and 100
        plt.ylim([0.0, 100])    # Gain is always between 0 and 100. Curve always starts at 0 and ends at 100. Refer Sample Graph
        plt.xlabel('Cumulative % of Polulation')
        plt.ylabel('Cumulative % of Right')
        plt.title('Gain Chart')
        plt.legend(loc="lower right")   # Lower right as the graph touches 100 at right end
        plt.show()
        # Sample graph:
        # http://www.analyticsvidhya.com/wp-content/uploads/2015/01/CumGain.png   
        
    def ROC_Curve(self):    # Receiver Operating Characteristic Chart
        # f: False positive rate
        # t: True positive rate 
        # fpr, tpr
        plt.figure()
        plt.plot(self.fpr,self.tpr,label='ROC curve' )
        plt.plot([0, 1], [0, 1], 'k--')     # Graph of random pick model, used as a refence to see how much better our model does than randomly selecting
        plt.xlim([0.0, 1.0])    # False Positive Rate always less than 1
        plt.ylim([0.0, 1.01])   # True POsitive Rate always less than 1 however 1.01 used to visualize the top of the graph better
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")   # Curve finally reaches (1,1) freeing up the lower right corner
        plt.show()
        print 'ROC Score: ' +self.Score
        # Sample Graph:
        # http://www.analyticsvidhya.com/wp-content/uploads/2015/01/ROC.png
    
    def SenSpec(self): # Shows Sensitivity vs Cutoff and Specificity vs Cutoff on same graph
        # Point of intersecting of the two graphs is ideal threshold
        # FPR: False Positive Rate
        # SEN: Sensitivity or True Positive Rate
        # C: Thresholds
        plt.figure()
        plt.plot(self.thresholds, self.tpr*100, label='Sensitivity curve' )
        plt.plot(self.thresholds, (1-self.fpr)*100, label='Specificity curve' ) # Specificity= 1-FalsePositiveRate
        # Sensitivity and Specificity graphs change based on model. If slopes are too steep the values can saturate quickly
        # In such cases, displaying beyond the maximum can reduce the quality of the graph and impair visualization
        if (max(self.thresholds)-min(self.thresholds)) > 0.5:
            plt.xlim([0, 1])
        else:
            plt.xlim([0, max(self.thresholds)+0.05]) # +0.05 for better visualization
        plt.ylim([0.0, 100])
        plt.xlabel('Cutoff Probability')
        plt.ylabel('Percentage')
        plt.title('Sensitivity Specificity Plot')
        plt.legend(loc="lower right")
        plt.show()
        # Sample Graph:
        # http://www.analyticsvidhya.com/wp-content/uploads/2015/01/curves.png
    
    def LoadClassifier(self):
        f = open(self.PathToSavedClassifier,'rb')   # Reading has to be done in 'rb' Mode
        self.classifier = pickle.load(f)
        f.close()
        
    def LoadFeatures(self):
        f = open(self.PathToTrainPara,'rb')   # Reading has to be done in 'rb' Mode
        self.train_para = pickle.load(f)
        f.close()
        
    def ClassifySentences(self):
        test=[]
        for i in xrange(self.phrase.__len__()):
            feature={}
            for parai in self.train_para:
                if parai in self.phrase[i].lower():
                    value=True
                else:
                    value=False
                feature[parai]=value
            test.append(feature)
        answer=[]
        for featureset in test:
            pdist = self.classifier.prob_classify(featureset)
            answer.append([pdist.prob('Yes'), pdist.prob('No')])
        self.AnswerFrame=pd.DataFrame(columns=['actual','y','n'])
        master=0
        for i in xrange(self.phrase.__len__()):
            self.AnswerFrame.loc[master]=[self.indicator[i],answer[master][0],answer[master][1]]
            master+=1
    
    def FindOptimal(self):
        idx = np.argwhere(np.isclose((1-self.fpr)*1000,self.tpr*1000, atol=10)).reshape(-1)
        self.Optimal = round(self.thresholds[idx[idx.__len__()/2]],3)        
    
    def Process(self):
        self.y_scores=self.AnswerFrame.loc[:,'y']   # Model given probability of yes
        self.y_true =self.AnswerFrame.loc[:,'actual'] # Actual Yes or No 
        self.y_scores = np.array(self.y_scores)
        self.y_true=[(item=='Yes') for item in self.y_true if 1]  # Coverting from Yes No to True and False
        self.y_true_count=Counter(self.y_true).items() # Clusters True and false together in an tuple and gives its freq
        if self.y_true_count[0][0]: 
            self.yT=self.y_true_count[0][1]
            self.fT=self.y_true_count[1][1]
        else:
            self.yT=self.y_true_count[1][1]
            self.fT=self.y_true_count[0][1]
        
        self.Score=str(round(roc_auc_score(self.y_true, self.y_scores),3)) # Area under ROC
        # Score Ranges and Model Evaluation
        # 0.90-1 = excellent (A)
        # 0.80-.90 = good (B)
        # 0.70-.80 = fair (C)
        # 0.60-.70 = poor (D)
        # 0.50-.60 = fail (F)
        
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_true, self.y_scores)
    
    def CalculateConfusion(self):
        # Nomenclature: ModelPrediction Actual
        self.tt=0 # Model predicts True and is Actually True
        self.tf=0 # Model predicts True but is Actually False
        self.ff=0 # Model predicts False and is Actually False
        self.ft=0 # Model Predicts False but is Aactually True
        
        if self.UseOptimumCutoff:
            self.cutoff=self.Optimal
            print 'Using Optimal Cutoff for confusion matrix\n'
        self.output=[]
        for i in xrange(self.AnswerFrame.__len__()):
            if self.AnswerFrame.loc[i,'y']>self.cutoff:
                self.output.append((self.AnswerFrame.loc[i,'actual'],True))
                if self.AnswerFrame.loc[i,'actual']=='Yes':
                    self.tt+=1
                else:
                    self.tf+=1
            else:
                self.output.append((self.AnswerFrame.loc[i,'actual'],False))
                if self.AnswerFrame.loc[i,'actual']=='No':
                    self.ff+=1
                else:
                    self.ft+=1
        
        # How parameters are calculated: http://www.analyticsvidhya.com/wp-content/uploads/2015/01/Confusion_matrix.png
        #                               Definitions:
        # Accuracy : the proportion of the total number of predictions that were correct.
        # Positive Predictive Value or Precision : the proportion of positive cases that were correctly identified.
        # Negative Predictive Value : the proportion of negative cases that were correctly identified.
        # Sensitivity or Recall : the proportion of actual positive cases which are correctly identified.
        # Specificity : the proportion of actual negative cases which are correctly identified.
        self.accuracy= str(round(float(self.tt+self.ff)*100/float(self.tt+self.ff+self.tf+self.ft),2))
        self.precision= str(round(float(self.tt)*100/float(self.tt+self.tf),2)) # Same as Positive Predicted Value
        self.sensitivity= str(round(float(self.tt)*100/float(self.tt+self.ft),2))
        self.specificity= str(round(float(self.ff)*100/float(self.tf+self.ff),2))
        self.npv=str(round(float(self.ff)*100/float(self.ft+self.ff),2)) # Negative predicted value
        
    def ShowModel(self):
        #-----------------------------------------------------------------------
        #            Calculations of Model Metrics
        print '-----------------------------------------------------------\n'
        print 'Confusion Matrix with threshold as: '+ str(self.cutoff)+'\n'
        print '-----------------------------------------------------------\n'
        print '\t  Actual\t\n'
        print 'Model\tYes\tNo\t\n'
        print 'Yes\t'+str(self.tt)+'\t'+str(self.tf)+'\t\t'+'Precision: '+self.precision+'\n'
        print 'No\t'+str(self.ft)+'\t'+str(self.ff)+'\t\t'+'NPV: '+self.npv+'\n'
        print 'Sensitivity:\tSpecificity:\tAccuracy:\n'
        print self.sensitivity+'\t\t'+self.specificity+'\t\t'+self.accuracy+'\n'
        print '-----------------------------------------------------------\n'
        print 'Model Evaluation Metrics\n'
        print 'ROC Score: ' +self.Score
        print '-----------------------------------------------------------\n'
        print 'Optimal Cutoff: '+str(self.Optimal)+'\n'
    def PrepareForOutput(self):
        self.df=pd.DataFrame(columns=['a','p','0/1'])
        for i in xrange(self.y_scores.__len__()):
            if self.y_scores[i]>self.cutoff:
                #output.append(1)
                self.df.loc[i]=[self.y_true[i],self.y_scores[i],1]
            else:
                self.df.loc[i]=[self.y_true[i],self.y_scores[i],0]
        
    def RUN(self):
        self.LoadClassifier()
        self.LoadFeatures()
        self.ClassifySentences()
        self.Process()
        self.FindOptimal()
        self.CalculateConfusion()
        self.LiftGainTable()
        #self.ShowModel()
        self.PrepareForOutput()
        return self.df['p'].tolist(),self.df['0/1'].tolist()

    def __init__(self,P,A,ClassifierLocation,FeatureLocation,OptimumCutoff=True,c=0.44572):
        self.phrase=P
        self.indicator=A
        self.PathToSavedClassifier=ClassifierLocation
        self.PathToTrainPara=FeatureLocation
        self.UseOptimumCutoff=OptimumCutoff
        self.cutoff=c


#***************************************************************************
######################  SVM  ###############################################
#***************************************************************************

'''Class svm creates an object through which we are able to:
    -- Predict the test data
    -- Show model evaluation metrics'''

class svm(object):
      
    def __init__(self, phraseList, actualInd):    #Initialises the variables required for prediction    
        with open('C:/Users/U505118/Desktop/FinalReview/SVM/stopwords_SVM.txt','rb') as f:   #Creates a list of stopwords
            self.stopwords = f.read().split(',')                                      #from a text file 
        f.close()
        with open('C:/Users/U505118/Desktop/FinalReview/SVM/SVMVocab.txt','rb') as f:    #Creates a list of features
            self.vocab = f.read().split(',')                                      #from a text file
        f.close()
        
        self.phraseList = phraseList    #List of sentences to be classified by the model (Test data)
        self.actualInd = actualInd    #List of actual indicators to evaluate the model
        self.df_model = DataFrame()    #Creating a df to be used for model evaluation
        f.close()
        
    def Clean_Data(self):    #Cleans the test data
        for i in range(len(self.phraseList)):    #Cleaning the data by removing numbers so as to make vectorization faster
            self.phraseList[i] = re.sub('\d+', '', self.phraseList[i])
        
    def predict(self):   #Classifies the data based on model
        
        self.Clean_Data()
        
        with open('C:/Users/U505118/Desktop/FinalReview/SVM/SVMmodel.pickle', 'rb') as filehandler:    #Using pickle to import the
            model = pickle.load(filehandler)                                                    #model saved as a .pickle file
    
        '''Vectorizing the test data'''
    
        vect_test = CountVectorizer(stop_words = self.stopwords, ngram_range = (1, 2), vocabulary = self.vocab) #Using CountVectorizer
        testArray = vect_test.fit_transform(self.phraseList).toarray() #to tokenize the data with respect to the given vocabulary
        
        '''Testing the model'''
        
        testInd = model.predict(testArray)    #Predicting the class of each sentence using the model object
        self.predicted_ind = testInd.tolist()    #Converting the output of the model to a list
        
        self.indnum = []
        for x in self.predicted_ind:    #Converting the YES or NO output of the predict() to numerical 1 or 0
            if x == 'Yes':              #where 1 corrosponds to YES and 0 corrosponds to NO
                self.indnum.append(1)
            elif x == 'No':
                self.indnum.append(0)
        
        self.prob = model.predict_proba(testArray).tolist()    #Extracts the probability of each sentence to be YES
        prob_list = []                                         #as decided by the model
        for x in self.prob:
            prob_list.append(x[1])
        self.prob = prob_list 
        
        self.df_model['SVM Indicator'] = self.predicted_ind    #Appending the predicted indicator to df_model
        self.df_model['Indicator'] = self.actualInd    #Appending the actual indicator to df_model
        self.df_model['Probability'] = self.prob    #Appending the calculated probabilities to df_model
            
        self.prob = np.asarray(prob_list)   #Converting the list of probablities to a NUMPY array
        
        self.CalcBasicMetrics()    #Calculate basic metrics
        
        return self.indnum, self.prob   #Retunrs the list of 1/0 and probablities to the user
        
    def CalcBasicMetrics(self):    #Calculate TP, FP, FN, TN, FPR, TPR and thresholds
        self.TP = 0.0    #True Positive
        self.TN = 0.0    #True Negative
        self.FP = 0.0    #False Positive
        self.FN = 0.0    #False Negative
        self.sensitivity = 0.0    #Also known as Recall
        self.specificity = 0.0    #Also known as Recall
        self.precision = 0.0    #Also known as Postive Predictive Value
        self.NPV = 0.0    #Negative Predictive Value
        self.accuracy = 0.0
        
        for i in range(len(self.df_model)):
            if self.df_model.loc[i, 'Indicator'] == 'Yes':          #If the indicator is Yes:
                if self.df_model.loc[i, 'SVM Indicator'] == 'Yes':  #and predicted is YES, then TP
                    self.TP = self.TP + 1 
                elif self.df_model.loc[i, 'SVM Indicator'] == 'No': #and predicted NO, then FN
                    self.FN = self.FN + 1
            elif self.df_model.loc[i, 'Indicator'] == 'No':        #If the indicator is No:
                if self.df_model.loc[i, 'SVM Indicator'] == 'Yes': #and predicted is YES, then FP
                    self.FP = self.FP + 1
                elif self.df_model.loc[i, 'SVM Indicator'] == 'No':#and predicted is NO, then TN
                    self.TN = self.TN + 1
        
        '''Calculating actual number of positive cases'''
        
        actual = []
        for x in self.df_model['Indicator']:
            if x == 'Yes':
                actual.append(1)
            elif x == 'No':
                actual.append(0)
                
        actualvalArray = np.asarray(actual)
        del actual        
        
        self.FPR, self.TPR, self.thresholds = roc_curve(actualvalArray, self.prob)
    
    def ConfusionMatrix(self):    #Calculates and prints the confusion 
                       
        '''Calculating metrics'''
                
        self.sensitivity = (self.TP/(self.TP + self.FN))*100.0
        self.specificity = (self.TN/(self.TN + self.FP))*100.0
        self.precision = (self.TP/(self.TP + self.FP))*100.0
        self.NPV = (self.TN/(self.TN + self.FN))*100.0
        self.accuracy = (self.TP + self.TN)/(self.TP + self.TN + self.FP + self.FN)
        
        '''Printing the Confusion Matrix'''
        
        print '\n\n\n\t\t\t---------Confusion Matrix---------\n'
        print '\t\t\t\t     Predicted\n'
        print '\t\t\t\tYes\t\tNo'
        print '\t\t\t---------------------------------'
        print '\t\t\t|\t\t|\t\t|' 
        print '\t\tYes\t|\t' + str(self.TP) + '\t|\t' + str(self.FN) + '\t|  Sensitivity : %f' %self.sensitivity
        print '\t\t\t|\t\t|\t\t|'
        print '      Actual\t\t---------------------------------'
        print '\t\t\t|\t\t|\t\t|'       
        print '\t\tNo\t|\t'+ str(self.FP)+ '\t|\t' + str(self.TN) + '\t|  Specificity : %f' %self.specificity
        print '\t\t\t|\t\t|\t\t|'
        print '\t\t\t---------------------------------\n\n'
        print ('\t\tPrecisiom : %f\t\tNPV : %f\n\n' %(self.precision, self.NPV))
        print'\t\t\t\tAccuracy : %f' %self.accuracy
        
    def ROC(self):
        '''Calculating actual number of positive cases'''
        
        actual = []
        for x in self.df_model['Indicator']:
            if x == 'Yes':
                actual.append(1)
            elif x == 'No':
                actual.append(0)
                
        actualvalArray = np.asarray(actual)
        del actual
        
        '''Calculating FPR, TPR and AUCROC'''
        
        self.FPR, self.TPR, self.thresholds = roc_curve(actualvalArray, self.prob)
        self.roc_auc = roc_auc_score(actualvalArray, self.prob)
        
        
        print("Area under the ROC curve : %f" % self.roc_auc)
        
        '''Plotting the AUCROC'''
        
        pl.clf()
        pl.plot(self.FPR, self.TPR, label='ROC curve')
        pl.plot([0, 1], [0, 1], 'k--')
        pl.xlim([0.0, 1.0])
        pl.ylim([0.0, 1.0])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic')
        pl.legend(loc='lower right')
        pl.show()
        
    def GainChart(self):
        '''Declaring necessary variables'''    
        
        gain_model = [0]
        gain_random = [0]
        random_cumul = 0.0
        model_cumul = 0.0
        dec = len(self.df_model)/10
        df_GLsort = self.df_model.sort(columns = 'Probability', inplace = False, ascending = False)
        df_GLsort = df_GLsort.reset_index(drop = True)
        
        '''Dividing the data set into deciles and calculating gain'''    
        
        for i in range(10):
            random = 0.0
            model = 0.0
            for k in range(i*dec, (i*dec) + dec):
                if df_GLsort.loc[k, 'Indicator'] == 'Yes':
                    model = model + 1
                if self.df_model.loc[k, 'Indicator'] == 'Yes':
                    random = random + 1
            random_cumul = random_cumul + random
            model_cumul = model_cumul + model
            gain_model.append(((model_cumul)/(self.TP + self.FN))*100.00)
            gain_random.append(((random_cumul)/(self.TP + self.FN))*100.00)
        
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
        
    def SenSpec(self):
        
        '''Calculating specificity from FPR calculated in AUROC'''    
        
        self.ideal_threshold = 0.0
        self.TNR = self.FPR
        for i in range(len(self.FPR)):    
           self.TNR[i] = 1 - self.FPR[i]
           if abs((self.TNR[i] - self.TPR[i])) <= 0.0001:
               self.ideal_threshold = self.thresholds[i]
        
        '''Plotting sensitivity vs specificity curve'''    
        print '\nIdeal threshold : %f\n' %self.ideal_threshold
        pl.clf()
        pl.plot(self.thresholds, self.TPR, label='Sensitivity')
        pl.plot(self.thresholds, self.TNR, label='Specificity')
        pl.xlim([-0.025, 1.0])
        pl.xlabel('Probability')
        pl.title('Sensitivity vs Specificity')
        pl.legend(loc='lower right')
        pl.show()
        
    def ShowMetrics(self):
        print '\n\n'
        self.ConfusionMatrix()
        print '\n\n'
        self.ROC()
        print '\n\n'
        self.GainChart()
        print '\n\n'
        self.SenSpec()
        print '\n\n'
    

###############################################################################
##########################  MAIN  #############################################
###############################################################################

#SVM_Parse('C:/Users/U505118/Desktop/P/Final_SVM.csv')

#path = raw_input('\nPlease enter the path of the file : ')
#m = raw_input('\nWould you like the model evaluation metrics for each model to be displayed? : ')

#def Main():

df = pd.read_csv("C:/Users/U505118/Desktop/FinalReview/Final_SVM.csv")
df_test = df[1857:].reset_index(drop = True)
phraseList = df_test['Phrase'].tolist()
actualList = df_test['Indicator'].tolist()
ind_Final = []
prob = []

SVM_Model = svm(phraseList, actualList)
ind_SVM, prob_SVM = SVM_Model.predict()

maxe = MaxEnt(phraseList,actualList,'C:/Users/U505118/Desktop/FinalReview/Maxent/classifierMaxent.pickle','C:/Users/U505118/Desktop/FinalReview/Maxent/featuresMaxent.pickle')
prob_Maxent,ind_Maxent = maxe.RUN()

wf = WF("C:\\Users\\U505118\\Desktop\\FinalReview\\ANN\\Final_Achuth.csv")
wf.run()
ind_ANN, prob_ANN = wf.final_scores, wf.predicted_prob.tolist()

rf = randomforest()
ind_RF,prob_RF = rf.rf(df)

for i in range(len(ind_SVM)):
        if (ind_SVM[i] + ind_RF[i] + ind_Maxent[i] + ind_ANN[i]) >= 3:
            ind_Final.append('Yes')
        else:
            ind_Final.append('No')
        prob.append((prob_SVM[i] + prob_RF[i] + prob_Maxent[i] + prob_ANN[i])/4)
    
    
    
    
    
df_model = df_test
df_model['Model Indicator'] =  ind_Final
m = 0
       


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
        if df_model.loc[i, 'Model Indicator'] == 'Yes':
            TP = TP + 1 
        elif df_model.loc[i, 'Model Indicator'] == 'No':
            FN = FN + 1
    elif df_model.loc[i, 'Indicator'] == 'No':
        if df_model.loc[i, 'Model Indicator'] == 'Yes':
            FP = FP + 1
        elif df_model.loc[i, 'Model Indicator'] == 'No':
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
print '\t\tYes\t|\t' + str(TP) + '\t|\t' + str(FN) + '\t|  Sensitivity : %f' %sensitivity
print '\t\t\t|\t\t|\t\t|'
print '      Actual\t\t---------------------------------'
print '\t\t\t|\t\t|\t\t|'       
print '\t\tNo\t|\t'+ str(FP)+ '\t|\t' + str(TN) + '\t|  Specificity : %f' %specificity
print '\t\t\t|\t\t|\t\t|'
print '\t\t\t---------------------------------\n\n'
print ('\t\tPrecisiom : %f\t\tNPV : %f\n\n' %(precision, NPV))
print'\t\t\t\tAccuracy : %f' %accuracy

#-----------------ROC Curve-------------------------#

'''Calculating FPR, TPR and AUCROC'''

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
dec = len(df_test)/10
df_test = df_test.sample(n = len(df_test)).reset_index()
df_model['Probability'] = prob
df_model.sort(columns = 'Probability', inplace = True, ascending = False)
df_model = df_model.reset_index(drop = True)
df_GL = pd.DataFrame([],columns  = ['Decile', 'Random', 'Model'])

'''Dividing the data set into deciles and calculating gain'''

for i in range(10):
    df_GL.loc[i, 'Decile'] = str(i + 1)
    random = 0.0
    model = 0.0
    for k in range(i*dec, (i*dec) + dec):
        if df_model.loc[k, 'Indicator'] == 'Yes':
            model = model + 1
        if df_test.loc[k, 'Indicator'] == 'Yes':
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
   if abs((TNR[i] - TPR[i])) <= 0.001:
       ideal_threshold = thresholds[i]

'''Plotting sensitivity vs specificity curve'''       

pl.clf()
pl.plot(thresholds, TPR, label='Sensitivity')
pl.plot(thresholds, TNR, label='Specificity')
pl.xlim([-0.025, 1.0])
pl.xlabel('Probability')
pl.title('Sensitivity vs Specificity')
pl.legend(loc='lower right')
pl.show()

print '\nIdeal threshold : %f\n' %ideal_threshold