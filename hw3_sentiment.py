# STEP 1: rename this file to hw3_sentiment.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
from collections import Counter
import math

"""
Your name and file comment here: Nimra Sharnez
"""


"""
Cite your sources here:
"""

"""
Implement your functions that are not methods of the Sentiment Analysis class here
"""


def generate_tuples_from_file(training_file_path):
    tuples=[]
    f = open(training_file_path, "r")
    for line in f:
        line = line.strip('\n')
        tuples.append(tuple(line.split('\t')))
    return(tuples)
    pass


def precision(gold_labels, classified_labels):
    tp=0
    fp=0
    
    for i in range(len(gold_labels)):
        if (classified_labels[i] == "1"):
            if (gold_labels[i] == "1"):   
                tp+=1
            else:
                fp+=1
                
    tpfp = fp+tp
    p = tp/(tpfp)
    
    return(p)


def recall(gold_labels, classified_labels):
    tp = 0
    fn = 0
    
    for i in range(len(gold_labels)):
        if (gold_labels[i] == "1"):
            if (classified_labels[i] == "1"):   
                tp+=1
            else:
                fn+=1
                
    tpfn = fn+tp
    
    r = tp/(tpfn)
    
    return(r)

def f1(gold_labels, classified_labels):
    
    p = precision(gold_labels, classified_labels)
    r = recall(gold_labels, classified_labels)
    
    f1 = 2*((p*r)/(p+r))
    
    return(f1)


"""
Implement any other non-required functions here
"""



"""
implement your SentimentAnalysis class here
"""
class SentimentAnalysis:


    def __init__(self):
    # do whatever you need to do to set up your class here
        self.gC = None
        self.ncG = 0
        self.bC = None
        self.ncB = 0
        self.vC = None
        
        pass

    def train(self, examples): 
        #getting count of each word in good & bad
        bad = []
        good = []
        
        
        for i in range (len(examples)): #go through all docs
            if(examples[i][2] == '0'): #if class 0
                self.ncB += 1 #increment to count all "bad" documents
                bad.append(examples[i][1]) #put all sentences in this list
            else:
                self.ncG += 1 #increment to count all the "good" documents
                good.append(examples[i][1]) #else we will put all the sentences in this list
        
        
        
        def helper(l): #helps to make the list in dict of counts
            dictOfWords={}

            for s in range(len(l)):
                for word in l[s].split():
                    try:
                        dictOfWords[word]+=1  
                    except KeyError:
                        dictOfWords.update({word : 1})
            return(dictOfWords)
        
        
        self.gC = helper(good) #documents of good w/ their counts
        self.bC = helper(bad) #documents of bad w/ their counts
        self.vC = {**self.gC , **self.bC} #count of all unique words in the documents      
        

        
        pass
    
        
    def score(self, data):
        
        v = len(self.vC) #v = unique words
        ndoc = self.ncG+self.ncB
        pG = (self.ncG)/(ndoc)
        pB = (self.ncB)/(ndoc)
        
        
        probsG = []
        probsB = []
        for word in data.split():
            if (word in self.vC): #if word in training set
                #do something
                gN = sum(self.gC.values()) #N for good
                bN = sum(self.bC.values()) #N for bad
                try:
                    numG = math.log((self.gC[word]+1)/(gN+v))
                except KeyError:
                    numG = math.log((1)/(gN+v))
                try:
                    numB = math.log((self.bC[word]+1)/(bN+v))
                except KeyError:
                    numB = math.log((1)/(bN+v))
                    
                
                probsG.append(numG)
                probsB.append(numB)

            else:
                continue
        
        
        
        b = (math.log((pB))) + (sum(probsB))
        bb = math.e**(b)
        g = (math.log((pG))) + (sum(probsG))
        gg = math.e**(g)
        dic = {"0": bb, "1": gg}
        
                

        return(dic)

    def classify(self, data):
        
        dic = self.score(data)
        
        keymax = max(dic, key=dic.get) 
        
        return(keymax)
        

        


    def featurize(self, data): #return dictionary of word with their counts as value
        l_tuples = []
         #go through the doc
        words = data.split()
        for word in words:
            tuples = []
            tuples.append(word)
            tuples.append('True')
            l_tuples.append(tuple(tuples))
        print(l_tuples)
        

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"

########### ########### ########### ########### ########### ########### ########### ########### ########### ##########
from nltk.stem import PorterStemmer 

class SentimentAnalysisImproved:

    def __init__(self):
        self.ncG = 0
        self.ncB = 0
        pass

    def train(self, examples):
        stopwords = ['and', 'the', 'a', 'because', 'to', 'too', 'you', 'he', 'him', 'his', 'her', 'she', 'is', 'i', 'was', 'me', 'by', 'hotel']
        #I made a list of stop words that one may see commonly in a hotel review list
        bad = []
        good = []
        ps = PorterStemmer() 
        #I will also be stemming the words with the PorterStemmer from nltk 
        
        for i in range(len(examples)):
            if (examples[i][2]=='0'): 
                self.ncB+=1
                b = (examples[i][1].lower().split())
                [bad.append(w) for w in b]
                
            else:
                self.ncG+=1
                g = (examples[i][1].lower().split())
                [good.append(w) for w in g]

                
        

        

        bad = [word for word in bad if word not in stopwords] #Removing all stop words
                
        good = [word for word in good if word not in stopwords] 
        
        bad = [ps.stem(ww) for ww in bad] #stemming all words
        good = [ps.stem(w) for w in good]
        
        badD ={}
        goodD ={}
        def dictionaryit(l, d):
            for word in l: 
                d[word] = l.count(word)
            return(d)
        
        self.badD = dictionaryit(bad, badD)
        self.goodD = dictionaryit(good, goodD)
        self.vC = {**self.goodD , **self.badD} #count of all unique words in the documents
        
                
        pass

    def score(self, data):
        
        v = len(self.vC) #v = unique words
        ndoc = self.ncG+self.ncB
        pG = (self.ncG)/(ndoc)
        pB = (self.ncB)/(ndoc)
        
        
        probsG = []
        probsB = []
        ps = PorterStemmer()
        
        for word in data.split():
            word = ps.stem(word)
            if (word in self.vC): #if word in training set
                #do something
                gN = sum(self.goodD.values()) #N for good
                bN = sum(self.badD.values()) #N for bad
                try:
                    numG = math.log((self.goodD[word]+1)/(gN+v))
                except KeyError:
                    numG = math.log((1)/(gN+v))
                try:
                    numB = math.log((self.badD[word]+1)/(bN+v))
                except KeyError:
                    numB = math.log((1)/(bN+v))
                    
                
                probsG.append(numG)
                probsB.append(numB)

            else:
                continue
        
        
        
        b = (math.log((pB))) + (sum(probsB))
        bb = math.e**(b)
        g = (math.log((pG))) + (sum(probsG))
        gg = math.e**(g)
        dic = {"0": bb, "1": gg}
        
        return(dic)

    def classify(self, data):
        
        dic = self.score(data)
        
        keymax = max(dic, key=dic.get) 
        
        return(keymax)


    def featurize(self, data):
        l_tuples = []
        words = data.split()
        for word in words:
            tuples = []
            tuples.append(word)
            tuples.append('True')
            l_tuples.append(tuple(tuples))
        print(l_tuples)
        
 

    def __str__(self):
        return "NAME FOR YOUR CLASSIFIER HERE"
    
########### ########### ########### ########### ########### ########### ########### ########### ########### ##########

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
        sys.exit(1)

    training = sys.argv[1]
    testing = sys.argv[2]

    sa = SentimentAnalysis()
    print(sa)
  # do the things that you need to with your base class
  

    improved = SentimentAnalysisImproved()
    print(improved)
  # do the things that you need to with your improved class

