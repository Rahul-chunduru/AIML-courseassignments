import util
import numpy as np
import sys
import random

PRINT = True

###### DON'T CHANGE THE SEEDS ##########
random.seed(42)
np.random.seed(42)

class BaggingClassifier:
    """
    Bagging classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    
    """

    def __init__( self, legalLabels, max_iterations, weak_classifier, ratio, num_classifiers):

        self.ratio = ratio
        self.num_classifiers = num_classifiers
        self.classifiers = [weak_classifier(legalLabels, max_iterations) for _ in range(self.num_classifiers)]

    def train( self, trainingData, trainingLabels):
        """
        The training loop samples from the data "num_classifiers" time. Size of each sample is
        specified by "ratio". So len(sample)/len(trainingData) should equal ratio. 
        """

        self.features = trainingData[0].keys()
        # "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()      
        C = [ i for i in range(len(trainingData))]
        len_sample = int(self.ratio * len(trainingData))
        for i in range(self.num_classifiers):
            # for j in range(len_sample):
            #     choice = np.random.choice(C)
            #     sampleData.append(trainingData[choice])
            #     sampleLabels.append(trainingLabels[choice])
            L = [j for j in  range(len(trainingData))]  
            v = [1]* len(trainingData)
            v = util.normalize(v)
            indices = util.nSample(v , L , len_sample)  # get a sample_data_set of size ratio.
            sampleData = [trainingData[choice] for choice in indices]
            sampleLabels = [trainingLabels[choice] for choice in indices]
            self.classifiers[i].train( sampleData, sampleLabels)




    def classify( self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label. This is done by taking a polling over the weak classifiers already trained.
        See the assignment description for details.

        Recall that a datum is a util.counter.

        The function should return a list of labels where each label should be one of legaLabels.
        """

        # "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        result = []
        for datum in data:
            # vec = util.Counter()
            # vec[-1] = 0
            # vec[1] = 0
            # for i in range(self.num_classifiers):
            #     vec[self.classifiers[i].classify([datum])[0]] += 1
            #  add the labels of each classifier, give the majority
            A = 0 
            for i in range(self.num_classifiers):
            	A += self.classifiers[i].classify([datum])[0] 
            result.append(util.sign(A))

        return result
