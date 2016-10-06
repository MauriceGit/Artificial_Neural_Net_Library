#!/usr/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import math

class ANN:
    # Initialises the network with the most basic values (not yet able to do anything).
    def __init__(self, inputSize, outputSize, hiddenLayerCount, hiddenLayerSize):
        self.inputSize  = inputSize
        self.outputSize = outputSize
        self.hiddenLayerCount = hiddenLayerCount
        self.hiddenLayerSize = hiddenLayerSize

        self.trainingData = None
        self.targetData = None

        self.testInputData = None
        self.testTargetData = None

        self.batchSize = None
        self.inputCount = None

        self.debugPrints = True
        self.maxErrorRate = 0.0005
        self.maxRounds = 30000
        self.alpha = 0.02
        self.rejectionRate = 0.9
        self.maxValueDiff = 0.1

        self.syns = self._initNetwork()

    # Initialises the specified layers.
    # Possible upgrade for multilayer perceptrons:
    #   - Provide and array of hidden layer sizes, so that we can get hidden layers with different sizes.
    def _initNetwork(self):
        np.random.seed(1)
        syns = []
        syns.append(2*np.random.random((self.inputSize, self.hiddenLayerSize)) - 1)
        for i in range(self.hiddenLayerCount):
            syns.append(2*np.random.random((self.hiddenLayerSize,self.hiddenLayerSize)) - 1)
        syns.append(2*np.random.random((self.hiddenLayerSize,self.outputSize)) - 1)
        return syns

    # Training data has to be provided for training purposes.
    def setTrainingData(self, trainingData, targetData):
        self.trainingData = np.array(trainingData)
        self.targetData = np.array(targetData)
        self.inputCount = len(trainingData)
        self.batchSize = self.inputCount

    # If test-data is provided, we can automatically test the test-data after
    # the training is finished.
    def setValidationData(self, testInputData, testTargetData):
        self.testInputData = np.array(testInputData)
        self.testTargetData = np.array(testTargetData)

    # Updates the user, how the error values are changing during training phase and if the
    # the training was finished because of exceeding the round-count or minimising error rate.
    def setDebugPrints(self, debugPrints):
        self.debugPrints = debugPrints

    # Resets the default set of parameters, to influence the learning, error rates, ...
    def setParams(self, maxErrorRate, maxRounds, alpha, rejectionRate, maxValueDiff):
        self.maxErrorRate = maxErrorRate
        self.maxRounds = maxRounds
        self.alpha = alpha
        self.rejectionRate = rejectionRate
        self.maxValueDiff = maxValueDiff

    #                    .....  #
    #                .          #
    #             .             #
    #           .               #
    #          .                #
    #         .                 #
    #       .                   #
    #.....                      #
    def _sigmoid(self, x, deriv=False):
        if(deriv):
            return x*(1-x)
        return 1/(1+np.exp(-x))

    # Training the neural network until either the error rate is too low or the maximum
    # round count is exceeded.
    def _train(self, inputData, expectedData, printDebugEvery):

        dataSize = len(self.syns)

        ls = []
        l_errors = []
        l_deltas = []
        roundCount = 0

        # init:
        for i in xrange(dataSize):
            l_errors.append([])
            l_deltas.append([])

        while (roundCount < 1 or np.mean(np.abs(l_errors[dataSize-1])) > self.maxErrorRate) and roundCount < self.maxRounds:
            ls = []

            # Forward Propagation
            ls.append(inputData)
            for atLayer in xrange(dataSize):
                ls.append(self._sigmoid(np.dot(ls[len(ls)-1], self.syns[atLayer])))

            # How far are we from the perfect expected value?
            l_errors[dataSize-1] = ls[dataSize] - expectedData

            # What is the derivative and in what direction do we have to change our delta to get closer to the
            # local/global minimum?
            l_deltas[dataSize-1] = l_errors[dataSize-1] * self._sigmoid(ls[dataSize], deriv=True)

            if self.debugPrints and roundCount % printDebugEvery == 0:
                print "Error:", np.mean(np.abs(l_errors[dataSize-1]))

            for i in xrange(dataSize-2, -1, -1):
                # How much did the last weight change (delta) contribute to the error?
                l_errors[i] = l_deltas[i+1].dot(self.syns[i+1].T)

                # In what direction is our local minimum? (ls[i+1])
                # If we are good, don't change much.
                l_deltas[i] = l_errors[i] * self._sigmoid(ls[i+1], deriv=True)

            for i in xrange(dataSize-1, -1, -1):
                # Gradient descent over alpha. Change weight accordingly
                self.syns[i] -= self.alpha * np.dot(ls[i].T, l_deltas[i])

            roundCount += 1

        if self.debugPrints:
            if roundCount < self.maxRounds:
                print "-----> finished because of very small error values."
            else:
                print "-----> finished because round count was exceeded."

    # Applies the neural network to a given input. Returnes predicted values.
    def predict(self, data):
        ls = self._sigmoid(np.dot(np.array(data), self.syns[0]))
        for syn in self.syns[1:]:
            ls = self._sigmoid(np.dot(ls, syn))
        return ls

    # Compares the calculated result to the should-be target and determines,
    # if it is good enough.
    def _resultIsValid(self, result, target):
        validCount = 0
        for i in range(len(result)):
            if abs(result[i] - target[i]) <= self.maxValueDiff:
                validCount += 1
        return validCount / float(len(result)) >= self.rejectionRate

    # Tests a given dataset and returnes the count of correct predicted
    # values.
    def testDataset(self):
        correctPredicted = 0

        for i in range(len(np.array(self.testInputData))):
            test = self.testInputData[i]
            target = np.array(self.testTargetData)[i]
            # predict one dataset.
            result = self.predict(test)

            if self._resultIsValid(result, target):
                correctPredicted += 1

        return correctPredicted, 100.0*correctPredicted/len(self.testInputData)

    # Uses the provided training and target data to train the neural network.
    def train(self, printTests=True, printDebugEvery=500):

        # Batch size
        for i in range(self.inputCount/self.batchSize):
            # Trains the neural network to predit from trainingData to targetData.
            self._train(np.array(self.trainingData[i:i+self.batchSize]),  np.array(self.targetData[i:i+self.batchSize]), printDebugEvery)

        if printTests and len(self.testInputData) > 0 and len(self.testTargetData) > 0:
            correctPredicted = self.testDataset()
            print "correct predicted:", correctPredicted[0], " --> ", str(correctPredicted[1]) + "%"


    def __str__(self):
        return "{}".format("ANN stuff...")







