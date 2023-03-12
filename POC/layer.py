
import math
import random

from activation import Activation

class Layer:
    

    def __init__(self, current_layer, past_layer):        
        self.past_size = past_layer
        self.current_size = current_layer
        self.weight = [[random.random() * 2 - 1 for _ in range(past_layer)]for _ in range(current_layer)]
        self.bias = [random.random() * 2 - 1 for _ in range(current_layer)] 

        self.costGradientW = [[0] * past_layer] * current_layer
        self.costGradientB = [0] * current_layer


    def calculateOutputs(self, inputs):
        weightedInputs = [] * (self.current_size )

        for nodeOut in range(self.current_size):
            weightedInput = self.bias[nodeOut]

            for nodeIn in range(self.past_size):
                weightedInput += inputs[nodeIn] * self.weight[nodeIn][nodeOut]

            weightedInputs[nodeOut+1] = weightedInput
        activations = [] * self.current_size
        for nodeOut in range(self.current_size):
            activations[nodeOut ] = Activation.activationFunction(weightedInputs, nodeOut )

        return activations

    def use(self, input):
        output = [0] * self.current_size
        for i in range(self.current_size):
            for j in range(self.past_size):
                output[i] += (input[j] * self.weight[i][j])
            output[i] += self.bias[i]
            output[i] = Activation.activationFunction(output[i])

        return output
    
    
    
    def applyGradient(self, learningRate):
        for nodeOut in range(self.current_size):
            self.bias[nodeOut] -= self.costGradientB[nodeOut] * learningRate
            for nodeIn in range(self.past_size):
                self.weight[nodeOut][nodeIn] = self.costGradientW[nodeOut][nodeIn] * learningRate

