
import math
import random


class Layer:
    

    def __init__(self, current_layer, past_layer):        
        self.past_size = past_layer
        self.current_size = current_layer
        self.weight = [[random.random() * 2 - 1 for _ in range(past_layer)]for _ in range(current_layer)]
        self.bias = [random.random() * 2 - 1 for _ in range(current_layer)] 


    def use(self, input):
        output = [0] * self.current_size
        for i in range(self.current_size):
            for j in range(self.past_size):
                output[i] += (input[j] * self.weight[i][j])
            output[i] += self.bias[i]
            output[i] = self.activationFunction(output[i])

        return output
    
    
    
    


    def activationFunction(self, weightedInput):
        return self.sigmoid(weightedInput)
    
    
  
    
    # Activation Function 
    def sigmoid(self, weightedInput):
        if (weightedInput > 100):
            return 1
        elif (weightedInput < -100):
            return 0
        else:
            return 1/math.exp(0-weightedInput)
        
    def stepFunction(self, weightedInput):
        return 1 if weightedInput > 0 else 0
    
    def HyperbolicTangent(self, weightedInput):
        a = math.exp(2 * weightedInput)
        return (a-1)/(a+1)
    
    def SiLU(self, weightedInput):
        return weightedInput / (1 + math.exp(-weightedInput) ) 
    
    def ReLu(self, weightedInput):
        return max(weightedInput, 0)