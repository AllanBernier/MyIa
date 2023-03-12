import math

class Activation:
    
    def activationFunction(self, weightedInput):
        return self.sigmoid(weightedInput)

    def activationFunctionDerivate(self, weightedInput):
        return self.sigmoidDerivate(weightedInput)



    def sigmoid(self, weightedInput):
        if (weightedInput > 100):
            return 1
        elif (weightedInput < -100):
            return 0
        else:
            return 1/math.exp(0-weightedInput)
        
    def sigmoidDerivate(self, weightedInput):    
        s = self.sigmoid(weightedInput)
        return s * (1 - s)
