from layer import Layer

from cost import Cost

class NeuralsNetwork:
    
    
    
    # neural_size: La taille du réseau de neurone, tableau sous la forme :
    # [input, hidden_layer ,.., hidden_layer, output]
    def __init__(self, neural_size):
        print(neural_size)
        self.len_neural = len(neural_size)
        self.layer_list = []        
        self.layer_list.extend(
            Layer(neural_size[i], neural_size[i - 1])
            for i in range(1, self.len_neural)
        )
        
    def clasify(self,inputs):
        outputs = self.calculateOutputs(inputs)
        return outputs, max(outputs)
        
    def calculateOutputs(self, inputs):
        for layer in self.layer_list:
            inputs = layer.calculateOutputs(inputs)
        return inputs
	
    
    def pointCost(self, inputs, exceptedOutput):
        output = self.use(inputs)
        return sum(
            Cost.costFunction(output[i], exceptedOutput[i])
            for i in range(len(output))
        )

    def datasetCost(self, inputsList, exceptedOutputList):
        return sum(
            self.pointCost(inputsList[i], exceptedOutputList[i])
            for i in range(len(inputsList))
        ) / len(inputsList)


    def learn(self, inputsList, exceptedOutputList, learningRate):
        h = 0.0001
        actualCost = self.datasetCost(inputsList, exceptedOutputList)
        
        for layer in self.layer_list:
            for nodeIn in range(layer.past_size):
                for nodeOut in range(layer.current_size):
                    layer.weight[nodeOut][nodeIn] += h
                    newCost = self.datasetCost(inputsList, exceptedOutputList) - actualCost
                    layer.weight[nodeOut][nodeIn] -= h
                    layer.costGradientW[nodeOut][nodeIn] = newCost / h
            
            for bias in range(layer.current_size):
                    layer.bias[bias] += h
                    newCost = self.datasetCost(inputsList, exceptedOutputList) - actualCost
                    layer.bias[bias] -= h
                    layer.costGradientB[bias] = newCost / h
            layer.applyGradient(learningRate)
        
    