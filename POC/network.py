from layer import Layer
    
class NeuralsNetwork:
    
    
    
    # neural_size: La taille du réseau de neurone, tableau sous la forme :
    # [input, hidden_layer ,.., hidden_layer, output]
    def __init__(self, neural_size):
        print(neural_size)
        self.len_neural = len(neural_size)
        self.neural_list = []        
        self.neural_list.extend(
            Layer(neural_size[i], neural_size[i - 1])
            for i in range(1, self.len_neural)
        )
        
        
    def use(self, inputs):
        for layer in self.neural_list:
            inputs = layer.use(inputs)
        return inputs
        
    
    def cost(self, inputs):
        pass
    
    def costFunction(self, output, excepted):
        return self.meanSquare(output, excepted)
        
        
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Cost Function
    def meanSquare(self, output,excepted):
        error = output- excepted
        return error*error