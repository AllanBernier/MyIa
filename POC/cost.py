
class Cost:
    def costFunction(self, output, excepted):
        return self.meanSquare(output, excepted)
    
    def costDerivate(self, output, excepted):
        self.meanSquareDerivate(self, output,excepted)
    
    
    def meanSquare(self, output,excepted):
        error = output- excepted
        return error*error
    
    
    
    def meanSquareDerivate(self, output,excepted):
        return output- excepted
    
    
    

    