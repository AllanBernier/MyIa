

class Dataset:
    
    
    def __init__(self, inputList, outputList):
        try:
            self.dataPoints = [] * len(inputList)     

            for i in range(len(inputList)):
                self.dataPoints[i] = DataPoint(inputList[i], outputList[i]) 
            
        except Exception:
            print(f"Error : {Exception}")
        
        
    def __init__(self, dataPoints):
        self.dataPoints = dataPoints        
           
class DataPoint:
    
    
    
    def __init__(self, inputValue, exceptedOutput):
        self.inputValue = inputValue
        self.outputValue = exceptedOutput        
