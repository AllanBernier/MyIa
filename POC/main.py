import time
from network import NeuralsNetwork
from dataset import Dataset
from dataset import DataPoint
import matplotlib.pyplot as plt


def main():

    x1 = [1,2,1,2,3,1,3,4,4,3,2]
    y1 = [1,1,2,2,1,3,2,2,1,3,3]
    
    x2 = [6,4,4,9,6,7,9,6,4,4,7,4,9]
    y2 = [5,8,9,4,8,8,9,6,4,5,4,6,9]
    dataPoints = [
        DataPoint([1,1],[1]),
        DataPoint([2,1],[1]),
        DataPoint([1,2],[1]),
        DataPoint([2,2],[1]),
        DataPoint([3,1],[1]),
        DataPoint([1,3],[1]),
        DataPoint([3,2],[1]),
        DataPoint([4,2],[1]),
        DataPoint([4,1],[1]),
        DataPoint([3,3],[1]),
        DataPoint([2,3],[1]),
        DataPoint([6,5],[0]),
        DataPoint([4,8],[0]),
        DataPoint([4,9],[0]),
        DataPoint([9,4],[0]),
        DataPoint([6,8],[0]),
        DataPoint([7,8],[0]),
        DataPoint([9,9],[0]),
        DataPoint([6,6],[0]),
        DataPoint([4,4],[0]),
        DataPoint([4,5],[0]),
        DataPoint([7,4],[0]),
        DataPoint([4,6],[0]),
        DataPoint([9,9],[0])
    ]

    plt.scatter(x1,y1, label= "stars", color= "green", 
        marker= "*", s=30)
    plt.scatter(x2,y2, label= "stars", color= "red", 
        marker= "*", s=30)


    plt.xlabel('x - axis')
    plt.ylabel('y - axis')

    plt.show()


    NN = NeuralsNetwork([2,3,1])

    print(NN.clasify([1,2]))
    
    
    
if __name__ == '__main__':
    main()


