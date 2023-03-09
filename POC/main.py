import time
from network import NeuralsNetwork


def main():
    NN = NeuralsNetwork([1,4,2])




    for i in range (100):
        t = round(time.time() * 1000)
        for j in range(100):
            NN.learn([[1],[2],[1],[2],[1],[2]],[[1,0], [0,1], [1,0], [0,1], [1,0], [0,1]], 0.02)
        print("time + cost")
        print(round(time.time() * 1000) - t)
        print(NN.datasetCost([[1],[2]],[[1,0], [0,1]]))

    print(f"USE :{NN.use([100])}")
    print(f"USE :{NN.use([0])}")
    print(NN.neural_list[1].weight)
    print(NN.neural_list[1].bias)
    
if __name__ == '__main__':
    main()


