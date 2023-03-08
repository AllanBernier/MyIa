import time
from network import NeuralsNetwork


def main():
    NN = NeuralsNetwork([1,728,16,16,10])

    
    t = round(time.time() * 1000)
    
    print(NN.use([1]))
    
    print(round(time.time() * 1000) - t)
    
    
if __name__ == '__main__':
    main()


