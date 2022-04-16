import sys
sys.path.append('../game')
import SpaceInvaders as SpaceInvader
import Genetic 
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import threading 


#Create X number of Genetic 
def Initialise_Genetic(number) : 
    Genetic = []
    for i in range(number) : 
        Genetic.append(Genetic.Genetic())
    return Genetic
#Create X number of SpaceInvader
def Initialise_SpaceInvader(number) :
    SpaceInvader = []
    for i in range(number) : 
        SpaceInvader.append(SpaceInvader.SpaceInvader())
    return SpaceInvader

#Create thread function 
def thread_function(Genetic, SpaceInvader):
    is_done = False 
    while is_done == False :
        state,reward,is_done = SpaceInvader.get_state()
        action = Genetic.run(state)
        SpaceInvader.step(action)
        time.sleep(0.1)
    return Genetic , SpaceInvader.get_score()
        

def main():
    Genetics = Initialise_Genetic(10)
    SpaceInvaders = Initialise_SpaceInvader(10)
    for i in range(10) :
        thread = threading.Thread(target=thread_function, args=(Genetics[i], SpaceInvaders[i]))
        thread.start()

    Returns = []
    for i in range(10) : 
        Returns.append(thread.join())
        
    print(Returns)
        
    

if __name__ == "__main__":
    main()
    
    