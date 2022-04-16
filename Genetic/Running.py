import sys
sys.path.append('../')
from game.SpaceInvaders import SpaceInvaders as SpaceInvader
import Genetic 
import random
import numpy as np
import matplotlib.pyplot as plt
import time
from threading import Thread
import math

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


#Create X number of Genetic 
def Initialise_Genetic(number) : 
    genetics = []
    for i in range(number) : 
        genetics.append(Genetic.Genetic(3,5,4,3))
    return genetics
#Create X number of SpaceInvader
def Initialise_SpaceInvader(number) :
    SpaceI = []
    for i in range(number) : 
        SpaceI_temp = SpaceInvader()
        SpaceI.append(SpaceI_temp)
    return SpaceI

#Create thread function 
def thread_function(Genetic, SpaceInvader):
    is_done = False 
    compt = 0 
    while compt < 5000 or is_done :
        tab = [[]]
        state = SpaceInvader.get_state()
        for element in state :
            tab[0].append(element)

        action = Genetic.run(tab)
        max = -math.inf
        for i in range(len(action[0])) :
            if action[0][i] > max :
                max = action[0][i]
                index = i
        state , reward , is_done = SpaceInvader.step(index)
        compt += 1

        #print(index)
    #print("C'est fait !", Genetic.get_network().summary())
    #print("SpaceInvader : ", SpaceInvader.get_score())
    return SpaceInvader.get_score()
        

def main():
    number_of_thread = 10
    Genetics = Initialise_Genetic(number_of_thread)
    SpaceI = Initialise_SpaceInvader(number_of_thread)
    print("OK")
    threads = []
    for i in range(number_of_thread) :
        threads.append(ThreadWithReturnValue(target=thread_function, args=(Genetics[i], SpaceI[i])))
        print(i)
        threads[i].start()
        print(threads)
    Returns = []
    for i in range(number_of_thread) : 
        Returns.append((threads[i].join(),i))



    print("On print les retours")    
    print(Returns)


if __name__ == "__main__":
    main()
    
    