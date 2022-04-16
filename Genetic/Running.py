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
    while compt < 2000 or is_done :
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


# Sort array of tuple with juste the first element of the tuple
def sort_array(array) :
    array.sort(key=lambda tup: tup[0])
    return array 


#Copy Weight of the best result in the Genetic
def copy_weight_and_mutate (Genetics, Best_Result) :
    index = []
    index_already_done = []
    for tup in Best_Result :
        index.append(tup[1])
    for i in index : 
        #print("l'index est : ", i,"\n \n")
        compt = 0
        for k in range(len(Genetics)) : 
            if k not in index_already_done : 
                if i != k :
                    Genetics[i].copy(Genetics[k])
                    Genetics[k].mutate(10)
                    index_already_done.append(k)
                    #print("On copie le génétique", k, "et on mutate")
                    #print(index_already_done, "\n \n")
                    compt += 1
                if compt == 10 :
                    break
        


#Find 3 best results in Returns 
#Divise array in two array , one wich get specific index
#and the other wich get the other index
def find_best_results(Returns, number_of_genetic) :
    best_results = []
    for i in range(number_of_genetic) :
        best_results.append([])
    for i in range(len(Returns)) :
        best_results[Returns[i][1]].append(Returns[i][0])
    return best_results

def main(number_of_genetic , number_of_thread):
    for i in range(number_of_genetic) :
        Genetics = Initialise_Genetic(number_of_thread)
        SpaceI = Initialise_SpaceInvader(number_of_thread)
        print("OK")
        threads = []
        for i in range(number_of_thread) :
            threads.append(ThreadWithReturnValue(target=thread_function, args=(Genetics[i], SpaceI[i])))
            threads[i].start()
        Returns = []
        for i in range(number_of_thread) : 
            Returns.append((threads[i].join(),i))



        print("On print les retours")    
        print(Returns)
        print("On trie")
        Best_Result = sort_array(Returns)
        #keep 3 best results
        Best_Result = Best_Result[-10:]
        print(Returns)
        print(Best_Result)
        for Space in SpaceI : 
            Space.reset()
        print(Genetics)
            
        copy_weight_and_mutate(Genetics, Best_Result)
        print("On print les genetics")


if __name__ == "__main__":
    main(1,50)
    
    