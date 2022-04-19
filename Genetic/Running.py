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
from guppy import hpy
import os 

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
def thread_function(Genetic, SpaceInvader , steps):
    is_done = False 
    compt = 0 
    while (compt < steps or is_done ):
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

#Save all Genetic in Genetics 
def save_all(Genetics) : 
    for i in range(len(Genetics)) : 
        folder = "logs_network/network_save_"+str(i)
        file_name = folder+"/checkpoint.h5"
        Genetics[i].save_network(folder,file_name)


 #Crossover between two genetics 
 def crossover(Genetic1, Genetic2) :
     Weight1 = Genetic1.get_network().get_weights()
     Weight2 = Genetic2.get_network().get_weights()
     new_weight1 = Weight1
     new_weight2 = Weight2
     gene = random.randint(0,len(Weight1)-1)
     new_weight1[gene] = Weight2[gene]
     new_weight2[gene] = Weight1[gene]
     return new_weight1, new_weight2
 
 # find k not in a array 
def find_not_in_array(k, array) :
     for i in array :
        if i == k :
            return False
     return True
     

#Copy Weight of the best result in the Genetic
def copy_weight_and_mutate (Genetics, Best_Result , number_of_copy=10 , rate_between_mutate_and_crossover) :
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
                    if random.uniform(0,1) < rate_between_mutate_and_crossover : 
                        Genetics[i].copy(Genetics[k])
                        Genetics[k].mutate_network(10,0.05)
                        index_already_done.append(k)
                        #print("On copie le génétique", k, "et on mutate")
                        #print(index_already_done, "\n \n")
                        compt += 1
                    else : 
                        second_index = -1
                        Genetics[i].copy(Genetics[k])
                        for j in range(len(Genetics)) : 
                            if find_not_in_array(j, index_already_done) :
                                second_index = j
                                break
                        if second_index != -1 :
                            new_index = i 
                            while new_index ==i :
                                new_index = random.randint(0,len(index)-1)
                                
                            
                            new_wheight1 , new_wheight_2 = crossover(Genetics[i],Genetics[new_index])
                            Genetics[k].get_network().set_weights(new_wheight1)
                            Genetics[second_index].get_network().set_weights(new_wheight_2)
                            
                            index_already_done.append(k)
                            index_already_done.append(second_index)
                            #print("On copie le génétique", k, "et on crossover")
                            #print(index_already_done, "\n \n")
                            compt += 2
                            
                if compt == number_of_copy :
                    break
#Test crossover 
def test_crossover() :
    genetics = Initialise_Genetic(2)
    genetics[0].get_network().set_weights([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    genetics[1].get_network().set_weights([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
    new_weight1 , new_weight2 = crossover(genetics[0],genetics[1])
    print(new_weight1)
    print(new_weight2)
#Load number_of_thread from log_network.csv 
def load_network(number_of_thread) :
    genetics = []
    for i in range(number_of_thread) : 
        genetics.append(Genetic.Genetic(3,5,4,3))
    for i in range(number_of_thread) : 
        genetics[i].load_network("logs_network/network_save_"+str(i)+"/checkpoint.h5")
    return genetics



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

def main(number_of_genetic , number_of_thread,steps , mode , increased_steps):
    for i in range(number_of_genetic) :
        if mode == 1 : 
            Genetics = Initialise_Genetic(number_of_thread)
            print("On est bien dans if ")
        else :
            print("on est dans else")
            Genetics = Initialise_Genetic(number_of_thread)
            for i in range(number_of_thread) : 
                folder = "logs_network/network_save_"+str(i)
                file_name = folder+"/checkpoint.h5"
                Genetics[i].load_network(file_name)

        print(Genetics[0].get_network().get_weights())
        SpaceI = Initialise_SpaceInvader(number_of_thread)
        print("OK")
        threads = []
        for j in range(number_of_thread) :
            threads.append(ThreadWithReturnValue(target=thread_function, args=(Genetics[j], SpaceI[j],steps+j*increased_steps)))
            threads[j].start()
        Returns = []
        print("\n \n")
        h = hpy()
        print(h.heap())
        for j in range(number_of_thread) : 
            Returns.append((threads[j].join(),j))


        print("On print les retours")    
        print(Returns)
        print("On trie")
        Best_Result = sort_array(Returns)
        #keep 3 best results
        Best_Result = Best_Result[-int(number_of_thread/10):]
        print(Returns)
        print(Best_Result)
        for Space in SpaceI : 
            Space.reset()
        #print(Genetics)
            
        copy_weight_and_mutate(Genetics, Best_Result,0.2)
        save_all(Genetics)
        print("On print les genetics")
        mode = 0


if __name__ == "__main__":
    number_of_genetic = 100
    number_of_thread = 50
    steps = 2000
    mode = 0
    increased_steps = 50
    main(number_of_genetic,number_of_thread,steps,mode,increased_steps)
    
    