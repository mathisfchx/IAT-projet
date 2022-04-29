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
import os 

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        #print(type(self._target))
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
        genetics.append(Genetic.Genetic(6,25,4,3))
    return genetics
#Create X number of SpaceInvader
def Initialise_SpaceInvader(number) :
    SpaceI = []
    for i in range(number) : 
        SpaceI_temp = SpaceInvader()
        SpaceI.append(SpaceI_temp)
    return SpaceI

#Create thread function 
def thread_function(Genetic, SpaceInvader , steps,pas):
    previous_state_bullet = 0
    bullet_state = 0
    is_done = False 
    compt = 0 
    tab = [[]]
    while (compt < steps+(steps-pas)*SpaceInvader.get_score() or is_done ):
        tab[0].clear()
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
        bullet_state = state[4]
        if bullet_state != previous_state_bullet and previous_state_bullet == 0 : 
            Genetic.set_score(Genetic.get_score()-0.1)
        previous_state_bullet = bullet_state
        compt += 1
        #print(index)
    #print("C'est fait !", Genetic.get_network().summary())
    #print("SpaceInvader : ", SpaceInvader.get_score())
    Genetic.set_score(Genetic.get_score()+SpaceInvader.get_score())
    return Genetic.get_score()

#sigmoide 
def sigmoid(x , max_gen):
    return 1 / (1 + np.exp((x-2*max_gen)/max_gen))

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
 

def average_from_best_result(BestResult): 
    sum = 0
    for tup in BestResult : 
        sum += tup[0]
    
    average = sum/len(BestResult)
    return average

 # find k not in a array 
def find_not_in_array(k, array) :
     for i in array :
        if i == k :
            return False
     return True
     

#Copy Weight of the best result in the Genetic
def copy_weight_and_mutate (Genetics, Best_Result , number_of_copy=10 , rate_between_mutate_and_crossover = 0.1 , mutation_ratio = 0.1) :
    index = []
    index_already_done = []
    for tup in Best_Result :
        index.append(tup[1])
    for i in index : 
        #print("l'index est : ", i,"\n \n")
        compt = 0
        for k in range(len(Genetics)) : 
            if k not in index_already_done : 
                if i != k and k not in index :
                    choice = random.uniform(0,1)
                    if choice > rate_between_mutate_and_crossover : 
                        Genetics[i].copy(Genetics[k])
                        Genetics[k].mutate_network(10,mutation_ratio)
                        index_already_done.append(k)
                        #print("On copie le génétique", i, "et on mutate")
                        #print("Le genetic qu'on a modifié est : ", k)
                        #print(index_already_done, "\n \n")
                        compt += 1
                    else : 
                        second_index = -1
                        Genetics[i].copy(Genetics[k])
                        for j in range(len(Genetics)) : 
                            if find_not_in_array(j, index_already_done) :
                                second_index = j
                                break
                        print("after for",second_index)
                        if second_index != -1 :
                            new_index = i 
                            if len(index) != 1 :
                                while new_index ==i :
                                    new_index = random.randint(0,len(index)-1)
                                
                            
                                new_wheight1 , new_wheight_2 = crossover(Genetics[i],Genetics[new_index])
                                Genetics[k].get_network().set_weights(new_wheight1)
                                Genetics[second_index].get_network().set_weights(new_wheight_2)
                            
                                index_already_done.append(k)
                                index_already_done.append(second_index)
                                #print("On copie le génétique", i, "et on crossover")
                                #print("Le genetic qu'on a modifié est : ", k)
                                #print(index_already_done, "\n \n")
                                compt += 2
                            
                if compt == number_of_copy :
                    #print(index_already_done, "\n \n")
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
        genetics.append(Genetic.Genetic(6,5,4,3))
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

def main(number_of_genetic , number_of_thread,steps , mode , increased_steps,pas):
    score = []
    for i in range(number_of_genetic) :
        print("Genetic : ", i)
        print("steps : ", steps+i*increased_steps)
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
        SpaceI = Initialise_SpaceInvader(number_of_thread)
        print("OK")
        threads = []

        for j in range(number_of_thread) :
            threads.append(ThreadWithReturnValue(target=thread_function, args=(Genetics[j], SpaceI[j],steps+j*increased_steps,pas)))
            threads[j].start()
        Returns = []
        print("\n \n")
        for j in range(number_of_thread) : 
            Returns.append((threads[j].join(),j))
            #print("join")


        print("On print les retours")    
        print("On trie")
        Best_Result = sort_array(Returns)
        #keep 3 best results
        Best_Result = Best_Result[-int(number_of_thread/10):]
        print(Returns)
        print(Best_Result)
        for Space in SpaceI : 
            Space.reset()
        for k in range(len(Genetics)) : 
            Genetics[i].reset_score()
        #print(Genetics)
        mutation_rate = max(sigmoid(i,number_of_genetic)-0.4,0.05)
        copy_weight_and_mutate(Genetics, Best_Result,int(10),0,mutation_rate)
        save_all(Genetics)
        print("On print les genetics")
        mode = 0

        score.append(average_from_best_result(Best_Result))
        if i%int(number_of_genetic/4)==0 :
            steps *= 2
        #pas = pas + 5
    SpaceInvader.save_plot_Genetic(number_of_genetic,score)


# play a game with one IA 
def play_game(Genetic, SpaceInvader, steps) :
    SpaceInvader.reset()
    is_done = False
    while is_done == False : 
        state = SpaceInvader.get_state()
        action = Genetic.predict(state)
        next_state, _, is_done = SpaceInvader.step(action)
        state = next_state
        
        
if __name__ == "__main__":
    number_of_genetic = 200
    number_of_thread = 80
    steps = 400
    mode = 1
    increased_steps = 0
    pas = int(steps*0.9)
    main(number_of_genetic,number_of_thread,steps,mode,increased_steps,pas)
    
    
