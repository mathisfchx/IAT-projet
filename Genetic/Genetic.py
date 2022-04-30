import tensorflow as tf
import random
import numpy as np
import os


#import pytorch as py
#Genetic algorithm with neural network


#Create neural network 
#Input : int int int int
#Network : 2 hidden layers & Relu layer & Linear layer
#Output: int int int int 


class Genetic:
    def __init__(self,input_size, hidden_size, output_size, num_layers):
        self.network = tf.keras.Sequential(
            [ 
                tf.keras.layers.InputLayer(input_size , name = 'input'),
                tf.keras.layers.Dense(hidden_size, activation="relu", name ="hidden_layer_1"),
                tf.keras.layers.Dense(hidden_size, activation="relu", name ="hidden_layer_2"),
                tf.keras.layers.Dense(output_size, activation="linear", name ="output_layer") 
            ])
        self.init_network(self.network,input_size)
        self.score = 0
    #initialise neural network using tf.keras 
    def init_network(self,network,input_size):
        #print("\n\n\n Init \n\n\n")
        input = tf.random.uniform([1,input_size])
        #print(input)
        output = network(input)

    #swap two weights in array
    def swap_weights(self,array, i_1, j_1, i_2, j_2):
        try : 
            temp = array[0][i_1][j_1]
            array[0][i_1][j_1] = array[0][i_2][j_2]
            array[0][i_2][j_2] = temp
        except:
            print("error")
            pass
    #change elements in 2D array with probability of mutation_rate
    def change_array(self,array, mutation_number):
        number = 0
        while number < mutation_number: 
            i_1 = int(random.random()*len(array))
            j_1 = int(random.random()*len(array[0]))
            i_2 = int(random.random()*len(array))
            j_2 = int(random.random()*len(array[0]))
            if i_1 != i_2 and j_1 != j_2:
                    self.swap_weights(array, i_1, j_1, i_2, j_2)
                    number += 1
        return array
    
    # add to a weight (pick in function of mutation rate) a random number between -0.5 and 0.5
    def change_array_2(self,array,mutation_rate):
        if len(array) != 0 and len(array[0]) != 0  :
            for i in range(len(array)):
                for j in range(len(array[0])):
                    if random.random() < mutation_rate:
                        change = random.uniform(-0.5,0.5)
                        try : 
                            array[i][j] += change
                        except:
                            array[0][j] += change
        else : 
            print("error")
        return array

    #Copy all weight from one genetic to a second one 
    def copy(self, genetic):
        for layer in self.network.layers: 
                array = layer.get_weights()
                for layer_2 in genetic.network.layers: 
                    if layer_2.name == layer.name : 
                        layer_2.set_weights(array)


                
    #Change one neural netork weight in function of mutation_ratio 
    def mutate_network(self, mutation_number=10 , mutation_ratio=0.1):
        for layer in self.network.layers:
                array = layer.get_weights()
                #print("mutate network array 1 : ", array , layer.name)
                #print("\n \n \n") 
                #array = self.change_array(array, mutation_number)
                array = self.change_array_2(array, mutation_ratio)
                #print("mutate network array 2 : ", array)
                #print("\n \n \n")
                layer.set_weights(array)

    def get_network(self):
        return self.network
    #print neural network
    def print_network(self):
        for layer in self.network.layers:
            #print("\n")
            #print(layer.name)
            #print(layer.weights)
            #print("\n")
            #print(layer.weights[0][1])
            pass

    #Save neural network on file 
    def save_network(self,folder_name,file_name):
        try : 
            os.mkdir(folder_name)
            print("Mkdir ok")
        except:
            pass
        self.network.save_weights(file_name)
    
    #Load network
    def load_network(self,file_name):
        self.network.load_weights(file_name)

    def run(self , input) : 
        temp = np.asarray(input)
        tensor = tf.convert_to_tensor(temp)
        return self.network(tensor)
    #Change score of genetic
    def set_score(self,score):
        self.score = score
    #get score of genetic
    def get_score(self):
        return self.score
    #reset score of genetic
    def reset_score(self):
        self.score = 0
    


def main():
    input_size = 6
    hidden_size = 25
    output_size = 4
    num_layers = 3
    network = Genetic(input_size, hidden_size, output_size, num_layers)
    #print(network.get_network().summary())
    print("\n\n\n Mutation \n\n\n")
    network.mutate_network(1,0.1)
    network.print_network()
    #create table numpy 4x4
    input = np.random.randint(0,2,(1,input_size))
    print(input)
    result = network.run(input)
    #normalize array 
    #print(result)   
    result = result/np.max(result)
    #print(result)

    

if __name__ == """__main__""":
    main()

