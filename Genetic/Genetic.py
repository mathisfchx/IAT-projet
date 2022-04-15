import tensorflow as tf
import random
import numpy as np
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
    #initialise neural network using tf.keras 
    def init_network(self,network,input_size):
        #print("\n\n\n Init \n\n\n")
        input = tf.random.normal([1,input_size])
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




    #Change one neural network weight
    def mutate(self, mutation_number):
        for layer in self.network.layers:
            if "hidden" in layer.name : 
                array = layer.get_weights()
                self.change_array(array, mutation_number)
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
    def save_network(self,network):
        network.save('network.h5')

    def run(self , input) : 
        return self.network.predict(input)


def main():
    input_size = 4
    hidden_size = 5
    output_size = 4
    num_layers = 3
    network = Genetic(input_size, hidden_size, output_size, num_layers)
    #print(network.get_network().summary())
    print("\n\n\n Mutation \n\n\n")
    network.mutate(5)
    network.print_network()
    #create table numpy 4x4
    input = np.random.randint(0,2,(1,input_size))
    #print(input)
    result = network.run(input)
    #normalize array 
    #print(result)   
    result = result/np.max(result)
    #print(result)

    

if __name__ == """__main__""":
    main()

