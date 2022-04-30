import tensorflow as tf
import visualkeras as vk
import Genetic as g


#Create a Genetic 
def create_genetic(input_size, hidden_size, output_size, num_layers):
    return g(input_size, hidden_size, output_size, num_layers)

#import weights from file
def import_weight_from_file(file_name,Genetic):
    Genetic.load_network(file_name)

#Visualize the network
def visualize_network(Genetic):
    vk.layered_view(Genetic.get_network())

def main(): 
    input_size = 6
    hidden_size = 25
    output_size = 4
    num_layers = 3
    Genetic = create_genetic(input_size, hidden_size, output_size, num_layers)
    import_weight_from_file("logs_network/network_save_33/checkpoint.h5",Genetic)
    visualize_network(Genetic)

