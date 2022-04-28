import Genetic as ge
import sys
sys.path.append('../')
import time
import math

import game.SpaceInvaders as Si 
def play() :
    i = input("Lequel voulez vous visualiser")
    i = int(i)
    print("playing todo")
    genetic = ge.Genetic(6,25,4,3)
    folder = "logs_network/network_save_"+str(i)
    file_name = folder+"/checkpoint.h5"
    genetic.load_network(file_name)
    is_done = False
    game = Si.SpaceInvaders(display=True)
    state = game.reset()
    tab = [[]]

    try :
        while is_done == False:
            tab[0].clear()
            for element in state : 
                tab[0].append(element)
            action = genetic.run(tab)
            print(action)
            max = -math.inf
            for i in range(len(action[0])) :
                if action[0][i] > max :
                    max = action[0][i]
                    index = i
            next_state, _, is_done = game.step(index)
            state = next_state
            time.sleep(0.0001)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == '__main__' :
    play()
