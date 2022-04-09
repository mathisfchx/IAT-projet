from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
import sys

def main():

    game = SpaceInvaders(display=True)
    #controller = KeyboardController()
    #controller = RandomAgent(game.na)
    controller = QAgent(num_actions=4, alpha=0.1, gamma=0.9, epsilon=0.9)
    is_done = False
    state = game.reset()
    rewards = {}
    total_rewards = 0
    tries = 0
    try :
        while is_done == False:
            action = controller.select_action(state)
            next_state, reward, is_done = game.step(action)
            #print(state)
            #exit(0)
            #playerX, playerY, invadersV, invadersH, bulletsV, bulletsH, bulletState = state
            #print("Player:", playerX, playerY, "\nInvaders:", invadersV, invadersH, "\nBullets:", bulletsV, bulletsH, "\nbullet State:", bulletState)
            #print(action, reward, is_done)
            #print("action", action)
            #print("reward", reward)
            #print("is_done", is_done)
            #exit(0)
            controller.update(int(state), action, reward, int(next_state))
            if reward == 1:
                rewards[tries] = 1
                total_rewards += 1
                controller.update_exploration_rate(game.score_val)
                print("state", state)
                print("action", action)
                print("reward", reward)
                print("next_state", next_state)
                print("is_done", is_done)
                print("\n")
            state = next_state
            tries +=1
            print("tries :", tries)
            print("total_rewards :", total_rewards)
        # print(f"state : {state}")
            
        # print(f"next state : {next_state}")
            #wait for user input before continuing
            
            #input()
            #if(game.score_val > 1000):
            #sleep(0.0001)
        controller.save() 
    except KeyboardInterrupt:
        controller.save()
        sys.exit(0)

if __name__ == '__main__' :
    main()
