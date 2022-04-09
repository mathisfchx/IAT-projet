from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
import sys
import argparse

parser = argparse.ArgumentParser(description='Run game')
parser.add_argument('--play', action="store_true", help='Run the program without trainning parameters')
parser.add_argument("--train_id", type=int, required=True, help="Train id")
parser.add_argument("--epsilon", type=float, required=False, help="Epsilon", default=0.8)
parser.add_argument("--gamma", type=float, required=False, help="Gamma", default=0.9)
parser.add_argument("--alpha", type=float, required=False, help="Alpha", default=0.1)

args = parser.parse_args()
def main(eps = 0):

    game = SpaceInvaders(display=args.play)
    #controller = KeyboardController()
    #controller = RandomAgent(game.na)
    if args.play:
        print("Playing")
        args.epsilon = 0.0
    if eps > 0:
        args.epsilon = eps
    controller = QAgent(num_actions=4, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, train_id=args.train_id)
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
            controller.update_exploration_rate(game.score_val)
            if reward > 0 and not args.play:
                rewards[tries] = reward
                total_rewards += reward
                if reward == 10:
                    
                    print("state", state)
                    print("action", action)
                    print("reward", reward)
                    print("next_state", next_state)
                    print("is_done", is_done)
                    print("Q", controller.Q[state])
                    print("eps", controller.epsilon)
                    print("tries :", tries)
                    print("total_rewards :", total_rewards)
                    print("\n")
            state = next_state
            tries +=1
            

        # print(f"state : {state}")
            
        # print(f"next state : {next_state}")
            #wait for user input before continuing
            
            #input()
            #if(game.score_val > 1000):
            if args.play:
                sleep(0.0001)
        controller.save() 
        return controller.epsilon
    except KeyboardInterrupt:
        controller.save()
        sys.exit(0)

if __name__ == '__main__' :
    eps = 0
    while True:
        eps = main(eps)
