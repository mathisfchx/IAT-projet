from time import sleep
import numpy as np
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
#import for plot 
import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser(description='Run game')
parser.add_argument('--play', action="store_true", help='Run the program without trainning parameters')
parser.add_argument("--train_id", type=int, required=True, help="Train id")
parser.add_argument("--epsilon", type=float, required=False, help="Epsilon paramètre d'exploration", default=0.8)
parser.add_argument("--gamma", type=float, required=False, help="Gamma paramètre de récompense future", default=0.9)
parser.add_argument("--alpha", type=float, required=False, help="Alpha paramètre d'apprentissage", default=0.1)
parser.add_argument("--num_state", type=int, required=False, help="Nombre d'états", default=64)
parser.add_argument("--num_action", type=int, required=False, help="Nombre d'actions", default=4)
parser.add_argument("--n_episodes", type=int, required=False, help="Nombre d'épisodes", default=1000)
parser.add_argument("--n_steps", type=int, required=False, help="Nombre d'étapes", default=1000)

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
    controller = QAgent(num_actions=args.num_action, num_state=args.num_state, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, train_id=args.train_id)
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
            
            #controller.update(state, action, reward, next_state)
            controller.update_exploration_rate(tries)
            if reward > 0 and not args.play:
                rewards[tries] = reward
                total_rewards += reward
                if reward == 1:
                    
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
#check if a 3D matrix is empty  (full 0 )
def is_empty(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            for k in range(len(matrix[i][j])):
                if matrix[i][j][k] != 0:
                    return False

def play() :
    print("playing todo")
    args.epsilon = 0.0
    controller = QAgent(num_actions=args.num_action, num_state=args.num_state, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, train_id=args.train_id)
    print(is_empty(controller.Q))

    is_done = False
    game = SpaceInvaders(display=args.play)
    state = game.reset()
    try :
        while is_done == False:
            action = controller.select_action(state)
            next_state, _, is_done = game.step(action)
            state = next_state
            sleep(0.0001)
    except KeyboardInterrupt:
        sys.exit(0)

def learn(n_episodes = 1000, max_steps=5000):
    """Cette méthode exécute l'algorithme de q-learning. 
        Il n'y a pas besoin de la modifier. Simplement la comprendre et faire le parallèle avec le cours.
        :param env: L'environnement 
        :type env: gym.Envselect_action
        :param num_episodes: Le nombre d'épisode
        :type num_episodes: int
        :param max_num_steps: Le nombre maximum d'étape par épisode
        :type max_num_steps: int
        # Visualisation des données
        Elle doit proposer l'option de stockage de (i) la fonction de valeur & (ii) la Q-valeur 
        dans un fichier de log
        """
    controller = QAgent(num_actions=args.num_action, num_state=args.num_state, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, train_id=args.train_id, episodes=n_episodes)
    is_done = False
    game = SpaceInvaders(display=args.play)
    
    n_steps = np.zeros(n_episodes) + max_steps
    courbe_1 = []
    episode_1 = []
    score_avg = 0
    initial_score_val =game.score_val
    window_size = 30
    sliding_window = []
    #fill a array with 0 until have x 0 
    for i in range(window_size):
        sliding_window.append(0)
    logs = np.zeros((n_episodes,args.num_action))
    # Execute N episodes 
    try:
        for episode in range(n_episodes):
            print("Episode :", episode)
            print("Epsilon :", controller.epsilon)
            # Reinitialise l'environnement
            state = game.reset()
            
            action_logs = np.zeros(args.num_action)
            # Execute K steps 
            for step in range(max_steps):
                # Selectionne une action 
                action = controller.select_action(state)
                #update the action log
                action_logs[action] += 1
                # Echantillonne l'état suivant et la récompense
                next_state, reward, is_done = game.step(action)
                # Mets à jour la fonction de valeur Q
                controller.update(state, action, reward, next_state)
                if is_done:
                    n_steps[episode] = step + 1  
                    break
                state = next_state
            print("Score :", game.score_val)
            print(step)
            if episode < 10:
                sliding_window[episode] = game.score_val
                courbe_1.append(game.score_val)
                episode_1.append(episode)
            else:
                sliding_window[episode % window_size] = game.score_val
                if initial_score_val == 0 :     
                    initial_score_val = game.score_val
                
                courbe_1.append(game.sliding_average( initial_score_val ,courbe_1[episode-1],episode,sliding_window))
                episode_1.append(episode)
            # Save the logs
            logs[episode] = action_logs
            # Mets à jour la valeur du epsilon
            controller.update_exploration_rate(nb_episode = episode)
            # Sauvegarde les données
            controller.save()

        # Visualisation des données

        # save the logs
        np.savetxt(f"logs/logs_{args.train_id}.csv", logs, delimiter=",")

        
        #print(courbe_1)
        #print(episode_1)
        plt.savefig('foo.png')
        #game.plot_score(courbe_1, episode_1)
        game.save_plot(courbe_1, episode_1, args.train_id)
    except KeyboardInterrupt:
        controller.save()
        np.savetxt(f"logs/logs_{args.train_id}.csv", logs, delimiter=",")
        sys.exit(0)


if __name__ == '__main__' :
    if args.play:
        main()
        play()
    else:
        learn(n_episodes = args.n_episodes, max_steps=args.n_steps)
