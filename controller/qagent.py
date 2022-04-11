import numpy as np
from tabulate import tabulate
#create the qagent and solve the problem
class QAgent():
    """ 
    Cette classe d'agent représente un agent utilisant la méthode du Q-learning 
    pour mettre à jour sa politique d'action.
    """
    def __init__(self, num_actions = 4, num_state = 1600 ,alpha=0.1, gamma=0.9, epsilon=0.1, train_id = 1):
        """
        :param num_actions: nombre d'actions possibles
        :param alpha: paramètre d'apprentissage
        :param gamma: paramètre de récompense future
        :param epsilon: paramètre d'exploration
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.max_epsilon = epsilon
        self.epsilon = epsilon
        self.numstates = num_state
        self.train_id = train_id
        print(f"QAgent : {self.train_id}")
        #load the Q matrix from the file if the file exists
        try:
            self.Q = np.load(f"qagent_{self.train_id}.npy")
        except:
            self.Q = np.zeros((20, self.numstates, self.num_actions))
            print("Q matrix not found, creating a new one")
            #self.random_fill()
        self.reset()
    #fill a 2D array with random values with sum of lines = 1
    def random_fill(self):
        for i in range(self.numstates):
            sum = 0
            for j in range(self.num_actions):
                self.Q[i][j] = np.random.rand()
                sum += self.Q[i][j]
            for j in range(self.num_actions):
                self.Q[i][j] /= sum


    def reset(self):
        """
        Réinitialise l'état de l'agent.
        """
        self.state = (None, None)
        self.action = None
        self.reward = None

    def select_action(self, state):
        """
        Sélectionne une action à partir de l'état courant.
        :param state: état courant
        :return: action choisie
        """
        if np.random.rand() < self.epsilon:
            #print("random action")
            return np.random.randint(self.num_actions)
        else:
            #print("greedy action")
            y,x = self.state
            if x == None or y == None:
                return np.random.randint(self.num_actions)
            #print(f"Q[{y},{x}] = {self.Q[y][x]}")
            return np.argmax(self.Q[y][x])

    def learn(self, env, n_episodes, max_steps):
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
        n_steps = np.zeros(n_episodes) + max_steps
        
        # Execute N episodes 
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = env.reset_using_existing_maze()
            # Execute K steps 
            for step in range(max_steps):
                # Selectionne une action 
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = env.step(action)
                # Mets à jour la fonction de valeur Q
                self.update(state, action, reward, next_state)
                
                if terminal:
                    n_steps[episode] = step + 1  
                    break

                state = next_state
            # Mets à jour la valeur du epsilon
            self.update_exploration_rate()

            # Sauvegarde et affiche les données d'apprentissage
            """if n_episodes >= 0:
                state = env.reset_using_existing_maze()
                print("\r#> Ep. {}/{} Value {}".format(episode, n_episodes, self.Q[state][self.select_greedy_action(state)]), end =" ")
                self.save_log(env, episode)

        self.values.to_csv('partie_3/visualisation/logV.csv')
        self.qvalues.to_csv('partie_3/visualisation/logQ.csv')"""

    #return sigmoide fonciton of episode
    def sigmoid(self, x):
        return (1/(1+np.exp((-x+500)/150)))
        
    def update(self, state, action, reward, next_state):
        """
        Met à jour l'état de l'agent.
        :param state: état courant
        :param action: action effectuée
        :param reward: récompense reçue
        :param next_state: état suivant
        """
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        # Q(s, a) <- Q(s, a) + alpha * [r + gamma * max(Q(s', a')) - Q(s, a)]
        self.Q[self.state][self.action] += self.alpha * (self.reward + self.gamma * np.max(self.Q[self.next_state]) - self.Q[self.state][self.action])
        #Formule TP1
        #   print(tabulate(self.Q[self.state][self.action]))
        #print(f"Q[{y},{x},{self.action}] = {self.Q[y][x][self.action]}")
        #self.Q[self.state][action] = (1. - self.alpha) * self.Q[self.state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[self.next_state]))
        if reward == -1:
            print(f"État de mon Q : {self.Q[state]}")

    def update_exploration_rate(self, nb_episode):
        """
        Met à jour le paramètre d'exploration en fonction du score.
        L'exploration diminue plus le score est élevé.
        """

        self.epsilon = max(0.05, self.max_epsilon - self.sigmoid(nb_episode))
        #self.epsilon = max(0.05, self.epsilon - 0.005)
        #print(f"Epsilon : {self.epsilon}")

    def save(self):
        """
        Sauvegarde l'état de l'agent dans un fichier.
        :param filename: nom du fichier
        """
        np.save(f"qagent_{self.train_id}.npy", self.Q)
        print("Sauvegarde effectuée")