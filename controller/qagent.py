import numpy as np
from tabulate import tabulate
#create the qagent and solve the problem
class QAgent():
    """ 
    Cette classe d'agent représente un agent utilisant la méthode du Q-learning 
    pour mettre à jour sa politique d'action.
    """
    def __init__(self, num_actions, num_state = 1600 ,alpha=0.1, gamma=0.9, epsilon=0.1, train_id = 1):
        """
        :param num_actions: nombre d'actions possibles
        :param alpha: paramètre d'apprentissage
        :param gamma: paramètre de récompense future
        :param epsilon: paramètre d'exploration
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.numstates = num_state
        self.train_id = train_id
        #load the Q matrix from the file if the file exists
        try:
            self.Q = np.load(f"qagent_{self.train_id}.npy")
        except:
            self.Q = np.zeros((self.numstates, self.num_actions))
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
        self.state = None
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
            return np.argmax(self.Q[state])

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
        #self.Q[self.state][self.action] += self.alpha * (self.reward + self.gamma * np.max(self.Q[self.next_state]) - self.Q[self.state, self.action])
        #Formule TP1
        #   print(tabulate(self.Q[self.state][self.action]))
        self.Q[state][action] = (1. - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]))
        if reward == -1:
            print(f"État de mon Q : {self.Q[state]}")

    def update_exploration_rate(self, score):
        """
        Met à jour le paramètre d'exploration en fonction du score.
        L'exploration diminue plus le score est élevé.
        :param score: score du jeu
        """
        self.epsilon = max(0.05, self.epsilon - 0.0000001)
        #print(f"Epsilon : {self.epsilon}")

    def save(self, filename = "qagent.npy"):
        """
        Sauvegarde l'état de l'agent dans un fichier.
        :param filename: nom du fichier
        """
        np.save(f"qagent_{self.train_id}.npy", self.Q)
        print("Sauvegarde effectuée")