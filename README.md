# IAT-projet
To run the project, you need to install the dependencies.

```bash
pip install -r requirements.txt
```

You may run the project with the following command:

```bash
python3 run_game.py --train_id <train_id> 
```

You can also define optional parameters:

```bash
python3 run_game.py --train_id <train_id> --epsilon <epsilon> --alpha <alpha> --gamma <gamma>
```

Finally, you can test the project with the following command:

```bash
python3 run_game.py --train_id <train_id> --play
```

By adding the play parameter, the game will be played with the trained agent, epsillon will be set to 0.0

The train_id is used to save the trained agent in the folder "qagent_<train_id>.npy".

Train_id :
1 : Q-learning, state ∆x (px)
2 : Q-learning, state ∆x (px)
3 : Q-learning, state ∆x (px)
4 : Q-learning, state ∆x (px)
5 : Q-learning, state ∆x (px/(HITBOX/2))


SDI stands for "Sitting Dog Issue" : symbolizing the memory handling issue. 
Solving SDI can be done by increasing the memory size, or by increasing the number of steps during the learning.

"J'eusse obtenu de ma réflection la suivante :"

py run_game.py --train_id 20 --n_steps 5000 --n_episodes 20000 ===> ecart_X[0],self.direction(),ecart_Y[0]