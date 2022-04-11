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