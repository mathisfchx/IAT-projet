# IAT-projet
Pour lancer le projet, il faut tout d'abord installer les requirements

```bash
pip install -r requirements.txt
```

## QLearning

Pour lancer des parties avec une IA donnée, vous pouvez lancer la commande suivante en utilisant l'ID d'IA souhaitée.

```bash
python3 run_game.py --train_id <train_id> --play
```

Vous pouvez si vous le souhaitez passer des arguments pour lancer des phases d'apprentissage.

```bash
python3 run_game.py --train_id <train_id> --epsilon <epsilon> --alpha <alpha> --gamma <gamma>
```

Par défault, quand vous lancez avec ```--play``` le paramètre epsilon est initialisé à 0, vous etes donc en exploitation à 100%.

## Algorithme génétique

Pour lancer des parties avec une IA donnée, vous devez tout d'abord vous rendre dans le dossier ```/Genetic``` avec

```bash
cd /Genetic
```
Puis pouvez lancer la commande suivante,

```bash
./run_game.py
```

Vous serez invités à rentrer le numéro de Réseau de neurone que voius voulez utiliser.


