# Project 4: Tic-Tac-Toe with Policy Gradient

Use Reinforcement Learning, specifically policy gradient, to train a neural network to play Tic-Tac-Toe. The agent will be training against a random computer opponent, and our agent will always make the first move.

Unlike the first 3 projects in the course, this project includes a considerable amount of starter code, a large proportion of which is the tic-tac-toe game environment.

The code is tested with Python 3 and Pytorch 0.30.

For the detailed report of the project, click [here](https://github.com/joshxinjie/CSC411_Winter_2018/blob/master/Project_4/tictactoe.pdf)

The detailed requirements for the project can be found at [link](https://www.teach.cs.toronto.edu/~csc411h/winter/projects/proj4/).

## Summary of Results
In 100 games against a random computer opponent while making the first move, our trained agent achieved:

1. Win Rate: 83.0%
2. Tie Rate: 3.0%
3. Loss Rate: 14.0%

## Installations
Anaconda, Pytorch, OS

## Bonus Project
The bonus problem for this project involves 2 modifications for our agent:
1. Make the first move only 50% of the time against a random computer agent
2. Implementing self-play where our agent trains against a version of itself
