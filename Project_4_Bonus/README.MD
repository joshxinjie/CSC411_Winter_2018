# Project 4 Bonus: Tic-Tac-Toe with Self-Play

## Description
Continue training an agent to play Tic-Tac-Toe using Policy Gradient. Add additional functionalities to the machine learning agent from Project 4:

1. The agent will make the first move only 50% of the times
2. The agent will be trained by playing against a version of itself

The code is tested with Python 3 and Pytorch 0.30.

Detailed report of the project can be found [here](https://github.com/joshxinjie/CSC411_Winter_2018/blob/master/Project_4_Bonus/tictactoe_bonus.pdf).

The detailed requirements for the project can be found [here](https://www.teach.cs.toronto.edu/~csc411h/winter/projects/proj4b/).

## Files
1. tictactoe_bonus.pdf - The project report
2. tictactoe_bonus.py - The script for training and testing our agent. Also includes the game environment.
3. tictactoe_bonus.tex - The project report LaTeX file

## Summary of Results
Part a) Making the first move 50% of the times against a random computer agent, our agent achieved:
* Win Rate: 51.0%
* Tie Rate: 6.0%
* Loss Rate: 43.0%

Part b) After training against a version of itself, our agents achieved:

When starting first against a random computer opponent in 100 games:
* Win Rate: 83.0%
* Tie Rate: 4.0%
* Lose Rate: 13.0%

When starting second against a random computer opponent in 100 games:
* Win Rate: 63.0%
* Tie Rate: 3.0%
* Lose Rate: 34.0%

## Installations
Ananconda, Pytorch, OS
