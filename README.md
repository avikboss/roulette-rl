# roulette-rl

## Introduction

In roulette, the agent can bet on any of the 37 sections of the roulette wheel, and is rewarded if the parity of their bet matches the parity of the section the wheel lands on. The roulette problem is unique in that there are no states, only 38 actions (bet on one of the 37 sections, or stop playing). Games like roulette are particularly interesting to examine because lots of people find them entertaining. But like most casino games, the only way to win roulette is not to play. The optimal policy is simple - end the game without betting. But its simplicity does not mean that it is an easy task for an algorithm to learn, since the rewards are completely random for most actions.

## Algorithms 
The game is episodic, and the agent can choose to end the episode at any time, but rewards are also received throughout the episode. This makes it an excellent game on which to test several reinforcement learning algorithms. Because it is episodic, it is an ideal problem for Monte Carlo methods, and because rewards are distributed throughout the game, it may also work well with TD methods like Q Learning and Sarsa.

## Results

## Future Work