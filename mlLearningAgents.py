# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util

# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.05, gamma=0.8, numTraining = 10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        self.qValues = {}
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.prevState = None        
        self.prevAction = None
        self.prevScore = 0

    
    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar +=1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
            return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value
        
    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts    

    # Returns q value, will return 0.0 if state not seen
    def getQValue(self, s, a):
        # Combine to get state action pair
        if ( (s,a) in self.qValues.keys() ):            
            return self.qValues[(s,a)]         
        else:
            return 0.0
    
    # Find max q values based on current legal action
    def getQMax(self, s, nextActions):
        q_accumulator = []
        if (len(nextActions) > 0):            
            for a in nextActions:
                q_accumulator.append(self.getQValue(s,a))
            return max(q_accumulator)  
        else:
            return 0   
    
    # Gets the best action
    def getBestAction(self, s, nextActions):        
        max_q = 0
        bestAction = None

        if (len(nextActions) > 0):            
            for a in nextActions:
                q = self.getQValue(s,a)
                if q > max_q or bestAction is None:
                    max_q = q
                    bestAction = a            
        
        if (bestAction == None):
            bestAction = random.choice(nextActions)

        return bestAction    

    
    
    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):

        # The data we have about the state of the game        
        
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # print "Legal moves: ", legal
        # print "Pacman position: ", state.getPacmanPosition()
        # print "Ghost positions:" , state.getGhostPositions()
        # print "Food locations: "
        # print state.getFood()
        # print "Score: ", state.getScore()
                
        s_next = state.getPacmanPosition()            
        nextActions = state.getLegalPacmanActions()

        # Perform Q learning update
        if (self.prevAction != None):
            # Get data for update
            s = self.prevState.getPacmanPosition()
            a = self.prevAction            
            # print(nextActions)
            # raw_input()
            R = state.getScore() - self.prevState.getScore()              
            Qmax = self.getQMax(s_next, nextActions)            
            
            Q = self.getQValue(s,a)        
            A = self.alpha
            Y = self.gamma          
            updateVal = Q + A * (R + Y * Qmax - Q )
            print("updateVal:" + str(updateVal))
            # Make update                                                                                   
            self.qValues[ (s,a) ] = Q + A * (R + Y * Qmax - Q )               

        
        if util.flipCoin(self.epsilon):
            pick = random.choice(legal)
        else:
            pick = self.getBestAction(s_next, nextActions)        
        

        # Save current state as previous state        
        self.prevState = state
        self.prevAction = pick

               
        
        # Now pick what action to take. For now a random choice among
        # the legal moves
        #pick = random.choice(legal)
        # We have to return an action
        return pick
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        
        print "A game just ended!"

        # Get data for final update
        s = self.prevState.getPacmanPosition()
        a = self.prevAction
        s_next = state.getPacmanPosition()            
        nextActions = state.getLegalPacmanActions()
        R = state.getScore() - self.prevState.getScore()              
        Qmax = self.getQMax(s_next, nextActions)                    
        Q = self.getQValue(s,a)        
        A = self.alpha
        Y = self.gamma

        

        # Make final update                                                                                                             
        self.qValues[ (s,a) ] = Q + A * (R + Y * Qmax - Q )  

        # Clear previous game data
        self.prevState = None
        self.prevAction = None        
        
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


