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
        # Count the number of games we have played
        self.episodesSoFar = 0

        # Use counter because dictionary was too slow
        self.qValues = util.Counter()
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

    # Returns q value
    def getQValue(self, s, a):        
        if a == Directions.STOP:
            return 0.0
        else:            
            return self.qValues[(s,a)]
    
    # Find max q values based on current legal action
    def getQMax(self, s, nextActions):
        q_accumulator = []
        if (len(nextActions) > 0):            
            for a in nextActions:
                q_accumulator.append(self.getQValue(s,a))
            return max(q_accumulator)  
        else:
            return 0   
    
    def qLearningUpdate(self, state):                      
        # Perform Q learning update
        if (self.prevAction != None):
            # Get data for update
            s = self.prevState
            a = self.prevAction            
            s_next = state
            nextActions = self.getNextActions(state)  
                        
            R = state.getScore() - self.prevState.getScore()              
            Qmax = self.getQMax(s_next, nextActions)                        
            Q = self.getQValue(s,a)        
            A = self.alpha
            Y = self.gamma                      
            
            # Make update                                                                                               
            self.qValues[ (s,a) ] =  Q + A * (R + Y * Qmax - Q )
    
    # Gets the best action
    def getBestAction(self, s, nextActions):        
        largest_q = 0
        bestAction = None

        if (len(nextActions) > 0):            
            for a in nextActions:
                q = self.getQValue(s,a)                
                if q > largest_q or bestAction is None:
                    largest_q = q
                    bestAction = a           
        
        # If no good action was found, do random action
        if (bestAction == None):
            bestAction = random.choice(nextActions)

        return bestAction            

    
    def getNextActions(self, state):
        nextActions = state.getLegalPacmanActions()
        if Directions.STOP in nextActions:
            nextActions.remove(Directions.STOP) 

        return nextActions

    
    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):               
        # Perform Q Learning update
        self.qLearningUpdate(state)         
        
        # Get next actions
        nextActions = self.getNextActions(state)         

        # Force exploration with epsilon               
        if random.random() < self.epsilon:
            pick = random.choice(nextActions)
        else:
            pick = self.getBestAction(state, nextActions) 

        # Save data before continuing        
        self.prevState = state
        self.prevAction = pick
        return pick
            

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        
        print "A game just ended!"

        # Final update
        self.qLearningUpdate(state)

        # Clear previous game data
        self.prevState = None
        self.prevAction = None       
                
        print "Completed runs: " + str(self.getEpisodesSoFar())        


        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg,'-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


