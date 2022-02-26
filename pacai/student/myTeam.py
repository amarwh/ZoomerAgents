from pacai.util import reflection
from pacai.core.directions import Directions
import logging
import random
import time

from pacai.agents.capture.capture import CaptureAgent
from pacai.agents.capture.defense import DefensiveReflexAgent
from pacai.agents.capture.offense import OffensiveReflexAgent
from pacai.agents.capture.reflex import ReflexCaptureAgent

from pacai.util import util

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.offense.OffensiveReflexAgent',
        second = 'pacai.agents.capture.defense.DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = reflection.qualifiedImport(first)
    secondAgent = reflection.qualifiedImport(second)

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]

# class ReflexCaptureAgent(CaptureAgent):
#     """
#     A base class for reflex agents that chooses score-maximizing actions.
#     """

#     def __init__(self, index, **kwargs):
#         super().__init__(index, **kwargs)

#     def chooseAction(self, gameState):
#         """
#         Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
#         """

#         actions = gameState.getLegalActions(self.index)

#         start = time.time()
#         values = [self.evaluate(gameState, a) for a in actions]
#         logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

#         maxValue = max(values)
#         bestActions = [a for a, v in zip(actions, values) if v == maxValue]

#         return random.choice(bestActions)

#     def getSuccessor(self, gameState, action):
#         """
#         Finds the next successor which is a grid position (location tuple).
#         """

#         successor = gameState.generateSuccessor(self.index, action)
#         pos = successor.getAgentState(self.index).getPosition()

#         if (pos != util.nearestPoint(pos)):
#             # Only half a grid position was covered.
#             return successor.generateSuccessor(self.index, action)
#         else:
#             return successor

#     def evaluate(self, gameState, action):
#         """
#         Computes a linear combination of features and feature weights.
#         """

#         features = self.getFeatures(gameState, action)
#         weights = self.getWeights(gameState, action)
#         stateEval = sum(features[feature] * weights[feature] for feature in features)

#         return stateEval

#     def getFeatures(self, gameState, action):
#         """
#         Returns a dict of features for the state.
#         The keys match up with the return from `ReflexCaptureAgent.getWeights`.
#         """
        
#         successor = self.getSuccessor(gameState, action)

#         return {
#             'successorScore': self.getScore(successor)
#         }

#     def getWeights(self, gameState, action):
#         """
#         Returns a dict of weights for the state.
#         The keys match up with the return from `ReflexCaptureAgent.getFeatures`.
#         """

#         return {
#             'successorScore': 1.0
#         }


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -10,
            'stop': -100,
            'reverse': -2
        }
# class OffensiveReflexAgent(ReflexCaptureAgent):
#     """
#     A reflex agent that seeks food.
#     This agent will give you an idea of what an offensive agent might look like,
#     but it is by no means the best or only way to build an offensive agent.
#     """

#     def __init__(self, index, **kwargs):
#         super().__init__(index)

#     def getFeatures(self, gameState, action):
#         features = {}
#         successor = self.getSuccessor(gameState, action)
#         features['successorScore'] = self.getScore(successor)

#         # Compute distance to the nearest food.
#         foodList = self.getFood(successor).asList()

#         # This should always be True, but better safe than sorry.
#         if (len(foodList) > 0):
#             myPos = successor.getAgentState(self.index).getPosition()
#             minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
#             features['distanceToFood'] = minDistance

#         return features

#     def getWeights(self, gameState, action):
#         return {
#             'successorScore': 100,
#             'distanceToFood': -1
#         }

# class ReflexCaptureAgent(CaptureAgent):
        
#         pass

# class OffensiveReflexAgent(ReflexCaptureAgent):
        
#         pass

# class DefensiveReflexAgent(ReflexCaptureAgent):
        
#         pass
