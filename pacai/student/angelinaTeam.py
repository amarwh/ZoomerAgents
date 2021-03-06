# from pacai.core import game
# from pacai.util import reflection
from pacai.core.directions import Directions
# from pacai.agents.learning.reinforcement import ReinforcementAgent
# from pacai.util import probability
import logging
import random
import time

from pacai.agents.capture.capture import CaptureAgent
from pacai.util import util


def createTeam(firstIndex, secondIndex, isRed,
        first = 'OffensiveReflexAgent',
        second = 'DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    return [
        eval(first)(firstIndex),
        eval(second)(secondIndex),
    ]

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        self.actions = []
        self.epsilon = 0.5
        self.alpha = 0.5
        self.discountRate = 1.0

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        actions = gameState.getLegalActions(self.index)

        start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        stateEval = sum(features[feature] * weights[feature] for feature in features)

        return stateEval

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that tries to keep its side Pacman-free.
    This is to give you an idea of what a defensive agent could be like.
    It is not the best or only way to make such an agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

        self.prevLocations = []

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        self.prevLocations.append(myPos)
        features['lazy'] = 0

        if len(self.prevLocations) > 3:
            for location in self.prevLocations[-3:]:
                if myPos == location:
                    features['lazy'] = 1

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

        # Compute distance to the nearest food and capsule.
        myFoodList = self.getFoodYouAreDefending(successor).asList()
        myCapList = self.getCapsulesYouAreDefending(successor)

        # This should always be True, but better safe than sorry.
        if (len(myFoodList) > 0):
            minDistanceFood = min([self.getMazeDistance(myPos, food) for food in myFoodList])
            features['foodDistance'] = minDistanceFood

        if (len(myCapList) > 0):
            minDistanceCapsule = min([self.getMazeDistance(myPos, cap) for cap in myCapList])
            features['capsuleDistance'] = minDistanceCapsule

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
            'foodDistance': -4,
            'capsuleDistance': -6,
            'stop': -100,
            'reverse': -2,
            'lazy': -850
        }

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

        self.scaredEnemies = 0
        # self.actions = []
        # self.alpha = 0.2
        # self.discountRate = 0.9
        self.prevLocations = []

    def getFeatures(self, gameState, action):
        features = {}

        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        myPos = successor.getAgentState(self.index).getPosition()

        self.prevLocations.append(myPos)
        features['lazy'] = 0

        if len(self.prevLocations) > 3:
            for location in self.prevLocations[-3:]:
                if myPos == location:
                    features['lazy'] = 1

        if self.getMazeDistance(myPos, self.prevLocations[-1]) > 1:
            features['dead'] = 1

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()
        capsuleList = self.getCapsules(successor)

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        # Calculating minimum distance to enemy
        if (len(capsuleList) > 0):
            mazeDist = []
            for capsule in capsuleList:
                mazeDist.append(self.getMazeDistance(myPos, capsule))

            minDistanceCapsule = min(mazeDist)
            features['distanceToCapsule'] = minDistanceCapsule

        # Calculating distance to enemy
        enemies = [gameState.getAgentState(enemy) for enemy in self.getOpponents(gameState)]
        if len(enemies) > 0:
            enemyDist = []
            for enemy in enemies:
                enemyDist.append(self.getMazeDistance(myPos, enemy.getPosition()))
            enemyDistance = min(enemyDist)
            features['distanceToEnemy'] = enemyDistance

            # Getting the inverse to discourage getting close to the ghost
            if enemyDistance <= 10 and enemyDistance > 0:   # checking if close to ghost
                features['distanceToEnemyInversed'] = 1 / enemyDistance

        # Condering scared ghosts
        enemies = [gameState.getAgentState(e) for e in self.getOpponents(successor)]
        scaredGhosts = [g for g in enemies if g.getScaredTimer() > 0]
        self.scaredEnemies = len(scaredGhosts)

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (action == Directions.STOP):
            features['stop'] = 1

        return features

    def getWeights(self, gameState, action):
        weights = {
            'successorScore': 100,
            'distanceToFood': -1,
            'distanceToCapsule': -2,
            'distanceToEnemy': 0,
            'distanceToEnemyInversed': -10,
            'stop': -100,
            'numInvaders': -1000,
            'dead': -500,
            'lazy': -1050
        }
        if self.scaredEnemies:
            weights['distanceToEnemy'] = -1     # starts to prioritize scared ghosts
            weights['distanceToEnemyInversed'] = 0  # forgets about keeping distance
            weights['distanceToFood'] = -2
            weights['distanceToCapsule'] = -0.5

        return weights
