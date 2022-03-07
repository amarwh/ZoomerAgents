from pacai.core import game
from pacai.util import reflection
from pacai.util import priorityQueue
from pacai.core.directions import Directions
import logging
import random
import time
import sys

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


class starAgent(CaptureAgent):
    '''
    A base class for star agent that chooses cost effective actions
    based on A* pathfinding

    Credit: pacai.core.distanceCalculator.py 
    '''

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        self.fScore = {} 
        self.gScore = {}
        self.hScore = {}
        self.parent = {}

    def h(self, gameState):
        pass

    def pathfinding(self, start, end):
        node = end
        path = [node]
        while self.parent[node] is not start:
            node = self.parent[node]
            path.append((node))
        path.append(start)
        return path[-1]

    
    def aStar(self, gameState, start, end):
        '''
        Runs A* Search on all nodes to find the most cost effective path
        '''



        layout = gameState.getInitialLayout()
        
        graph = {}
        allNodes = layout.walls.asList(False)
        self.parent = {} # for backtracking path
        closed = {}

        for node in allNodes:
            self.fScore[node] = sys.maxsize
            self.gScore[node] = sys.maxsize 
            self.hScore[node] = sys.maxsize

        queue = priorityQueue.PriorityQueue()
        queue.push(start, 0)
        self.gScore[start] = 0

        while not queue.isEmpty():
            node = queue.pop()
            if node in closed:
                continue
            # to prevent repeating a node
            closed[node] = True 
            adjacent = []
            x, y = node

            # Up - North
            if not layout.isWall((x, y + 1)):
                adjacent.append((x, y + 1))
                parent[(x, y + 1)] = node

            # Down - South
            if not layout.isWall((x, y - 1)):
                adjacent.append((x, y - 1))
                parent[(x, y - 1)] = node

            # Right - East
            if not layout.isWall((x + 1, y)):
                adjacent.append((x + 1, y))
                parent[(x + 1, y)] = node

            # Left - West
            if not layout.isWall((x - 1, y)):
                adjacent.append((x - 1, y))
                parent[(x - 1, y)] = node

        



    def chooseAction(self, gameState):
        """
        Picks the next step in the optimal path returned from A* Search pathfinding
        """

        actions = gameState.getLegalActions(self.index)
        start = time.time()
        #values = [self.evaluate(gameState, a) for a in actions]
        logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        #maxValue = max(values)
        #bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        #return random.choice(bestActions)
        return random.choice(actions) # dummy return statement

    
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


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        actions = gameState.getLegalActions(self.index)
        print(actions)
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

    def getFeatures(self, gameState, action):
        """
        Returns a dict of features for the state.
        The keys match up with the return from `ReflexCaptureAgent.getWeights`.
        """
        
        successor = self.getSuccessor(gameState, action)

        return {
            'successorScore': self.getScore(successor)
        }

    def getWeights(self, gameState, action):
        """
        Returns a dict of weights for the state.
        The keys match up with the return from `ReflexCaptureAgent.getFeatures`.
        """

        return {
            'successorScore': 1.0
        }


class Graph:

    '''
    Credit: https://stackabuse.com/basic-ai-concepts-a-search-algorithm/
    
    Map of adjacencyList
        A, B, C are the keys to the node values. 
        Try coordinates as keys instead?

    adjacencyList = {
        'A': [('B', 1), ('C', 3), ('D', 7)],
        'B': [('D', 5)],
        'C': [('D', 12)]
    }
    '''    

    def __init__(self, adjacencyList):
        self.adjacencyList = adjacencyList

    def addEdge(self, pos, edge):
        self.adjacencyList[pos] = edge

    def getEdge(self, v):
        return self.adjacencyList[v]

    def h(self, n):
        pass


    
    def aStarAlgorithm(self, startNode, stopNode):
        '''
        The startNode will always be the current position of the agent
        The stopNode will always be the goal position
        '''

        # List of visited nodes whose edges haven't been inspected
        openList = set([startNode])
        # List of visited nodes whose edges have been inspected
        closedList = set([])
        
        # Contains current distances from startNode to all other nodes 
        # the default value (if not found in the map) is +infinity
        g = {}
        # self.distancer.cache

        g[startNode] = 0

        parents = {}
        parents[startNode] = startNode

        while len(openList) > 0:
            n = None
            
            # find a node with the lowest value of f() - evaluation function
            for v in openList:
                if n == None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v;
            
            if n == None:
                print('Path does not exist') # Switch to Q-learning
                return None

            # if the current node is the goal node, 
            # reconstruct the path from current node to the startNode
            if n == stopNode:
                reconstPath = []

                while parents[n] != n:
                    reconstPath.append(n)
                    n = parents[n]

                reconstPath.append(startNode)
                reconstPath.reverse()

                print('Path found: {}'.format(reconstPath))
                return reconstPath
            
            for (m, weight) in self.getEdge(n):
                # if the current node isn't in both openList and closedList,
                # add it to openList and note n as its parent
                if m not in openList and m not in closedList:
                    openList.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                # otherwise, check if it's quicker to first visit n, then m
                # and if it is, update parent data and g data
                # and if the node was in the closedList, move it to openList
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n

                        if m in closedList:
                            closedList.remove(m)
                            openList.add(m)

            # remove n from the openList, and add it to closedList
            # because all of n's edges were inspected
            openList.remove(n)
            closedList.add(n)
        
        print('Path does not exist') # Switch to Q-learning agent
        return None



class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food.
    This agent will give you an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

        self.scaredEnemies = 0
        self.onDefense = 0
        self.loopCounter = 0
        self.agentState = []

    def getFeatures(self, gameState, action):
       


        features = {}

        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()


         # loop problem

        if (len(self.agentState) < 20):
            pass
        else:

            if self.agentState[-1] and not myState.isPacman():
                self.onDefense = 1
                self.loopCounter = 5

        if myState.isPacman():
            self.agentState.append(True)
        else:
            self.agentState.append(False)


        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()
        capsuleList = self.getCapsules(successor)

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        # Calculating minimum distance to enemy
        if (len(capsuleList) > 0):
            minDistanceCapsule = min([self.getMazeDistance(myPos, capsule) for capsule in capsuleList])
            features['distanceToCapsule'] = minDistanceCapsule

        # Calculating distance to enemy
        enemies = [gameState.getAgentState(enemy) for enemy in self.getOpponents(gameState)]
        for enemy in enemies:
            #print("Enemy position")
            #print(enemy.getPosition())
            print(myPos)
        if len(enemies) > 0:
            enemyDistance = min([self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemies]) 
            features['distanceToEnemy'] = enemyDistance
            if enemyDistance <= 10 and enemyDistance > 0: # checking if close to ghost
                features['distanceToEnemyInversed'] = 1 / enemyDistance 
                # Getting the inverse to discourage getting close to the ghost, more incentive the closer

        # Considering scared ghosts
        enemies = [gameState.getAgentState(e) for e in self.getOpponents(successor)]
        scaredGhosts = [g for g in enemies if g.getScaredTimer() > 0 and not g.isPacman() and g.getPosition() is not None]
        self.scaredEnemies = len(scaredGhosts)

        # Computes distance to invaders we can see.
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
            'distanceToEnemy': 0, # regular distance to closest enemy
            'distanceToEnemyInversed': -10,
            'stop': -100,
            'numInvaders': -1000
        }
        
        if self.scaredEnemies:
            weights['distanceToEnemy'] = -10 # starts to prioritize scared ghosts
            weights['distanceToEnemyInversed'] = 0 # forgets about keeping distance 
            weights['distanceToFood'] = -2
        
        if self.onDefense and not self.agentState[-1]:
            # print("defending")
            # print(self.onDefense)
            # print(self.loopCounter)
            # print("-")
            weights['stop'] = 1000
            self.loopCounter -= 1
            if self.loopCounter == 1:
                self.onDefense = 0


        return weights 





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

        #
        # Compute distance to the nearest food and capsule.
        myFoodList = self.getFoodYouAreDefending(successor).asList()
        myCapList = self.getCapsulesYouAreDefending(successor)
        # This should always be True, but better safe than sorry.
        if (len(myFoodList) > 0):
            minDistanceFood = min([self.getMazeDistance(myPos, food) for food in myFoodList])
            features['foodDistance'] = minDistanceFood #average distance maybe?
        #
        if (len(myCapList) > 0):
            minDistanceCapsule = min([self.getMazeDistance(myPos, cap) for cap in myCapList])
            features['capsuleDistance'] = minDistanceCapsule #average distance maybe?
        #
        #
        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def getOpponentLocations(self, gameState):
        return [gameState.getAgentPosition(enemy) for enemy in self.getOpponents(gameState)]

    def getWeights(self, gameState, action):
        return {
            'numInvaders': -1000,
            'onDefense': 100,
            'invaderDistance': -12,
            'foodDistance': -4,
            'capsuleDistance': -6,
            'stop': -100,
            'reverse': -2
        }



