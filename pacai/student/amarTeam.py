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
        first = 'starAgent',
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
    Credit: https://towardsdatascience.com/search-algorithm-dijkstras-
            algorithm-uniform-cost-search-with-python-ccbee250ba9
    Credit: https://levelup.gitconnected.com/a-star-a-search-for-
            solving-a-maze-using-python-with-visualization-b0cae1c3ba92
    '''

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)
        # 3/8/22
        # self.fScore = {} 
        # self.gScore = {}
        # self.hScore = {}
        # self.parent = {}
        # 3/7/22
        self.goal = None
        self.myPos = None
        #self.nodeCost = {}


    def pathFinding(self, parent, start, goal):
        path = []
        node = goal
        while node != start:
            path.insert(0, node)
            node = parent[node]
        path.insert(0, start)
        return path


    def checkIfGhost(self, node):
        pass


    def h(self, node, goal):
        '''
        Input: Position of node and position of goal
        Output: Manhattan distance
        '''
        x1, y1 = node
        x2, y2 = goal
        return abs(x1-x2) + abs(y1-y2)


    def aStar(self, gameState, start, goal):
        '''
        Runs A* Search from start to goal to find 
        the most cost effective path
        
        Input: gameState - the current field
            start - the node to start the search
            goal - the node to reach

        Output: Path from start to goal
                path[1] to access next step

        '''

        layout = gameState.getInitialLayout()
        allNodes = layout.walls.asList(False)

        gScore = {node: float('inf') for node in allNodes}
        fScore = {node: float('inf') for node in allNodes}
        gScore[start] = 0
        fScore[start] = self.h(start, goal)

        parent = {} 
        # Might add closed list?
        # closed = {}
        queue = priorityQueue.PriorityQueue()
        queue.push(start, fScore[start])

        while not queue.isEmpty():
            node = queue.pop()

            # To prevent repeats
            # if node in closed:
                # continue

            if node == goal:
                return self.pathFinding(parent, start, goal)
            
            # Neighbor segment ---------- #
            children = []
            x, y = node
            #x = int(x)
            #y = int(y)

            up = int(y + 1)
            if not layout.isWall((x, up)):
                children.append((x, up))

            down = int(y - 1)
            if not layout.isWall((x, down)):
                children.append((x, down))

            right = int(x + 1)
            if not layout.isWall((right, y)):
                children.append((right, y))

            left = int(x - 1)
            if not layout.isWall((left, y)):
                children.append((left, y))
            # ----------------------------- #

            for child in children:
                # If child is a ghost, abandon
                tempGScore = gScore[node] + 1
                tempFScore = tempGScore + self.h(child, goal)

                if tempFScore < fScore[child]:
                    gScore[child] = tempGScore
                    fScore[child] = tempFScore
                    queue.push(child, tempFScore)
                    parent[child] = node

    # 3/8/22
    def chooseAction(self, gameState):
        '''
        Picks the next action in the optimal path returned from A* search 
        '''

        actions = gameState.getLegalActions(self.index)
        start = time.time()
        # --------- Core ---------- #
        currentState = gameState.getAgentState(self.index)
        x, y = currentState.getPosition()
        self.myPos = int(x), int(y)

        self.findGoal(gameState)
        path = self.aStar(gameState, self.myPos, self.goal)
        print(path)
        print("Goal: {0}".format(self.goal))
        print(self.myPos)
        #---------------------------#
        logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))
        
        # Mapped every action position with its action
        # Makes returning the right action much easier
        mapActions = {}
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            successorState = successor.getAgentState(self.index)
            actionPosition = successorState.getPosition()
            mapActions[actionPosition] = action

        # if self.myPos is None:
        #     return random.choice(actions)
        
        if mapActions[path[1]] not in actions:
            return random.choice(actions)
        return mapActions[path[1]]

    
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

    def findGoal(self, gameState):
        '''
        Sets a position for self.goal
        ---> means implementing
        Goals to implement in order for Offense:
            ---> Food
            ---> Capsule
            Scared Ghost
            Home
            Invader

        * ---------- Needs improvement ---------- * 
            - Look into self.getAgentStates  
            - Change enemy to ghost or scared ghost 
            - Make capsules cost less than food
        '''

        #self.myPos = gameState.getPosition(self.index)

        foodList = self.getFood(gameState).asList()
        capsuleList = self.getCapsules(gameState)
        goalList = foodList + capsuleList
        
        goalCost = {}
        for nodePos in goalList:
            # cost[node] = myDistanceCost + ghostDistanceCost
            # my distance from node
            myDistanceCost = self.getMazeDistance(self.myPos, nodePos)
            enemies = [gameState.getAgentState(enemy) for enemy in self.getOpponents(gameState)]
            # closest enemy distance to current node 
            enemyDistance = min([self.getMazeDistance(enemy.getPosition(), nodePos) for enemy in enemies])
            # getting ghostDistanceCost
            ghostDistanceCost = 0
            if enemyDistance <= 3:
                ghostDistanceCost = 100
            elif enemyDistance > 3 and enemyDistance <= 5:
                ghostDistanceCost = 50
            
            goalCost[nodePos] = myDistanceCost + ghostDistanceCost
        
        # Finding the least cost goal node
        tempCosts = min(goalCost.values())
        leastCostGoal = [key for key in goalCost if goalCost[key] == tempCosts]
        self.goal = leastCostGoal[0]

        # Above code only finds safest food/capsule


    # --------------------------- Star Agent Above -------------------------- #

    # def chooseAction(self, gameState):
    #     """
    #     Picks the next step in the optimal path returned from A* Search pathfinding
    #     """

    #     actions = gameState.getLegalActions(self.index)
    #     start = time.time()
    #     # 3/7/22 Below
    #     # My Position
    #     tempState = gameState.getAgentState(self.index)
    #     self.myPos = tempState.getPosition()
    #     #print(self.myPos)
    #     # Run functions
    #     self.findGoal(gameState)
    #     self.setCost(gameState)

    #     # Getting layout
    #     layout = gameState.getInitialLayout()
    #     # Checking every next valid move
    #     adjacent = []
    #     x, y = self.myPos
    #     print(x, y)

    #      # Up - North
    #     if not layout.isWall((int(x), int(y + 1))):
    #         adjacent.append((int(x), int(y + 1)))

    #     # Down - South
    #     if not layout.isWall((int(x), int(y - 1))):
    #         adjacent.append((int(x), int(y - 1)))

    #     # Right - East
    #     if not layout.isWall((int(x + 1), int(y))):
    #         adjacent.append((int(x + 1), int(y)))

    #     # Left - West
    #     if not layout.isWall((int(x - 1), int(y))):
    #         adjacent.append((int(x - 1), int(y)))
        
    #     # Finding next least cost action
    #     lowestAdjacentCost = 99999
    #     for pos in adjacent:
            
    #         tempCost = self.nodeCost[pos]
    #         #print(type(tempCost))
    #         if tempCost < lowestAdjacentCost:
    #             lowestAdjacentCost = tempCost
    #             #print(lowestAdjacentCost)
    #     lowestCostActions = [key for key in self.nodeCost if self.nodeCost[key] == lowestAdjacentCost]
    #     print("Lowest Cost Actions")
    #     print(lowestCostActions)
    #     lowestAction = lowestCostActions[0] # Next best action

    #     # Need to look at successorState action
    #     bestAction = None
    #     print(actions)
    #     for action in actions:

    #         successor = self.getSuccessor(gameState, action)
    #         nextState = successor.getAgentState(self.index)
    #         nextPos = nextState.getPosition()
    #         print("Future Position: ")
    #         print(nextPos)
    #         print("Lowest Action: {}".format(lowestAction))
    #         if nextPos == lowestAction:
    #             bestAction = action
    #             print("Best Action:")
    #             print(bestAction)
    #             break
    #     if bestAction is None:
    #         bestAction = random.choice(actions)
    #     # 3/7/22 Above
    #     #values = [self.evaluate(gameState, a) for a in actions]
    #     logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

    #     #maxValue = max(values)
    #     #bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    #     #return random.choice(bestActions)
    #     #return random.choice(actions) # dummy return statement
    #     return bestAction

    # def setCost(self, gameState):
    #     '''
    #     Set the cost of all nodes according to Goal and Ghost
    #     Least cost, best action:
    #         Food - 1
    #         Capsule - 0
    #         Scared Ghost - 2
    #         ---> Ghost - 100
    #         Home - 50 if ghost < 5 steps for past 5 moves
    #         Offense - 25 if invader <= 1
    #         Invader - 3 if not Pacman
    #     '''
    #     layout = gameState.getInitialLayout()
    #     allNodes = layout.walls.asList(False)
    #     for node in allNodes:
    #         distanceToGoal = self.getMazeDistance(node, self.goal)
    #         enemies = [gameState.getAgentState(enemy) for enemy in self.getOpponents(gameState)]
    #         # closest enemy distance to current node 
    #         enemyDistance = min([self.getMazeDistance(enemy.getPosition(), node) for enemy in enemies])
    #         # getting ghostDistanceCost
    #         print("Goal")
    #         print(self.goal)
    #         print("Enemy Distance")
    #         print(enemyDistance)
    #         ghostDistanceCost = 0
            
    #         # Do this if I'm pacman
    #         myState = gameState.getAgentState(self.index)
    #         if not myState.isPacman():
    #             if enemyDistance == 0: # ghost node
    #                 ghostDistanceCost = 100
    #             elif enemyDistance == 1:
    #                 ghostDistanceCost = 50
    #             elif enemyDistance == 2:
    #                 ghostDistanceCost = 40
    #             elif enemyDistance == 3:
    #                 ghostDistanceCost = 30
    #             elif enemyDistance == 4:
    #                 ghostDistanceCost = 20
    #             elif enemyDistance == 5:
    #                 ghostDistanceCost = 10

    #         self.nodeCost[node] = distanceToGoal + ghostDistanceCost
    #         print("Pos: ")
    #         print(node)
    #         print("Cost: ")
    #         print(self.nodeCost[node])

    # def searchAlgorithm(self, gameState):
    #     pass

    # # Work above


    # def h(self, gameState):
    #     pass

    #def pathfinding(self, start, end):
        #pass
        # node = end
        # path = [node]
        # while self.parent[node] is not start:
        #     node = self.parent[node]
        #     path.append((node))
        # path.append(start)
        # return path[-1]


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
        #print(actions)
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





# ------------------------------ Tried and Tested ---------------------------------- #

# -------------- Good Start -------------------------------------------------------#
#----------------------------------------------------------------------------------#
'''
def aStar(self, gameState, start, end):
    
    #Runs A* Search on all nodes to find the most cost effective path
    

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
            self.parent[(x, y + 1)] = node

        # Down - South
        if not layout.isWall((x, y - 1)):
            adjacent.append((x, y - 1))
            self.parent[(x, y - 1)] = node

        # Right - East
        if not layout.isWall((x + 1, y)):
            adjacent.append((x + 1, y))
            self.parent[(x + 1, y)] = node

        # Left - West
        if not layout.isWall((x - 1, y)):
            adjacent.append((x - 1, y))
            self.parent[(x - 1, y)] = node
'''
 #----------------------------------------------------------------------------------#       
#----------------------------------------------------------------------------------#



'''
class Graph:
    
    # Credit: https://stackabuse.com/basic-ai-concepts-a-search-algorithm/
    
    # Map of adjacencyList
    #     A, B, C are the keys to the node values. 
    #     Try coordinates as keys instead?

    # adjacencyList = {
    #     'A': [('B', 1), ('C', 3), ('D', 7)],
    #     'B': [('D', 5)],
    #     'C': [('D', 12)]
    # }
       

    def __init__(self, adjacencyList):
        self.adjacencyList = adjacencyList

    def addEdge(self, pos, edge):
        self.adjacencyList[pos] = edge

    def getEdge(self, v):
        return self.adjacencyList[v]

    def h(self, n):
        pass


    
    def aStarAlgorithm(self, startNode, stopNode):
        
        #The startNode will always be the current position of the agent
        #The stopNode will always be the goal position
        

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
'''