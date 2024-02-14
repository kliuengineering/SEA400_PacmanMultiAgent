# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions from legal moves with game state being the scope
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]

        # search for the highest score
        bestScore = max(scores)

        # consolidate the best legal moves available, transfer them to another list
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]

        # choose the best legal moves in random
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (successor_food) and Pacman position after moving (successor_pacman_pos).
        ghost_scared_time holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        "*** YOUR CODE HERE ***"
        print("\n---------- iteration begins ----------\n") #debug
        #========== Defining current game state parameters ============#

        # obtain the current state of the food dots
        current_food = currentGameState.getFood()
        current_food_list = current_food.asList()


        #========== Defining successor game state parameters ==========#

        # generate the successor
        successor_game_state = currentGameState.generatePacmanSuccessor(action)
        # print("successor_game_state\n", successor_game_state) #debug

        # pacman's position in the successor
        successor_pacman_pos = successor_game_state.getPacmanPosition()
        print("successor_pacman_pos", successor_pacman_pos) #debug

        # food position in the successor
        successor_food = successor_game_state.getFood()
        # print("\nsuccessor_food\n", successor_food) #debug

        # ghost state (object) in the successor
        successor_ghost_state = successor_game_state.getGhostStates()
        print("\nsuccessor_ghost_state\n", successor_ghost_state) #debug

        # ghost scare-time counter
        ghost_scared_time = [ghostState.scaredTimer for ghostState in successor_ghost_state]
        print("\nghost_scared_time", ghost_scared_time) #debug

        # convert the successor food data into a list
        successor_food_list = successor_food.asList()
        print("\nsuccessor_food_list -> ", successor_food_list) #debug

        # ghost positions (coordinates) in the successor
        successor_ghosts_positions = successor_game_state.getGhostPositions()
        print("\nsuccessor_ghosts_positions -> ", successor_ghosts_positions) #debug

        # determines which ghost is actually closer to pacman
        dxy_pacman_to_ghost = 2**12
        for ghost_coordinates in successor_ghosts_positions:
            dxy_pacman_to_ghost = min(manhattanDistance(ghost_coordinates, successor_pacman_pos), dxy_pacman_to_ghost)
        print("\nclosest ghost is -> ", dxy_pacman_to_ghost) #debug

        # search for the closest food
        dxy_pacman_to_food = 2**12
        for food in successor_food_list:
            dxy_pacman_to_food = min(manhattanDistance(food, successor_pacman_pos), dxy_pacman_to_food)
        print("\ncloest food is -> ", dxy_pacman_to_food) #debug


        #==================== approach #1, works but not elegant... ====================#

        # score_food = len(successor_food_list)
        # score_pacman_to_ghost = 0

        # gives a heavy reward when pacman eats food in the successor
        # if len(successor_food_list) < len(current_food_list):
        #     score_food = 2**13
        #
        # # gives a mild penalty when pacman does not eat food in the successor
        # else:
        #     score_food = -2**10

        # # gives a heavy penalty when the ghost is too close to the pacman
        # if dxy_pacman_to_ghost < 2:
        #     score_pacman_to_ghost = -2**16
        #
        # # gives neutral when ghost is reasonably far
        # else:
        #     score_pacman_to_ghost = 0
        #
        # # if ghost is scared, then pacman is fearless!
        # if ghost_scared_time[0] > 0:
        #     score_pacman_to_ghost = 0
        #
        # # consolidate the parameters into more manageable variables
        # score_pacman_to_food = 1/dxy_pacman_to_food + score_food
        # score_successor_game_state = successor_game_state.getScore()
        #
        # print("\nsuccessor game score -> ", successor_game_state.getScore()) #debug
        # print("\n---------- iteration ends ------------\n") #debug
        # return score_pacman_to_ghost + score_pacman_to_food - score_successor_game_state



        #============================== approach #2, WIP. ==============================#

        # accounts for time-outs using the game state's counter
        score_successor_game_state = successor_game_state.getScore() / 2**len(successor_food_list)

        # gives a heavy reward when pacman eats food in the successor
        if len(successor_food_list) < len(current_food_list):
            score_food = 2**12

        # gives a mild penalty when pacman does not eat food in the successor
        else:
            score_food = -2**12

        # Adjusting penalties and rewards based on field theory concepts
        score_ghost_field = (sum
        (
            [-2 ** 16 / (manhattanDistance(ghost, successor_pacman_pos) ** 2)
             for ghost in successor_ghosts_positions
             if manhattanDistance(ghost, successor_pacman_pos) > 0]
        ))

        # Direct relationship for food attraction (similar field approach but with less steep curve)
        score_food_attraction = (sum
        (
            [2 ** 10 / (manhattanDistance(food, successor_pacman_pos) ** 1)
             for food in successor_food_list if
             manhattanDistance(food, successor_pacman_pos) > 0]
        ))

        # Modify to include scared ghost condition more explicitly
        if any(scared > 0 for scared in ghost_scared_time):
            score_ghost_field = 0  # No penalty if any ghost is scared, Pac-Man is fearless

        final_score = score_ghost_field + score_food_attraction + score_food #- score_successor_game_state

        print("\nscore_ghost_field -> ", score_ghost_field)
        print("score_food_attraction -> ", score_food_attraction)
        print("score food dots -> ", score_food)
        print("successor game score -> ", score_successor_game_state) #debug
        print("final game score", final_score, "\n\n") #debug

        return final_score

        # ============================== debug mode ends ================================#
        # return successor_game_state.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState): # -> returns an action
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # constants declaration
        initial_depth = 0
        return_action = 0
        pacman = 0
        ghost = 1

        # Max <- infinity
        eval_current = float("-inf")

        # generates the initial frontier
        frontier_list = gameState.getLegalActions(pacman)
        print("\n+++++++++++++++ iteration begins +++++++++++++++\n")
        print("current depth -> ", self.depth)
        print("frontier set -> ", frontier_list, "\n")

        # loops through all vertices inside the frontier
        for frontier_state in frontier_list:
            print("---------- frontier data begins ----------\n")
            print("frontier vertex -> ", frontier_state)

            # look into the successor maps of frontiers
            successor_state = gameState.generateSuccessor(pacman, frontier_state)

            print("successor map below: \n", successor_state)

            # calls the min first
            eval_successor = self.min_layer(successor_state, initial_depth, ghost)

            print("current evaluation-> ", eval_current)
            print("successor evaluation-> ", eval_successor, "\n")

            # if the successor has a higher evaluation than the current
            if eval_successor > eval_current:

                # then return this successor's action
                return_action = frontier_state

                # now we can replace the current state with the best successor
                eval_current = eval_successor
            print("---------- frontier data ends ----------\n")

        print("\n+++++++++++++++ iteration ends +++++++++++++++++\n")
        return return_action


    # max method
    def max_layer(self, input_state, input_depth):

        # agent indices, constant
        pacman = 0
        ghost = 1

        # max determines the layer stack
        input_depth += 1

        # passes -infinity down to the leaf
        return_value = float("-inf")

        # checks win/lose/game stop conditions
        if input_state.isWin() or input_state.isLose() or input_depth == self.depth:
            return self.evaluationFunction(input_state)

        # frontier list
        frontier_list = input_state.getLegalActions(pacman)

        # generates further frontiers from each vertex, DFS
        for frontier_state in frontier_list:
            successor = input_state.generateSuccessor(pacman, frontier_state)
            return_value = max(return_value, self.min_layer(successor, input_depth, ghost))

        # returns an eval
        return return_value


    # min method
    def min_layer(self, input_state, input_depth, ghost_index):

        # agent indices, constant
        pacman = 0
        ghost = 1

        return_value = float("inf")

        # checks win/lose
        if input_state.isWin() or input_state.isLose():
            return self.evaluationFunction(input_state)

        # generates the frontier
        frontier_list = input_state.getLegalActions(ghost_index)

        # generates the successors
        for frontier_state in frontier_list:
            successor = input_state.generateSuccessor(ghost_index, frontier_state)

            # checks the corner case -> whether the ghost is the latest ghost or not
            if ghost_index == (input_state.getNumAgents() - 1):
                return_value = min(return_value, self.max_layer(successor, input_depth))

            # if not, then the ghost must be in the intermediate level
            else:
                return_value = min(return_value, self.min_layer(successor, input_depth, ghost_index + 1))

        return return_value

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
