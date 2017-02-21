"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def aggressive_improved_score(game, player, weight=2):
    """
        As discussed in lecture by wieghting the number of opponents legal moves more heavily
        the agent is forced to consider more aggressive play. To maximise the score more emphasis is 
        placed on reducing the opponents moves compared to maximising the players own.
        
        In practice this heuristic does not seem to provide any additional benefit over the improved score heuristic.
    """
    return (len(game.get_legal_moves(player)) - weight*len(game.get_legal_moves(game.get_opponent(player))))

def defensive_improved_score(game, player, weight=2):
    """
    Out of curiosity the defensive heuristic weights maximizing a players own moves compared to 
    minimising the opponents.
    
    Again this doesn't seem to provide any benefit.  This could be because in this version of the game
    the player can only move in the an L-shape so on a small board the number of moves a player can make 
    quickly becomes small compared to a game where the player can move like a queen in chess.
    """
    return (weight*len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player))))

def number_neighbouring_blanks_score(game, player):
    """
        This heuristic implements the improved score from lecture. This score is then modified by subtracting 
        the number of occupied surrounding spaces that block future moves. The aim is to try to prevent the agent
        getting cornered.
        
        This performs worse than the other heuristics.  In retrospect later in the game this will remove deprioritise
        otherwise good moves on a crowded board.
    """
    score = (len(game.get_legal_moves(player)) - len(game.get_legal_moves(game.get_opponent(player))))
    loc = game.get_player_location(player)
    #print("[+] current location : " + str(loc))
    blanks = game.get_blank_spaces()
    for i in range(-2,3):
        for j in range(-2,3):
            neighbour = (loc[0]-i, loc[1]-j)
            if neighbour == loc or i == j:
                continue
            #print(neighbour)
            if neighbour not in blanks:
               score -= 1
    return score



def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    #raise NotImplementedError
    
    #score = aggressive_improved_score(game, player)
    #score = defensive_improved_score(game, player)
    score = number_neighbouring_blanks_score(game, player)
    
    #for move in opp_moves:
    #   if 0 in move or game.width in move or game.height in move:
    #       score += 1
    return float(score)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if len(legal_moves) == 0:
            return
        #game.apply_move((2, 3))
        #game.apply_move((0, 5))
        search_func = self.minimax
        if search_func == "alphabeta":
            search_func = self.alphabeta
        
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            #pass
            #score, move = self.minimax(game,1, False)
            #print(score, move)
            
            # implentation of iterative deepening
            depth = 1
            if self.iterative:
                best_score = float("-inf")
                best_move = None
                while time_left()>140:
                    score, move = search_func(game, depth)
                    if move == (-1, -1):
                        return
                    if score > best_score:
                        best_move = move
                        best_score = score
                    depth +=1
            else:
                score, best_move = search_func(game, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        #raise NotImplementedError
        #assert best_move in legal_moves
        return best_move
        
    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        if maximizing_player:
            return self.max_value(game, depth)
        else:
            return self.min_value(game, depth)

    def max_value(self, game, depth):
        # get the legal moves for the current state
        legal_moves = game.get_legal_moves()
        
        # if a leaf node has been reached or there are no more legal moves
        # then return the score at this node for the active player and (-1, -1)
        # as the move choice
        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, game.active_player), (-1, -1)
        best_score = float("-inf")
        best_move = None
        for move in legal_moves:
            score = max(best_score, self.min_value(game.forecast_move(move), depth-1)[0])
            if score > best_score:
                best_score = score
                best_move = move
        return best_score, best_move
    
    def min_value(self, game, depth):
        # get the legal moves for the current state
        legal_moves = game.get_legal_moves()
        
        # if a leaf node has been reached or there are no more legal moves
        # then return the score at this node for the opponent as this is a minimizing
        # level and need the score with respect to the minizing player and (-1, -1)
        # as the move choice
        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, game.get_opponent(game.active_player)), (-1, -1)
        min_score = float("inf")
        min_move = None
        for move in legal_moves:
            score = min(min_score, self.max_value(game.forecast_move(move), depth-1)[0])
            if score < min_score:
                min_score = score
                min_move = move
        return min_score, min_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # TODO: finish this function!
        if maximizing_player:
            return self.max_value_alpha_beta(game, depth, alpha, beta)
        else:
            return self.min_value_alpha_beta(game, depth, alpha, beta)

    def max_value_alpha_beta(self, game, depth, alpha, beta):
        # get the legal moves for the current state
        legal_moves = game.get_legal_moves()
        
        # if a leaf node has been reached or there are no more legal moves
        # then return the score at this node for the active player and (-1, -1)
        # as the move choice
        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, game.active_player), (-1, -1)
        best_score = float("-inf")
        best_move = None

        for move in legal_moves:
            score = max(best_score, self.min_value_alpha_beta(game.forecast_move(move), depth-1, alpha, beta)[0])
            if score > best_score:
                best_score = score
                best_move = move
            if best_score >= beta:
                return best_score, best_move
            alpha = max(best_score, alpha)
        return best_score, best_move

    def min_value_alpha_beta(self, game, depth, alpha, beta):
        # get the legal moves for the current state
        legal_moves = game.get_legal_moves()
        
        # if a leaf node has been reached or there are no more legal moves
        # then return the score at this node for the active player and (-1, -1)
        # as the move choice
        if depth == 0 or len(legal_moves) == 0:
            return self.score(game, game.get_opponent(game.active_player)), (-1, -1)
        min_score = float("inf")
        min_move = None

        for move in legal_moves:
            score = min(min_score, self.max_value_alpha_beta(game.forecast_move(move), depth-1, alpha, beta)[0])
            if score < min_score:
                min_score = score
                min_move = move
            if min_score <= alpha:
                return min_score, min_move
            beta = min(min_score, beta)
        return min_score, min_move
