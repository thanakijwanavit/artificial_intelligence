from sample_players import DataPlayer
from search_algorithms import *
from isolation import DebugState
import sklearn as skl
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import json
import os
import sys
from filelock import FileLock
import random
MAX_DEPTH = 4


class CustomPlayer(DataPlayer):
    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def __init__(self,player_id=0):
        super(CustomPlayer,self).__init__(player_id)
        self.history = []
        self.q = {}
        self.load_q()

    def get_reward(self,state, action):
        next_state = state.result(action)
        own_locs = next_state.locs[self.player_id]
        own_liberties = next_state.liberties(own_locs)
        opp_locs = next_state.locs[ (self.player_id +1) % 2 ]
        opp_liberties = next_state.liberties(opp_locs)
        reward = len(own_liberties) - len(opp_liberties)
        return reward

    def normalize(self, input_data):
        '''normalize the data into range 0-1'''
        model = MinMaxScaler()
        transformed_input_data = input_data.reshape(-1, 1)
        output = model.fit_transform(transformed_input_data.astype(float)).squeeze()
        return output

    def add_to_q(self,board_state_raw,action, value, initial_value = 0):
        board_state = str(board_state_raw)
        if board_state in self.q.keys():
            if action in self.q[board_state]:
                self.q[board_state][action] += value
            else:
                self.q[board_state][action] = initial_value + value
        else:
            self.q[board_state] = {}
            self.q[board_state][action] = initial_value + value

    def record_q(self, value_to_add, initial_value=0):
        #print('history to record is shape {}'.format(self.history.shape))
        if len(self.history) > 2:
            for hist in self.history:
                #print(hist)
                action = hist[0]
                board = hist[1]
                #print('board to record q is {}'.format(board))
                self.add_to_q(board,
                        action,
                        value_to_add, initial_value = 0)

    def record_result(self,win):
        '''record if the agent has won or not
        win: BOOL record if the agent has won'''
        self.load_q()
        self.load_record_action()
        adjustment_to_q = 0.01
        if win:
            #print('the winner is player {}'.format('custom player'))
            self.record_q(adjustment_to_q * 1, initial_value=0)
        else:
            #print('custom player has lost')
            self.record_q(adjustment_to_q * -1, initial_value=0)
        #sys.stdout.flush
        #sys.stdout.write(str(self.history))
        #print(self.q)
        self.clear_record_action()
        self.save_q()

    def reward(self, state, possible_actions):
        '''calculate reward for each possible state after taken each actions'''
        reward = np.array( [ self.get_reward(state, possible_action) for possible_action in possible_actions])
        return reward

    def load_record_action(self):
        if os.path.exists('./action_cache.npy'):
            self.history = np.load('./action_cache.npy')
        else:
            self.history = np.empty((0,2), str)

    def save_record_action(self):
        #print('history to save is :{}'.format(self.history))
        np.save('./action_cache.npy',self.history)

    def clear_record_action(self):
        np.save('./action_cache.npy', np.empty((0,2),str ))

    def load_dict(self,directory):
        with open(directory, 'r') as f:
            return json.load(f)

    def save_dict(self,directory, dictionary_to_save):
        with FileLock(directory):
            with open(directory, 'w') as f:
                json.dump(dictionary_to_save,f)

    def save_q(self):
        #print('q to save is {}'.format(self.q))
        q = self.q
        path = './q/q{}.json'.format(np.random.randint(10000,99999))
        self.save_dict(path,q)

    def load_q(self):
        path = './q/q.json'
        if os.path.exists(path):
            self.q = self.load_dict(path)
        else:
            self.q = {}

    def get_q_score(self, state, possible_actions, default_score = 0.1):
        '''generate q score for each item'''
        #print(self.q.keys())
        q_scores = []
        state = str(state.board)
        #print('state is {} , q_keys is {}'.format(state, self.q.keys()))
        if state in self.q.keys():
            #print('state is in q keys')
            for action in possible_actions:
                action_str = str(action)
                if action_str in self.q[state].keys():
                    q_scores.append(self.q[state][action_str])
                else :
                    q_scores.append(default_score)
        else:
            #print('state not in q keys')
            for _ in possible_actions:
                q_scores.append(default_score)
        #print('q scores is {}'.format(q_scores))
        return np.array(q_scores)


    def decision(self,state, alpha=0.2):
        self.load_record_action()
        possible_actions = state.actions()
        if len(possible_actions) > 1:
            action_scores_from_minimax = self.reward(state, possible_actions)
            normalized_score_minimax = self.normalize(action_scores_from_minimax)
            q_action_score = self.get_q_score(state, possible_actions, 0.1)
            #print(q_action_score)
            normalized_score_q = self.normalize(q_action_score)
            action_scores = alpha * normalized_score_minimax + (1-alpha) * normalized_score_q
            action_scores = action_scores - min(action_scores)
            if action_scores.sum() != 0 :
                policy = action_scores / np.sum(action_scores)
            else :
                policy = np.ones(len(action_scores))/len(action_scores)
            choice =  np.random.choice(possible_actions ,p=policy)
            #print('board status is {}, choice is {}'.format(state.board, choice))
            self.history = np.append(self.history,
                    [[str(choice),str(state.board) ]], axis = 0)
            #print('history to add is {}'.format(self.history))
            self.save_record_action()
        if len(possible_actions) <= 1:
            #print('no choice')
            choice = possible_actions[0]
        return choice



    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)
        #print(self.load_q())
        choice = self.decision(state)
        debug_state = DebugState()
        debug_board = debug_state.from_state(state)
        #sys.stdout.write( str(debug_board))
        #sys.stdout.flush()
        #print(debug_board)
        self.queue.put(choice)
        


class CustomPlayer2(DataPlayer):
    '''modification of minimax algorithm'''
    def get_action(self,state):
        '''return get action'''
        if state.ply_count < 4:
            if state in self.data:
                self.queue.put(self.data[state])
            else:
                self.queue.put(random.choice(state.actions()))
        else:
            depth_limit = 5
            for depth in range(1, depth_limit +1):
                best_move = alpha_beta_search(state, self.player_id, depth)
            self.queue.put(best_move)


class CustomPlayer3(DataPlayer):
    ''' test the latest version of minimax plus openbook'''
    def get_action(self,state):
        if state.ply_count < 4 and self.data is not None:
            if state in self.data:
                action = self.data[state]
                self.queue.put(action)
                return
            else:
                print('state not in book')
        else:
            ##iterative deepening
            depth = 1
            while MAX_DEPTH is None or depth <= MAX_DEPTH:
                action = alpha_beta_search(state, self.player_id ,depth)
                if action is None:
                    print('error no move found')
                    return
                self.queue.put(action)
                depth +=1

class RandomPlayer(DataPlayer):
    '''modification of minimax algorithm'''
    def get_action(self,state):
        '''return random action'''
        self.queue.put(random.choice(state.actions()))


class MiniMaxPlayerDemo(DataPlayer):
    '''modification of minimax algorithm'''
    def get_action(self,state):
        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            depth_limit = 4
            for depth in range(1, depth_limit +1):
                best_move = alpha_beta_search(state, self.player_id, depth)
            self.queue.put(best_move)


CustomPlayer = MiniMaxPlayerDemo
