from isolation import Isolation, Agent, DebugState
from my_custom_player import CustomPlayer
import train


# main code
if __name__ == '__main__':
    board = DebugState()
    debug_board = board.from_state(board)
    test_agent = TEST_AGENTS['MINIMAX']¬
    custom_agent = Agent(CustomPlayer, "Custom Agent")¬
    wins, num_games = play_matches(custom_agent, test_agent, args)
    print(debug_board)
