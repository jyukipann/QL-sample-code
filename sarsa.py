import numpy as np
import pickle
import pathlib
import sys

def epsilon_greedy_selection(epsilon:float, values:list):
    # return 3
    if epsilon > np.random.uniform():
        return np.random.randint(0,len(values))
    else:
        # print(values)
        index = np.argmax(values)
        # print(index)
        # exit(0)
        return index
        
def show_board(board,state,Q,actions,reward_hole,start_state,goal_state):
    h,w = board.shape
    actions_icon = ["↓","↑","←","→"]
    board_icons = np.zeros((h,w),str)
    for y in range(h):
        for x in range(w):
            action_index = np.argmax(Q[y,x])
            board_icons[y,x] = actions_icon[action_index]
    board_icons[start_state[0],start_state[1]] = 'S'
    board_icons[goal_state[0],goal_state[1]] = "G"
    board_icons[board == reward_hole] = "×"
    # board_icons[state[0],state[1]] = "□"
    board_icons = "\n".join([" ".join(row) for row in board_icons])
    print(board_icons)

def main():
    alpha = 0.2
    epsilon = 0.3
    gamma = 0.9
    max_step = 200
    max_episode = 10000
    actions = np.array(
        [[1,0],[-1,0],[0,-1],[0,1]], # y x
        np.int8
    )
    board_visual = np.array(
        [
            ["S","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",],
            ["R","R","R","R","R","H","H","R","R","R","R","R","R","R","R","H","H","H","R","R",],
            ["R","R","R","H","R","H","H","R","R","R","R","R","R","R","R","H","H","H","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","H","R","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","H","H","R","R","R","H","R","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","H","H","R","R","R","R","R","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","H","R","R","R","R","R","R","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","H","R","R","R","R","R","R","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","H","R","R","R","R","H","R","R","R","R",],
            ["H","H","H","H","H","H","R","R","R","R","H","R","R","R","H","H","H","R","R","H",],
            ["R","R","R","R","R","R","R","R","R","R","H","H","H","H","H","H","R","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","R","H","H","H","H","H","R","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",],
            ["R","R","R","H","H","H","H","H","H","H","H","R","R","R","R","R","H","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","H","R","R","R","R","R","H","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","H","H","H","R","R","H","H","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R","R",],
            ["R","R","R","R","R","R","R","R","R","H","R","R","R","R","R","R","R","R","R","G",],
        ]
    )
    
    board_shape = board_visual.shape
    # print(board_shape)
    # exit(0)
    board_shape = np.array(board_shape,np.int8)
    reward_hole = -100
    reward_nomal = -1
    reward_goal = 100
    board = np.ones(board_shape,np.int0)*reward_nomal
    board[board_visual == "S"] = -100
    board[board_visual == "R"] = reward_nomal
    board[board_visual == "G"] = reward_goal
    board[board_visual == "H"] = reward_hole
    start_index = np.array(np.where(board_visual == "S")).ravel()
    goal_index = np.array(np.where(board_visual == "G")).ravel()
    Q_path = "sarsa_Q_val.pickle"
    Q_path = pathlib.Path(__file__).parent / Q_path
    if Q_path.exists():
        with open(Q_path, 'rb') as f:
            Q = pickle.load(f)
    else:
        Q = np.random.rand(board_shape[0],board_shape[1],4)
    s = start_index
    start_state = s.copy()
    goal_state = goal_index.copy()
    _board_shape = board_shape-1
    action_count = [0,0,0,0]
    goal_count = 0
    hole_count = 0
    print(f"\r{0}/{max_episode}",end="")
    print()
    show_board(board,s,Q,actions,reward_hole=reward_hole,start_state=start_state,goal_state=goal_state)
    print()
    for episode in range(max_episode):
        if (episode+1) % 100 == 0:
            print(f"\r{episode+1}/{max_episode}, goal:{goal_count}, hole:{hole_count}",end="")
        if (episode+1) % 1500 == 0:
            print()
            show_board(board,s,Q,actions,reward_hole=reward_hole,start_state=start_state,goal_state=goal_state)
            print()
            with open(Q_path, 'wb') as f:
                pickle.dump(Q, f)
        s = start_state
        action = epsilon_greedy_selection(epsilon,Q[s[0],s[1]])
        for step in range(max_step):
            action_count[action]+=1
            # print(action)
            s_prime = s + actions[action]
            s_prime = np.clip(s_prime,(0,0),_board_shape)
            r = board[s_prime[0],s_prime[1]]
            if r == reward_hole:
                s_prime = start_state
            action_prime = epsilon_greedy_selection(epsilon,Q[s_prime[0],s_prime[1]])
            # print(s)
            Q[s[0],s[1],action] += alpha*(r + gamma*Q[s_prime[0],s_prime[1],action_prime] - Q[s[0],s[1],action])
            if (s_prime == start_state).all():
                hole_count+=1
                # break

            if (s_prime == goal_state).all():
                goal_count += 1
                break
            # print(s,action)
            s = s_prime
            action = action_prime
    print()
    show_board(board,s,Q,actions,reward_hole=reward_hole,start_state=start_state,goal_state=goal_state)
    print()
    print(action_count,goal_count)
    with open(Q_path, 'wb') as f:
        pickle.dump(Q, f)
        


if __name__ == "__main__":
    main()