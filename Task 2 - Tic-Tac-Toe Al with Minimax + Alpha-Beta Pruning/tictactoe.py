"""
Task 2: Tic-Tac-Toe AI (Minimax + Alpha-Beta Pruning)
CodSoft AI Internship
Author: Doodakula Mohammad Abdulla
"""

import math
import os

# ─── Constants ────────────────────────────────────────────────────────────────
HUMAN = "X"
AI    = "O"
EMPTY = " "

# ─── Board ────────────────────────────────────────────────────────────────────

def make_board():
    return [EMPTY] * 9

def print_board(board):
    os.system("cls" if os.name == "nt" else "clear")
    print("\n  Tic-Tac-Toe  |  You: X   AI: O\n")
    print(f"  {board[0]} | {board[1]} | {board[2]}      1 | 2 | 3")
    print("  ---------      ---------")
    print(f"  {board[3]} | {board[4]} | {board[5]}      4 | 5 | 6")
    print("  ---------      ---------")
    print(f"  {board[6]} | {board[7]} | {board[8]}      7 | 8 | 9\n")

def available_moves(board):
    return [i for i, v in enumerate(board) if v == EMPTY]

def is_full(board):
    return EMPTY not in board

# ─── Win Detection ────────────────────────────────────────────────────────────

WINNING_COMBOS = [
    (0,1,2),(3,4,5),(6,7,8),   # rows
    (0,3,6),(1,4,7),(2,5,8),   # cols
    (0,4,8),(2,4,6),           # diagonals
]

def check_winner(board, player):
    return any(board[a]==board[b]==board[c]==player for a,b,c in WINNING_COMBOS)

def game_over(board):
    return check_winner(board, HUMAN) or check_winner(board, AI) or is_full(board)

def score(board):
    if check_winner(board, AI):    return  1
    if check_winner(board, HUMAN): return -1
    return 0

# ─── Minimax + Alpha-Beta Pruning ─────────────────────────────────────────────

def minimax(board, depth, is_maximizing, alpha, beta):
    if game_over(board):
        return score(board)

    if is_maximizing:
        best = -math.inf
        for move in available_moves(board):
            board[move] = AI
            val = minimax(board, depth + 1, False, alpha, beta)
            board[move] = EMPTY
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break          # Beta cut-off
        return best
    else:
        best = math.inf
        for move in available_moves(board):
            board[move] = HUMAN
            val = minimax(board, depth + 1, True, alpha, beta)
            board[move] = EMPTY
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break          # Alpha cut-off
        return best

def best_move(board):
    best_val = -math.inf
    move = None
    for m in available_moves(board):
        board[m] = AI
        val = minimax(board, 0, False, -math.inf, math.inf)
        board[m] = EMPTY
        if val > best_val:
            best_val = val
            move = m
    return move

# ─── Game Loop ────────────────────────────────────────────────────────────────

def get_human_move(board):
    while True:
        try:
            cell = int(input("Your move (1-9): "))
            idx = cell - 1          # convert 1-based → 0-based
            if 0 <= idx <= 8 and board[idx] == EMPTY:
                return idx
            print("  ⚠️  Invalid move — try again.")
        except ValueError:
            print("  ⚠️  Please enter a number between 1 and 9.")

def print_result(board):
    print_board(board)
    if check_winner(board, HUMAN):
        print("🎉 You win! (That shouldn't happen against a perfect AI...)")
    elif check_winner(board, AI):
        print("🤖 AI wins! Better luck next time.")
    else:
        print("🤝 It's a draw!")

def main():
    print("\n" + "=" * 45)
    print("  🎮  Tic-Tac-Toe AI  |  CodSoft Internship")
    print("=" * 45)
    print("  You are X.  AI is O.  Cells numbered 1-9.")

    while True:
        board = make_board()

        # Coin toss
        first = input("\nDo you want to go first? (y/n): ").strip().lower()
        human_turn = first != "n"

        while not game_over(board):
            print_board(board)
            if human_turn:
                move = get_human_move(board)
                board[move] = HUMAN
            else:
                print("AI is thinking... 🤔")
                move = best_move(board)
                board[move] = AI
                print(f"AI placed O at position {move + 1}.")
            human_turn = not human_turn

        print_result(board)

        again = input("\nPlay again? (y/n): ").strip().lower()
        if again != "y":
            print("\nThanks for playing! 👋\n")
            break

if __name__ == "__main__":
    main()
