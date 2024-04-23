
import TicTacToe

game = TicTacToe.TicTacToe()
game.current_player = game.players[0]
game.print_board()

while not game.game_over:
    move = input(f"{game.current_player}''s turn. Enter row and column (e.g. 0 0): ")
    move = tuple(map(int, move.split()))
    while move not in game.available_moves():
        move = input("Invalid move. Try again: ")
        move = tuple(map(int, move.split()))
    game.make_move(move)
    game.print_board()

if game.winner:
    print(f"{game.winner} wins!")
else:
    print("It''s a tie!")
