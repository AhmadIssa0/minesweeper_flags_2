

import tkinter as tk
from functools import partial
from gamestate import GameState
from PIL import ImageTk, Image

class MinesweeperGUI(tk.Tk):

    def __init__(self, board_size, policy, window_size=500):
        super(MinesweeperGUI, self).__init__()
        self._board_size = board_size
        self.geometry(f"{window_size}x{window_size}")
        self.title('Minesweeper Flags')
        frame = tk.Frame(self)
        frame.pack(expand=True)
        self._pixel = tk.PhotoImage(width=1, height=1)  # hacky way to force width/height of buttons to be in pixels.
        self._blue_flag_img = ImageTk.PhotoImage(Image.open('flag-blue.png').resize((20, 20)))
        self._red_flag_img = ImageTk.PhotoImage(Image.open('flag-red.png').resize((20, 20)))

        buttons = []
        for r in range(board_size):
            buttons.append([])
            for c in range(board_size):
                button = tk.Button(
                    frame,
                    text='',
                    image=self._pixel,
                    compound='c',
                    command=partial(self.button_pressed, r=r, c=c),
                    borderwidth=1
                )
                button.grid(column=c, row=r)
                button.config(width=window_size // (board_size + 2), height=window_size // (board_size + 2))
                buttons[-1].append(button)
        self.buttons = buttons

        self._gamestate = GameState.create_new_game(board_size)
        self._policy = policy
        self.update_gui()

    def update_gui(self):
        gs = self._gamestate
        if gs.is_game_over():
            self.title(f"Minesweeper Flags -- {gs.winner()} won.")
        else:
            turn = 'Blue' if gs.is_blues_turn else 'Red'
            self.title(f"Minesweeper Flags -- {turn} to move. Blue: {gs.num_blue_flags()}, Red: {gs.num_red_flags()}")
        for row in range(self._board_size):
            for col in range(self._board_size):
                if not gs.visible[row, col]:
                    self.buttons[row][col]['text'] = ''
                elif gs.board[row, col] == GameState.FLAG:
                    if (row, col) in gs.blue_flags_captured:
                        flag = self._blue_flag_img
                    else:
                        flag = self._red_flag_img
                    self.buttons[row][col].configure(text='', image=flag)
                elif gs.board[row, col] == GameState.EMPTY:
                    self.buttons[row][col].configure(text='', bg='green')
                else:
                    self.buttons[row][col]['text'] = str(gs.board[row, col].item())

    def button_pressed(self, r, c):
        print('Button pressed: ', r, c)
        if self._gamestate.is_move_valid(r, c):
            self._gamestate = self._gamestate.make_move(r, c)
            self.update_gui()
        else:
            print('Move invalid.')


gui = MinesweeperGUI(board_size=10, policy=None)
gui.mainloop()
