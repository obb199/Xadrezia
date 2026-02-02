import board
from pieces_representation import pieces_number_repr, pieces_str_repr
import numpy as np

class Piece(object):
    def __init__(self, pos_x, pos_y, piece_name):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.piece_name = piece_name
        self.verify_atributes()

        self.numerical_repr = pieces_number_repr[piece_name.upper()]
        self.name_repr = pieces_str_repr[piece_name.upper()]
        self.col_to_number = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7}
        self.number_to_col = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H'}

    def verify_atributes(self):
        if isinstance(self.pos_x, int):
            if self.pos_x < 0 or self.pos_x > 7:
                raise ValueError('pos_x must be between 0 and 7')
        elif isinstance(self.pos_x, str):
            if self.pos_x not in 'ABCDEFGH' or len(self.pos_x) != 1:
                raise ValueError("pos_x must be a letter beetween 'A' and 'H'")
        else:
            raise ValueError("pos_x must be an integer beetween 0 and 7 or a letter beetween 'A' and 'H'")

        if not isinstance(self.pos_y, int):
            raise ValueError('pos_y must be between 0 and 7')

        if isinstance(self.pos_y, int):
            if self.pos_y < 0 or self.pos_y > 7:
                raise ValueError('pos_y must be between 0 and 7')

        if self.piece_name.upper() not in pieces_number_repr.keys():
            raise ValueError('Invalid piece_name.')

    def is_white_piece(self):
        return self.numerical_repr > 0 and self.numerical_repr < 7

    def is_black_piece(self):
        return self.numerical_repr >= 7

    def is_empty_square(self):
        return self.numerical_repr == 0

    def get_position(self):
        return self.number_to_col[self.pos_y] + str(8-self.pos_x), self.pos_x, self.pos_y

    def search_moves_rook_like(self, board):
        moves = set()
        for row in range(self.pos_x + 1, 8, 1):
            if board.pieces[row][self.pos_y].is_empty_square():
                moves.add('R' + self.number_to_col[self.pos_y] + str(8 - row))

            elif (board.pieces[row][self.pos_y].is_black_piece() and self.is_white_piece()) \
                    or (board.pieces[row][self.pos_y].is_white_piece() and self.is_black_piece()):
                moves.add('R' + self.number_to_col[self.pos_y] + str(8 - row))
                break

            else:
                break


        for row in range(self.pos_x - 1, -1, -1):
            if board.pieces[row][self.pos_y].is_empty_square():
                moves.add('R' + self.number_to_col[self.pos_y] + str(8 - row))
                continue

            elif (board.pieces[row][self.pos_y].is_black_piece() and self.is_white_piece()) \
                    or (board.pieces[row][self.pos_y].is_white_piece() and self.is_black_piece()):
                moves.add('R' + self.number_to_col[self.pos_y] + str(8 - row))
                break

            else:
                break

        for col in range(self.pos_y + 1, 8):
            if board.pieces[self.pos_x][col].is_empty_square():
                moves.add('R' + self.number_to_col[col] + str(8 - self.pos_x))
                continue

            elif (board.pieces[self.pos_x][col].is_black_piece() and self.is_white_piece()) \
                    or (board.pieces[self.pos_x][col].is_white_piece() and self.is_black_piece()):
                moves.add('R' + self.number_to_col[col] + str(8 - self.pos_x))
                break

            else:
                break

        for col in range(self.pos_y - 1, -1, -1):
            if board.pieces[self.pos_x][col].is_empty_square():
                moves.add('R' + self.number_to_col[col] + str(8 - self.pos_x))
                continue

            elif (board.pieces[self.pos_x][col].is_black_piece() and self.is_white_piece()) \
                    or (board.pieces[self.pos_x][col].is_white_piece() and self.is_black_piece()):
                moves.add('R' + self.number_to_col[col] + str(8 - self.pos_x))
                break

            else:
                break

        return list(sorted(moves))

    def search_moves_bishop_like(self, board):
        moves = set()
        for row_sum, col_sum, pos_x_base, pos_y_base in ([1, 1, self.pos_x + 1, self.pos_y + 1],
                                                         [1, -1, self.pos_x + 1, self.pos_y - 1],
                                                         [-1, 1, self.pos_x - 1, self.pos_y + 1],
                                                         [-1, -1, self.pos_x - 1, self.pos_y - 1]):
            row = pos_x_base
            col = pos_y_base
            while (row >= 0 and row <= 7 and col >= 0 and col <= 7):
                if board.pieces[row][col].is_empty_square():
                    moves.add('B' + self.number_to_col[col] + str(8 - row))

                elif (board.pieces[row][col].is_black_piece() and self.is_white_piece()) \
                        or (board.pieces[row][col].is_white_piece() and self.is_black_piece()):
                    moves.add('B' + self.number_to_col[col] + str(8 - row))
                    break
                else:
                    break

                row += row_sum
                col += col_sum

        return list(sorted(moves))

class EmptySquare(Piece):
    def __init__(self, pos_x, pos_y, piece_name):
        super().__init__(pos_x, pos_y, piece_name)

class Pawn(Piece):
    def __init__(self, pos_x, pos_y, color):
        piece_name = color.upper() + '_PAWN'
        super().__init__(pos_x, pos_y, piece_name)

    def valid_moves(self):
        pass

class Knight(Piece):
    def __init__(self, pos_x, pos_y, color):
        piece_name = color.upper() + '_KNIGHT'
        super().__init__(pos_x, pos_y, piece_name)

    def valid_moves(self, board):
        moves = set()
        for row, col in [2,1], [2,-1], [-2,1], [-2,-1], [1,2], [1,-2], [-1,2], [-1,-2]:
            if self.pos_x + row < 0 or self.pos_x + row > 7:
                continue
            if self.pos_y + col < 0 or self.pos_y + col > 7:
                continue

            if board.pieces[self.pos_x + row][self.pos_y + col].is_empty_square():
                moves.add('N' + self.number_to_col[self.pos_y + col] + str(8 - self.pos_x - row))
                continue

            if (board.pieces[self.pos_x + row][self.pos_y + col].is_black_piece() and self.is_white_piece())\
                    or (board.pieces[self.pos_x + row][self.pos_y + col].is_white_piece() and self.is_black_piece()):
                moves.add('N' + self.number_to_col[self.pos_y + col] + str(8 - self.pos_x - row))

        return list(sorted(moves))

class Bishop(Piece):
    def __init__(self, pos_x, pos_y, color):
        piece_name = color.upper() + '_BISHOP'
        super().__init__(pos_x, pos_y, piece_name)

    def valid_moves(self, board):
        return super().search_moves_bishop_like(board)


class Rook(Piece):
    def __init__(self, pos_x, pos_y, color):
        piece_name = color.upper() + '_ROOK'
        super().__init__(pos_x, pos_y, piece_name)

    def valid_moves(self, board):
        return super().search_moves_rook_like(board)



class Queen(Piece):
    def __init__(self, pos_x, pos_y, color):
        piece_name = color.upper() + '_QUEEN'
        super().__init__(pos_x, pos_y, piece_name)

    def valid_moves(self, board):
        moves =  super().search_moves_rook_like(board) + super().search_moves_bishop_like(board)
        for idx, move in enumerate(moves):
            moves[idx] = 'Q' + moves[idx][1:]

        return moves

class King(Piece):
    def __init__(self, pos_x, pos_y, color):
        piece_name = color.upper() + '_KING'
        super().__init__(pos_x, pos_y, piece_name)

    def valid_moves(self, board):
        moves = set()
        for row in [1,-1,0]:
            for col in [1,-1,0]:
                if row == col == 0:
                    continue

                if self.pos_x + row < 0 or self.pos_x + row > 7:
                    continue
                if self.pos_y + col < 0 or self.pos_y + col > 7:
                    continue

                if board.pieces[self.pos_x+row][self.pos_y+col].is_empty_square():
                    moves.add('K' + self.number_to_col[self.pos_x+row] + str(8 - self.pos_x - col))
                    continue

                if (board.pieces[self.pos_x+row][self.pos_y+col].is_black_piece() and self.is_white_piece()) \
                        or (board.pieces[self.pos_x+row][self.pos_y+col].is_white_piece() and self.is_black_piece()):
                    moves.add('K' + self.number_to_col[self.pos_x+row] + str(8 - self.pos_x - col))

        return list(sorted(moves))

if __name__ == '__main__':
    p = Pawn(0, 0, 'WHITE')
    print(p.numerical_repr)
    print(p.name_repr)