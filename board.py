import pieces

class Board:
    def __init__(self):
        self.pieces = [[pieces.EmptySquare(j, i, 'EMPTY_SQUARE') for i in range(8)] for j in range(8)]

        for i in range(8):
            self.pieces[1][i] = pieces.Pawn(1, i, 'BLACK')
            self.pieces[6][i] = pieces.Pawn(6, i, 'WHITE')

        self.pieces[0][0] = pieces.Rook(0, 0, 'BLACK')
        self.pieces[0][1] = pieces.Knight(0, 1, 'BLACK')
        self.pieces[0][2] = pieces.Bishop(0, 2, 'BLACK')
        self.pieces[0][3] = pieces.Queen(0, 3, 'BLACK')
        self.pieces[0][4] = pieces.King(0, 4, 'BLACK')
        self.pieces[0][5] = pieces.Bishop(0, 5, 'BLACK')
        self.pieces[0][6] = pieces.Knight(0, 6, 'BLACK')
        self.pieces[0][7] = pieces.Rook(0, 7, 'BLACK')

        self.pieces[7][0] = pieces.Rook(7, 0, 'WHITE')
        self.pieces[7][1] = pieces.Knight(7, 1, 'WHITE')
        self.pieces[7][2] = pieces.Bishop(7, 2, 'WHITE')
        self.pieces[7][3] = pieces.Queen(7, 3, 'WHITE')
        self.pieces[7][4] = pieces.King(7, 4, 'WHITE')
        self.pieces[7][5] = pieces.Bishop(7, 5, 'WHITE')
        self.pieces[7][6] = pieces.Knight(7, 6, 'WHITE')
        self.pieces[7][7] = pieces.Rook(7, 7, 'WHITE')

        self.white_king_pos = [7, 4]
        self.black_king_pos = [0, 4]

        self.en_passant_paws = []

    def show_board(self):
        for i in range(8):
            for j in range(8):
                print(self.pieces[i][j].name_repr, end=' ')
            print(8-i)

        for l in 'ABCDEFGH':
            print(l, end='  ')

    def get_valid_moves(self, color_moves='WHITE'):
        if color_moves.upper() == 'WHITE':
            letter_repr = 'W'
        elif color_moves.upper() == 'BLACK':
            letter_repr = 'B'
        else:
            raise NameError('color_moves must be "WHITE" or "BLACK"')

        valid_moves = []
        for i in range(8):
            for j in range(8):
                if self.pieces[i][j].name_repr[0] == letter_repr and self.pieces[i][j].name_repr[1] != 'P':
                    valid_moves += board.pieces[i][j].valid_moves(self)
        return valid_moves

if __name__ == '__main__':
    board = Board()
    #board.pieces[0][4] = pieces.EmptySquare(0, 4, 'EMPTY_SQUARE')
    #board.pieces[3][3] = pieces.Rook(3, 3, 'BLACK')
    board.show_board()
    #print(board.pieces[3][3].name_repr)
    #print(board.pieces[3][3].valid_moves(board))
    #print(board.pieces[3][3].valid_moves(board))

    print(board.get_valid_moves('BLACK'))
