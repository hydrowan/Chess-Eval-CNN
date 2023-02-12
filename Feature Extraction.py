"""
> Takes a chess position or generates one.
> Evaluates the position using stockfish engine.

> Extracts important information such as pawn rank and square competition by colour.
> Outputs as (8,8,n) array for convolutional neural network.


Current feature extraction covers 3 dimensions but additional dimensions can be added
"""

import chess
import chess.engine
import random
import cv2
import numpy as np
import math
# import multiprocessing as mp
import os


def evaluate_fen(fen):
    """
    Uses stockfish to evaluate position strength from fen code
    """
    board = chess.Board(fen)
    engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish.exe")
    info = engine.analyse(board, chess.engine.Limit(time=0.1,depth=20))
    return info["score"].white().cp /100.0


def generate_fen(moves:int = None, p1_skill = None, p2_skill = None,time_per_move=0.1):
    """
    Generates an 'interesting' chess position.
    Pits stockfish engine against itself.
    
    Uses different skill levels such that the eval is not always 0.0

    param moves:int             :   number of moves, else defaults random 40-70.
    param p1_skill, p2_skill    :   stockfish skill for each side. range: 0-20. Else random.

    """

    engine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish.exe")

    if not moves:
        moves = random.randint(40,70)

    board = chess.Board()

    # Set skill
    if not p1_skill or not p2_skill:
        p1_skill, p2_skill = random.randint(1,20), random.randint(1,20)

    print(f'Skills are {p1_skill} and {p2_skill}')

    # Play x moves
    print(f'playing {moves} moves')

    for i in range(moves):
        if i % 2 == 0:
            skill = p1_skill
        else:
            skill = p2_skill
        engine.configure({"Skill Level": skill})

        if not board.is_game_over():
            result = engine.play(board, chess.engine.Limit(time=time_per_move))
            move = result.move
            board.push(move)
        else:
            print(f"Game over at move {i}")
            board.pop() # Avoid trying to evaluate a checkmate position
            break
    print(f'Game finished')
    return board.fen()


def fen_to_image(fen):
    """
    Dim 1: White Pieces (brighter), Black pieces (darker) according to piece value. Nothing = 255/2
    Dim 2: White King (brighter), Black King (darker)

    King value represents attacking potential, second dimension is to ensure location is a strong bias in ML eval.
        I.e. surrounded by enemy pieces or strongly attacked squares.
    """

    # remove non-useful info from fen (supplementary after basic position)
    fen = fen.split(" ")[0]

    # Create an 8x8 numpy array @ 50% gray
    image = np.full((8, 8, 2),255/2, dtype=np.uint8)


    pieces = fen.split("/")
    pieces = [list(row) for row in pieces]

    # Relative 'importance' of pieces based on chess values
    # Upper = White
    piece_brightness = {
        "P": 255/2 + 26/2,  # Pawn
        "N": 255/2 + 77/2, # Knight
        "B": 255/2 + 83/2, # Bishop
        "R": 255/2 + 128/2, # Rook
        "Q": 255/2 + 255/2, # Queen
        "K": 255/2 + 50/2, # King
        "p": 255/2 - 26/2,  # Pawn
        "n": 255/2 - 77/2, # Knight
        "b": 255/2 - 83/2, # Bishop
        "r": 255/2 - 128/2, # Rook
        "q": 255/2 - 255/2, # Queen
        "k": 255/2 - 50/2, # King @ low level, as importance represented in another dim, power of moves just over pawn
    }

    white_king_pos = None
    black_king_pos = None

    for i, row in enumerate(pieces):
        j = 0 # current column position
        for piece in row:
            if piece.isdigit():
                # Add empty squares
                j += int(piece)
            else:
                # Add the piece
                brightness = piece_brightness.get(piece, 0)

                # Pawns close to promotion are more dangerous so raise their contrast
                if piece == "P":
                    
                    # Exponential function I created that seems to approximate well piece value.
                    # Exponentially approaches queen value as it reaches rank 8/1
                    # x = 7 (rank 8), y = 8.9
                    # y = ((0.5) * (e ** (  (x-5)/0.72  )) + 1
                    # Plus 1 gives default value of 1 for most of the board

                    # Exponential
                    rank = (7-i)
                    func = ((math.exp((rank-5)/0.72))/2) + 1

                    base = 255/2
                    change = (func/10)*255
                    brightness = base + change/2

                    # Linear function (depreciated)
                    # brightness = (255/2) + ((1-(i+1)/8)*255)/2
                
                # Same for black pawns but inversely
                if piece == "p":
                    func = ((math.exp((i-5)/0.72))/2) + 1
                    base = 255/2
                    change = (func/10)*255
                    brightness = base - change/2

                if piece in ["K", "k"]:
                    # Save the position of the king
                    if piece == "K":
                        white_king_pos = (i, j)
                    else:
                        black_king_pos = (i, j)
                image[i][j][1] = brightness
                j += 1

    # Add the position of the kings to the other dimension
    if white_king_pos is not None:
        i, j = white_king_pos
        image[i][j][0] = 255
    if black_king_pos is not None:
        i, j = black_king_pos
        image[i][j][0] = 0

    return image

def attackers(board, sqr, colour, fee):
    # change = 0
    atck = board.attackers(color = colour, square = chess.Square(sqr))
    
    # Follows is code to discount pieces that are absolutely pinned
    # As they cannot contribute to attacking a square
    # Doesn't seem to work, as this code is specifically about squares? - take another look later.

    # for piece_sqr in atck:
    #     if not board.pin(color = colour, square = piece_sqr):
    #         print("not pinned")
    #         change += fee
    #     else:
    #         print("pinned")
    # return change
    return len(atck)*fee

def board_control(fen):
    """
    Returns an 8 x 8 array representing board control by each colour.
    Iterates over every square and calculates the opposing forces attacking the square.
    Similarly to fen_to_image() black attack darkens and white lightens from 50% gray.
    """
    image = np.zeros((8,8), dtype=np.uint8)
    board = chess.Board(fen)

    for sqr in range(0,64):
        row = 7- (sqr // 8)
        col = (sqr % 8)

        sqr_pwr = attackers(board, sqr, chess.WHITE, 1)
        sqr_pwr += attackers(board, sqr, chess.BLACK, -1)

        brightness = 255/2
        brightness += sqr_pwr * (brightness/7)

        image[row][col] = brightness

    return image


def visualise(fen_image, attackers_image):
    """
    Now a simple concat function.
    Commented code facilitates dimension reduction purely for visualisation purposes.
    This will be useful when adding another dimension, but currently the full data can be viewed as RGB.
    """

    # fen_image = fen_image.sum(axis=-1).reshape(8,8,1)
    # fen_image = np.interp(fen_image, (fen_image.min(), fen_image.max()), (0,1))

    # attackers_image = attackers_image.reshape(8,8,1)
    # attackers_image = np.interp(attackers_image, (attackers_image.min(), attackers_image.max()), (0,1))

    # merged = np.concatenate((attackers_image, fen_image, np.zeros((8,8,1))), axis = -1)
    # display_image(merged)
    
    fen_image = fen_image.reshape((8,8,2))
    attackers_image = attackers_image.reshape((8,8,1))
    merged = np.concatenate((fen_image, attackers_image), axis = -1)
    return merged


def split_dimensions(array):
    """
    Splits array into it's component dimensions on axis 2 and displays them next to each other.
    Good for visualising over 3 dimensions or understanding what is going on at a basic level.
    """
    print(array.shape)
    images = np.split(array, array.shape[2], axis=2)
    canvas = np.zeros((8, 8 * len(images), 1), dtype=np.uint8)
    for i, img in enumerate(images):
        canvas[:, 8 * i: 8 * (i + 1), :] = img

    return canvas


def display_image(array):
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    cv2.imshow("image", array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_dataset(num_images, folder_name):
    """
    Generates dataset of PNGs with corresponding evaluation results
    Previously asynchronous but stockfish engine doesn't respond well / uses a lot of memory.

    Param num_images    :   How many images to generate
    Param folder_name   :   Folder to put images
    """

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for game in range (num_images):
        fen = generate_fen(time_per_move=0.005)

        try:
            eval = (evaluate_fen(fen))
        except AttributeError as e:
            print(f"Eval failed: {e}") # TODO: Solve Mate error. Board.pop() should resolve, but doesn't.
            continue
        position = fen_to_image(fen)
        attack = board_control(fen)
        concat = visualise(position, attack)

        cv2.imwrite(f'{folder_name}/game{game} {eval}.png', concat)


def main():
    while True:
        try:
            generate_dataset(2**32, "data2")
        except chess.engine.EngineTerminatedError as e:
            print(e)
            
    print("Finished.")


if __name__ == "__main__":
    main()
