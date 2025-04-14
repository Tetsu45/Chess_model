import chess.pgn
import pandas as pd
import os

def extract_data_from_pgn(pgn_path, max_games=None):
    data = []
    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        game_counter = 0
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break  # End of file
            game_counter += 1
            if max_games and game_counter > max_games:
                break

            # Extract metadata
            white_rating = game.headers.get("WhiteElo")
            black_rating = game.headers.get("BlackElo")
            result = game.headers.get("Result")
            if not white_rating or not black_rating:
                continue  # Skip games without ratings

            board = game.board()
            move_number = 1
            for move in game.mainline_moves():
                fen = board.fen()
                san = board.san(move)
                uci = move.uci()
                player_to_move = "White" if board.turn else "Black"
                player_rating = white_rating if board.turn else black_rating

                data.append({
                    "MoveNumber": move_number,
                    "Player": player_to_move,
                    "PlayerRating": int(player_rating),
                    "FEN": fen,
                    "SAN": san,
                    "UCI": uci,
                    "Result": result
                })

                board.push(move)
                move_number += 1

    return pd.DataFrame(data)

# Example usage
if __name__ == "__main__":
    pgn_path = "my_lichess_games.pgn"  # Replace with your actual PGN file path
    df = extract_data_from_pgn(pgn_path, max_games=500)  # Limit to 500 games (optional)
    df.to_csv("chess_training_data.csv", index=False)
    print("Extracted data saved to 'chess_training_data.csv'")
