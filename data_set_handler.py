import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("chess_training_data.csv")
"""
# Encode FEN
df['FEN_encoded'] = df['FEN'].apply(encode_fen)

# Encode target
le = LabelEncoder()
df['Move_encoded'] = le.fit_transform(df['SAN'])

# Stack features into one array
X = np.stack(df['FEN_encoded'].values)
X = np.hstack([X, df[['PlayerRating', 'MoveNumber']].values])

y = df['Move_encoded'].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"""
# Map each piece to a channel index
piece_to_index = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def encode_fen_to_matrix(fen):
    board_matrix = np.zeros((8, 8, 12), dtype=np.int8)
    
    rows = fen.split()[0].split('/')
    
    for i, row in enumerate(rows):  # rank
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                channel = piece_to_index[char]
                board_matrix[i, col, channel] = 1
                col += 1
    return board_matrix
def make_input_tensor(fen, rating, move_number):
    board_tensor = encode_fen_to_matrix(fen).flatten()
    return np.concatenate([board_tensor, [rating, move_number]])


# Encode FEN into 8 x 8 x 12
df['FEN_encoded'] = df['FEN'].apply(encode_fen_to_matrix)
# Encode target
le = LabelEncoder()
df['Move_encoded'] = le.fit_transform(df['SAN'])
X = np.stack([
    make_input_tensor(row['FEN'], row['PlayerRating'], row['MoveNumber']) 
    for _, row in df.iterrows()
])
y = df['Move_encoded'].values

# Normalize features (optional but recommended)

scaler = StandardScaler()
X[:, -2:] = scaler.fit_transform(X[:, -2:])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





