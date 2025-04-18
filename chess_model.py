import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, Flatten, Reshape, Input, concatenate
from tensorflow.keras.models import Model
df = pd.read_csv("chess_training_data.csv")
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

num_classes = len(le.classes_)

model = Sequential([
    Dense(256, activation='relu', input_shape=(770,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(le.classes_), activation='softmax')  # Number of unique SAN moves
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=32
)


loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2%}")

# Predict on a sample
#sample_index = 0
#pred = model.predict(np.expand_dims(X_test[sample_index], axis=0))
#pred_move = le.inverse_transform([np.argmax(pred)])
 
#print("Predicted move:", pred_move[0])
# Number of moves to print
num_samples = 5

for sample_index in range(num_samples):
    pred = model.predict(np.expand_dims(X_test[sample_index], axis=0))
    pred_move = le.inverse_transform([np.argmax(pred)])

    # Print the predicted move and the corresponding input (FEN, rating, move number)
    print(f"Sample {sample_index + 1}:")
    print(f"Predicted move: {pred_move[0]}")
    print(f"FEN: {df.iloc[sample_index]['FEN']}")
    print(f"Rating: {df.iloc[sample_index]['PlayerRating']}")
    print(f"Move number: {df.iloc[sample_index]['MoveNumber']}")
    print(f"Actual move: {df.iloc[sample_index]['SAN']}")
    print("-" * 50)

