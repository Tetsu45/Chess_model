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
from tensorflow.keras.callbacks import ReduceLROnPlateau
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
def build_improved_model(num_classes):
    # Board input and processing
    board_input = Input(shape=(8, 8, 12))
    
    # CNN for board feature extraction
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(board_input)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Flatten()(x)
    
    # Metadata input
    meta_input = Input(shape=(2,))  # Rating and move number
    
    # Combine board features with metadata
    combined = concatenate([x, meta_input])
    
    # Dense layers for decision making
    x = Dense(512, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[board_input, meta_input], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model
# Encode FEN into 8 x 8 x 12
df['FEN_encoded'] = df['FEN'].apply(encode_fen_to_matrix)
# Encode target
le = LabelEncoder()
df['Move_encoded'] = le.fit_transform(df['SAN'])
# Prepare board tensors separately from metadata
X_board = np.stack([encode_fen_to_matrix(row['FEN']) for _, row in df.iterrows()])
X_meta = np.stack([[row['PlayerRating'], row['MoveNumber']] for _, row in df.iterrows()])
y = df['Move_encoded'].values

scaler = StandardScaler()
X_meta = scaler.fit_transform(X_meta)  # Scale metadata

# Split data
X_board_train, X_board_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
    X_board, X_meta, y, test_size=0.2, random_state=42
)
num_classes = len(le.classes_)


model = build_improved_model(num_classes)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

history = model.fit(
    [X_board_train, X_meta_train], y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=64,
    callbacks=[reduce_lr]
)
loss, acc = model.evaluate([X_board_test, X_meta_test], y_test)
print(f"Test Accuracy: {acc:.2%}")
# For prediction
"""sample_index = 0
pred = model.predict([
    np.expand_dims(X_board_test[sample_index], axis=0),
    np.expand_dims(X_meta_test[sample_index], axis=0)
])
pred_move = le.inverse_transform([np.argmax(pred)])
print("Predicted move:", pred_move[0])"""
num_samples = 5

for sample_index in range(num_samples):
    pred = model.predict([
    np.expand_dims(X_board_test[sample_index], axis=0),
    np.expand_dims(X_meta_test[sample_index], axis=0)
])
    pred_move = le.inverse_transform([np.argmax(pred)])

    # Print the predicted move and the corresponding input (FEN, rating, move number)
    print(f"Sample {sample_index + 1}:")
    print(f"Predicted move: {pred_move[0]}")
    print(f"FEN: {df.iloc[sample_index]['FEN']}")
    print(f"Rating: {df.iloc[sample_index]['PlayerRating']}")
    print(f"Move number: {df.iloc[sample_index]['MoveNumber']}")
    print(f"Actual move: {df.iloc[sample_index]['SAN']}")
    print("-" * 50)
