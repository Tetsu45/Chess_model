# Chess_model
My attempt to create chess ai model to predict a first move of an opponent.I used my 200 games of chess as the data set to train the model.The average rating of my opponents is 1500 on  Lichess and In most of the games,I played D4 as white and french defense as black.
First of all,I tried to ecode Fen into tensor shape 8 x 8 x 12.After that,I put rating and number of move into the dataset.And I utilized a simple deep neural network and train on the dataset.The accuaracy is around 13 percent and the predicted move is d5.

