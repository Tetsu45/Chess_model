import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("chess_training_data.csv")

# Check most common moves
move_counts = df['SAN'].value_counts()

# Show top 10 moves
print("Top 10 moves:\n", move_counts.head(10))

# Plot move frequency distribution
move_counts.head(20).plot(kind='bar', title='Top 20 Most Frequent Moves')
plt.xlabel("Move")
plt.ylabel("Count")
plt.show()
# Plot histogram of player ratings
plt.figure(figsize=(10, 5))
sns.histplot(df['PlayerRating'], bins=20, kde=True)
plt.title("Distribution of Player Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Moves")
plt.show()
