import requests

def download_lichess_games(username, max_games=1000, output_file="my_lichess_games.pgn", token=None):
    url = f"https://lichess.org/api/games/user/{username}"

    params = {
        "max": max_games,
        "opening": True,
        "perfType": "blitz",
        "analysed": False,
        "pgnInJson": False,
    }

    headers = {
        "Accept": "application/x-chess-pgn"
    }

    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✅ Successfully saved {max_games} games to '{output_file}'")
    else:
        print(f"❌ Failed to download games. Status code: {response.status_code}")
        print(response.text)

# Example usage
if __name__ == "__main__":
    # Replace with your new, regenerated token
    my_token = "lip_aqpLivjV6Nryi8xMU8wu"
    download_lichess_games("JACK_9087", max_games=1000, token=my_token)