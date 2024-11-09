import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# URL of the Premier League stats
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"

# Fetching the page content
data = requests.get(standings_url)

# Parsing the page content with BeautifulSoup
soup = BeautifulSoup(data.text)
standings_table = soup.select('table.stats_table')[0]
links = standings_table.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if '/squads/' in l]

# Extracting match data using Pandas
team_urls = [f"https://fbref.com{l}" for l in links]
data = requests.get(team_urls[0])
matches = pd.read_html(data.text, match="Scores & Fixtures")[0]

# Extract match shooting stats
soup = BeautifulSoup(data.text)
links = soup.find_all('a')
links = [l.get("href") for l in links]
links = [l for l in links if l and 'all_comps/shooting/' in l]

data = requests.get(f"https://fbref.com{links[0]}")
shooting = pd.read_html(data.text, match="Shooting")[0]
print(shooting.head())

# Cleaning and merging the scraped data
shooting.columns = shooting.columns.droplevel()
team_data = matches.merge(shooting[["Date", "Sh", "SoT","SoT%" ,"Dist", "FK", "PK", "PKatt"]], on="Date")
team_data.head()

# List of seasons to scrape data for (2020 - 2024)
years = list(range(2024, 2020, -1))
all_matches = []

# Initial standings URL
standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"

# Loop through each seasons
for year in years:

    # Fetching the standings page content for the current season
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text)
    standings_table = soup.select('table.stats_table')[0]

    # Extracting and filtering team squad URLs
    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]

    # Finding the URL for the previous season
    previous_season = soup.select("a.prev")[0].get("href")
    standings_url = f"https://fbref.com{previous_season}"

    # Looping through each team URL
    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")

        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]

        soup = BeautifulSoup(data.text)
        time.sleep(5)
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()

        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue

        team_data = team_data[team_data["Comp"] == "Premier League"]
        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)

        time.sleep(3)

match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]
match_df

# Writing the data to a CSV file
match_df.to_csv("matches.csv")