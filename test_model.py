import pickle
import json
import pandas as pd

# Load the saved RandomForest model
# Load the model from the file
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)


# Load the team mappings
with open("team_mapping.json", "r") as f:
    team_mapping = json.load(f)

# Define the predictors used during training
predictors = ["venue_code", "opp_code", "hour", "day_code", "xg", "xga", "team_code","gf_rolling","ga_rolling","sh_rolling","sot_rolling","dist_rolling","fk_rolling","pk_rolling","pkatt_rolling"]

def get_fixture_data():
    """
    Prompts the user to input fixture data, processes it, and returns it as a DataFrame.
    """
    date = input("Enter the match date (YYYY-MM-DD): ")
    venue = input("Enter the venue (e.g., Home, Away): ")
    opp_code = input("Enter the opponent team name: ")
    team_code = input("Enter your team name: ")
    hour = int(input("Enter the hour of match (0-23): "))
    day_of_week = pd.to_datetime(date).dayofweek
    xg = float(input("Enter expected goals (xG) for your team: "))
    xga = float(input("Enter expected goals against (xGA) for your team: "))
    dist_rolling = float(input("Enter Distance Rolling for your team: "))
    fk_rolling = float(input("Enter Free Kick Rolling for your team: "))
    ga_rolling = float(input("Enter Free GA Rolling for your team: "))
    gf_rolling = float(input("Enter Free GF Rolling for your team: "))
    pk_rolling = float(input("Enter Free PK Rolling for your team: "))
    pkatt_rolling = float(input("Enter Free PKATT Rolling for your team: "))
    sh_rolling = float(input("Enter Free SH Rolling for your team: "))
    sot_rolling = float(input("Enter Free SOT Rolling for your team: "))

    # Convert venue and opponent to categorical codes
    venue_code = 1 if venue.strip().lower() == "home" else 0  # Assuming "Home" as 1, "Away" as 0
    #opp_code = team_mapping.get(opponent.strip(), -1)  # -1 if opponent not found
    #team_code = team_mapping.get(team, -1)  # -1 if team not found

    # Organize fixture data as a DataFrame for consistency
    fixture_data = pd.DataFrame({
        "venue_code": [venue_code],
        "opp_code": [opp_code],
        "hour": [hour],
        "day_code": [day_of_week],
        "xg": [xg],
        "xga": [xga],
        "team_code": [team_code],
        "gf_rolling": [gf_rolling],
        "ga_rolling": [ga_rolling],
        "sh_rolling": [sh_rolling],
        "sot_rolling": [sot_rolling],
        "dist_rolling": [dist_rolling],
        "fk_rolling": [fk_rolling],
        "pk_rolling": [pk_rolling],
        "pkatt_rolling": [pkatt_rolling],
    })

    return fixture_data

# Get the fixture data from user input
fixture_data = get_fixture_data()

# Check for missing team or opponent codes (-1)
if -1 in fixture_data[["opp_code", "team_code"]].values:
    print("Warning: Unknown team or opponent. Prediction may not be accurate.")

# Make a prediction
prediction = model.predict(fixture_data[predictors])

# Output the prediction result
result = "Win" if prediction[0] == 1 else "Loss/Draw"
print(f"The predicted result for this fixture is: {result}")
