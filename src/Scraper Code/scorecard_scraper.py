import pandas as pd

from time import sleep
import requests
from bs4 import BeautifulSoup
import re
import numpy as np


leagueId="8048"

eventId=""

dat_url = pd.read_csv("./matchdetails_scraper/matchdetails_scraper/2009_matchedetails.csv")
eventId_gp= [str(url).split("/")[6] for url in dat_url["url"]]
len_url= len(eventId_gp)






def extract_batting_data(series_id, match_id):

    URL = 'https://www.espncricinfo.com/series/'+ str(series_id) + '/scorecard/' + str(match_id)
    page = requests.get(URL)
    bs = BeautifulSoup(page.content, 'lxml')

    table_body=bs.find_all('tbody')
    batsmen_df = pd.DataFrame(columns=["Name","Desc","Runs_Scored", "Balls_Faced", "4s_hit", "6s_hit", "SR", "Team"])
    for i, table in enumerate(table_body[0:4:2]):
        rows = table.find_all('tr')
        for row in rows[::2]:
            cols=row.find_all('td')
            cols=[x.text.strip() for x in cols]
            if cols[0] == 'Extras':
                continue
            if len(cols) > 7:
                batsmen_df = batsmen_df.append(pd.Series(
                [re.sub(r"\W+", ' ', cols[0].split("(c)")[0]).strip(), cols[1], 
                cols[2], cols[3], cols[5], cols[6], cols[7], i+1], 
                index=batsmen_df.columns ), ignore_index=True)
            elif len(cols)<7 and len(cols)>2:
                batsmen_df = batsmen_df.append(pd.Series(
                [re.sub(r"\W+", ' ', cols[0].split("(c)")[0]).strip(), cols[1], 0, 0, 0, 0, 0, i+1], index = batsmen_df.columns), ignore_index=True)
                
    for i in range(2):
        if len( bs.find_all("tfoot"))>1 :
                    dnb_row = bs.find_all("tfoot")[i].find_all("div")
                    for c in dnb_row:
                        dnb_cols = c.find_all('span')
                        dnb = [x.text.strip().split("(c)")[0] for x in dnb_cols]
                        dnb = filter(lambda item: item, [re.sub(r"\W+", ' ', x).strip() for x in dnb])
                        for dnb_batsman in dnb:
                            batsmen_df = batsmen_df.append(pd.Series([dnb_batsman, "DNB", 0, 0, 0, 0, 0, i+1], index = batsmen_df.columns), ignore_index =True)

    return batsmen_df
    
def extract_bowling_data(series_id, match_id):

    URL = 'https://www.espncricinfo.com/series/'+ str(series_id) + '/scorecard/' + str(match_id)
    page = requests.get(URL)
    bs = BeautifulSoup(page.content, 'lxml')

    table_body=bs.find_all('tbody')
    bowler_df = pd.DataFrame(columns=['Name', 'Overs_Bowled', 'Maidens', 'Runs_Given', 'Wickets',
                                      'Econ', 'Dots', '4s_Given', '6s_Given', 'Wd', 'Nb','Team'])
    for i, table in enumerate(table_body[1:4:2]):
        rows = table.find_all('tr')
        for row in rows:
            cols=row.find_all('td')
            cols=[x.text.strip() for x in cols]
            if len(cols)>10:
                bowler_df = bowler_df.append(pd.Series([cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], 
                                                    cols[6], cols[7], cols[8], cols[9], cols[10], (i==0)+1], 
                                                   index=bowler_df.columns ), ignore_index=True)
    return bowler_df

def calculate_batting_points(batsmen_df):

    int_cols = ["Runs", "Balls", "4s", "6s", "Team"]
    for col in int_cols:
        batsmen_df[col] = batsmen_df[col].astype(int)
        
    batsmen_df["base_points"] = batsmen_df["Runs"]
    batsmen_df["pace_points"] = batsmen_df["Runs"] - batsmen_df["Balls"]
    batsmen_df["milestone_points"] = (np.floor(batsmen_df["Runs"]/25)).replace(
                                      {1.0:5, 2.0:15, 3.0:30, 4.0:50, 5.0:50, 6.0:50, 7.0:50, 8.0:50})
    batsmen_df["impact_points"] = batsmen_df["4s"] + 2 * batsmen_df["6s"] + \
                                  (batsmen_df["Runs"] == 0) * (batsmen_df["Desc"] != "not out") * \
                                  (batsmen_df["Desc"] != "DNB") * (batsmen_df["Desc"] != "absent hurt") * (-5) 
    batsmen_df["batting_points"] = batsmen_df["base_points"] + batsmen_df["pace_points"] + \
                                    batsmen_df["milestone_points"] + batsmen_df["impact_points"]    
    
    return batsmen_df
    
    
def calculate_bowling_points(bowler_df):
 
    int_cols = ["Wickets","Runs", "Dots", "Maidens", "Team"]
    for col in int_cols:
        bowler_df[col] = bowler_df[col].astype(int) 
    bowler_df["Balls"] = bowler_df["Overs"].apply(lambda x: x.split(".")).\
                        apply(lambda x: int(x[0])*6 + int(x[1]) if len(x)>1 else int(x[0])*6)

    bowler_df["base_points"] = 20*bowler_df["Wickets"]
    bowler_df["pace_points"] = 1.5*bowler_df["Balls"] - bowler_df["Runs"]
    bowler_df["pace_points"] = bowler_df["pace_points"] + (bowler_df.loc[:,"pace_points"]>0) * bowler_df["pace_points"]
    bowler_df["milestone_points"] = bowler_df["Wickets"].replace({1:0, 2:5, 3:15, 4:30, 5:50, 6:50, 7:50, 8:50})
    bowler_df["impact_points"] = bowler_df["Dots"] + bowler_df["Maidens"]*25
    bowler_df["bowling_points"] = bowler_df["base_points"] + bowler_df["pace_points"] + \
                                    bowler_df["milestone_points"] + bowler_df["impact_points"]
                                    
    return bowler_df
def calculate_fielding_points(batsmen_df):
    
    fielder_df = batsmen_df[["Name","Team"]]
    fielder_df.loc[:,"fielding_points"] = 0
    for team in [1,2]:
        fielders = []
        for wicket in batsmen_df[batsmen_df["Team"] == team]["Desc"]:
            if wicket.find("c & b") == 0:
                fielders.append(wicket.split("c & b")[1].strip())
            elif wicket.find("c") == 0:
                fielders.append(wicket.split("c ")[1].split("b ")[0].strip())
            if wicket.find("st") == 0:
                fielders.append(wicket.split("st ")[1].split("b ")[0].strip())
            if wicket.find("run out") == 0:
                fielders.extend([x.strip() for x in wicket.split("run out")[1].replace('(', '').replace(')', '').split("/")])
            if wicket.find("sub (") != -1:
                del fielders[-1]

        fielders = [re.sub(r"\W+", ' ', fielder).strip() for fielder in fielders]          
        fielding_team = [1 if team==2 else 2]
        for fielder in fielders:
            s = fielder_df.loc[fielder_df["Team"]==fielding_team[0]]["Name"].str.contains(fielder)
            index_val = s[s].index.values[0]
            fielder_df.loc[index_val,"fielding_points"] += 10
    
    return fielder_df

#def calculate_bonus_points(batsmen_df):

 #   batsmen_df.loc[:,"bonus_points"] = 0
  #  if len(bs.find_all("div",{"class":"best-player-name"})) > 0:
   #     man_of_match = bs.find_all("div",{"class":"best-player-name"})[0].find('a').text.strip()
    #    s = fielder_df["Name"].str.contains(man_of_match)
     #   if len(s[s].index.values) > 0: ## Entire name matches
      #      index_val = s[s].index.values[0]
       # elif fielder_df["Name"].str.contains(man_of_match.split()[1]).sum() == 1: ## Second name matches with exactly one player
        #    s = (fielder_df["Name"].str.contains(man_of_match.split()[1]))
         #   index_val = s[s].index.values[0]
        #else: ## Check for second name match and match of first letter of initial with first letter of name
         #   s = (fielder_df["Name"].str.contains(man_of_match.split()[1])) & (fielder_df["Name"].str[0] == man_of_match.split()[0][0])
          #  index_val = s[s].index.values[0]
        #batsmen_df.loc[index_val,"bonus_points"] += 25

    #return batsmen_df

def get_scorecard(series_id, match_id):

    batsmen_df = extract_batting_data(series_id, match_id)
    bowler_df = extract_bowling_data(series_id, match_id)
    #batsmen_df = calculate_batting_points(batsmen_df)
    #bowler_df = calculate_bowling_points(bowler_df)
    #fielder_df = calculate_fielding_points(batsmen_df)
    #batsmen_df = calculate_bonus_points(batsmen_df)
    total_df = pd.merge(batsmen_df,bowler_df,how="outer")
    #total_df["total_points"] = total_df["batting_points"] + total_df["bowling_points"] + \
    #total_df["fielding_points"] + total_df["bonus_points"]

    return total_df

df=pd.DataFrame(columns=["Name","Desc","Runs_Scored", "Balls_Faced", "4s_hit", "6s_hit", "SR", "Team",'Overs_Bowled', 'Maidens', 'Runs_Given', 'Wickets','Econ', 'Dots', '4s_Given', '6s_Given', 'Wd', 'Nb'])

for count in range(len_url):
    eventId = eventId_gp[count]
    #print(count)
    dff=get_scorecard(leagueId,eventId)
    print(dff)
    df=df.append(dff)
    print(count)
    sleep(2)
    #dff.to_csv("match1.csv")
    
      

#dff.drop(["id","shortText","text","preText","postText","currentBatsmen","currentBowlers","currentInning.balls","currentInning.runs","currentInning.wickets","matchOver.maiden","matchOver.runs","matchOver.wickets","matchOver.totalRuns","matchOver.totalWicket","matchOver.runRate","matchOver.requiredRunRate","matchOver.batsmen","matchOver.bowlers","matchOver.teamShortName","matchOver.remainingOvers","matchOver.remainingBalls","matchOver.remainingRuns","matchWicket.id","matchWicket.batsmanRuns","matchWicket.batsmanBalls","matchWicket.text"],axis=1,inplace=True)
df.to_csv("2009_scoreboard.csv")