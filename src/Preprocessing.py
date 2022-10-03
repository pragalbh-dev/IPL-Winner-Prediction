#!/usr/bin/env python
# coding: utf-8

# In[56]:


import csv
import pandas as pd
import re
import os


# In[57]:




year = 2008

filename = str(year) + '_match_list.csv'

dir = os.getcwd()

new_dir = dir[:-4] + '\\data\\'

#print(filename)


# In[58]:


def remove_blanks(filename):
    #Scraped data file had leading and trailing spaces in 1st column. This function removes those
    with open(new_dir + filename,'r') as inp, open(new_dir + 'temp.csv','w',newline='') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            row[0]=row[0].strip()
            writer.writerow(row)

    with open(new_dir + 'temp.csv','r') as inp, open(new_dir + filename,'w',newline='') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            writer.writerow(row)


# In[59]:


#making list of teams for dictionary key list

team_list=set()

for year in range(2008,2020):
    filename = str(year) + '_match_list.csv'
    
    print(new_dir+filename)
    
    with open(new_dir + filename,'r') as inp:
        for row in csv.reader(inp):
            if row[0] != 'team_1':
                team_list.add(row[0].strip())
                team_list.add(row[1].strip())


# In[60]:


short_form = {'Deccan Chargers':'DCG',
              'Hyderabad':'SRH',
              'Sunrisers Hyderabad':'SRH',
              'Chennai Super Kings':'CSK',
              'Kings XI Punjab':'KXIP',
              'Pune Warriors':'PWI',
              'Gujarat Lions':'GL',
              'Rising Pune Supergiant':'RPS',
              'Rising Pune Supergiants':'RPS',
              'Rajasthan Royals':'RR',
              'Mumbai Indians':'MI',
              'Kolkata Knight Riders':'KKR',
              'Royal Challengers Bangalore':'RCB',
              'Delhi Daredevils':'DC',
              'Delhi Capitals':'DC',
              'Kochi Tuskers Kerala':'KTK'}


# In[61]:


def replace_name(filename):
    #to replace name with its abbreviation( Chennai Super Kings -> CSK , etc)
    with open(new_dir + filename,'r') as inp, open(new_dir + 'temp.csv','w',newline='') as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            if row[-1] != 'winning_team':
                row[2] = short_form[ row[2] ]
                row[-1] = short_form [ row[-1] ]
            writer.writerow(row)

    with open(new_dir + 'temp.csv','r') as inp, open(new_dir + filename,'w',newline='') as out:
        writer = csv.writer(out)
        
        for row in csv.reader(inp):
            writer.writerow(row)


# In[62]:


team_stats = {}


# In[63]:


#dictionary to keep track of stats and construct features

for tname in team_list:
    team_stats[tname]= {'won': 1,'won_when_toss_won': 1,'won_when_toss_lost': 1,'won_when_bat': 1,'won_when_chase': 1,'total': 2}

#print(team_stats)
#print( team_stats['SRH']['total'])

def print_custom(team_stats):
    for key,value in team_stats.items():
        print(key)
        print(value['total'])
print(team_list)


# In[64]:


def update_team_stats(row):
    #updates team stats after each game
    new_entry=row
    team_1 = row[1]
    team_2 = row[2]
    
    
    tot = team_stats[team_1]['total']
    new_entry.append( team_stats[team_1]['won'] / tot )
    new_entry.append( team_stats[team_1]['won_when_toss_won'] / tot )
    new_entry.append( team_stats[team_1]['won_when_toss_lost'] / tot )
    new_entry.append( team_stats[team_1]['won_when_bat'] / tot )
    new_entry.append( team_stats[team_1]['won_when_chase'] / tot )
    
    tot = team_stats[team_2]['total']
    new_entry.append( team_stats[team_2]['won'] / tot )
    new_entry.append( team_stats[team_2]['won_when_toss_won'] / tot )
    new_entry.append( team_stats[team_2]['won_when_toss_lost'] / tot )
    new_entry.append( team_stats[team_2]['won_when_bat'] / tot )
    new_entry.append( team_stats[team_2]['won_when_chase'] / tot )
        
    
    team_stats[team_1]['total'] = team_stats[team_1]['total'] + 1
    
    team_stats[team_2]['total'] += 1
    
    winner = row[13]
    
    team_stats[winner]['won'] += 1
    
    toss_winner = row[3]
    
    if toss_winner == winner :
        team_stats[winner]['won_when_toss_won'] += 1
        
        if row[4] == 'bat':
            team_stats[winner]['won_when_bat'] += 1
        else:
            team_stats[winner]['won_when_chase'] += 1
            
    else:
        team_stats[winner]['won_when_toss_lost'] += 1
        
        if row[4] == 'bat':
            team_stats[winner]['won_when_chase'] += 1
        else:
            team_stats[winner]['won_when_bat'] += 1
    
    
    return new_entry
        


# In[65]:


#dictionary to store whether team lost/won last few matches

history = {}

for tname in team_list:
    history[tname] = 0


# In[66]:


def add_form(row):
    
    #adds form column for both teams to database
    # won last match - 5
    # won last 2 matches - 10
    # anything else - 0
    
    
    
    new_entry = row
    
    team_1 = row[1]
    team_2 = row[2]
    
    new_entry.append( history[team_1] )
    new_entry.append( history[team_2] )
    
    winner = row[13]
    
    if( history[winner] in [0,5] ):
        history[ winner ] += 5
    
    if team_1 != winner :
        loser = team_1
    else:
        loser = team_2
        
    history[ loser ] = 0
    
    return new_entry


# In[43]:


with open(new_dir + 'DB_proto1.csv','r') as inp,open(new_dir + 'temp.csv','w',newline='') as out:
    
    #rearranging columns to have Y(winner) as last column
    
    csvreader = csv.reader(inp)
    
    csvwriter = csv.writer(out)
    
    fields = next(csvreader)
    
    fields.remove('Winner')
    fields.append('Form1')
    fields.append('Form2')
    fields.append('Winner')
    
    
    csvwriter.writerow(fields)
    
    for row in csvreader:
        
        new_entry = add_form(row)
        
        elem = new_entry[13]
        
        del new_entry[13]
        
        new_entry.append(elem)
        
        csvwriter.writerow(new_entry)


# In[67]:


#file containing pre-constructed player features post each season

read_file = pd.read_excel(new_dir + 'player_stats_final.xlsx')


# In[68]:


read_file.to_csv(new_dir + 'temp.csv',index=None,header=True)


# In[69]:


player_stats = {}
total_players = []


# In[70]:


with open(new_dir + 'temp.csv','r') as inp:
    
    #adding player features to dictionary
    csvreader = csv.reader(inp)
    
    fields=next(csvreader)
    
    del fields[0]
    
    #print(fields)
    
    for row in csvreader:
        
        player_stat = {}
        for index,value in enumerate(fields):
            player_stat[value] = row[index+1]
            
        #print(player_stat)
        name = row[0]
        total_players.append(name)
        
        player_stats[name] = player_stat


# In[71]:


#print(total_players)


# In[72]:


def list_players( squad1_list ):
    
    #convert string to list
    squad1 = squad1_list[1:-1]
    squad1_list = []
    
    for name in squad1.split(','):
        start_index = name.find('\'')
        end_index = name.rfind('\'')
        #print(start_index)
        #print(end_index)
        squad1_list.append( name[start_index+1:end_index] )
        
    return squad1_list


# In[75]:


def get_stats( player_stats, player_name, year):
    
    #retrieves feature values from dictionary given name and year
    
    #print('name : ' , player_name,' year : ',year)
    stat_list = player_stats[ player_name ]
    
    reqd_list = []
    
    r = re.compile("post_" + str( int(year) - 1) + ".*")
    for key,val in stat_list.items():
        if r.match(key):
            #print("matched with " + key)
            reqd_list.append(val)
    
    for index,val in enumerate(reqd_list):
        reqd_list[index] = float(val)
    #print( len(reqd_list) )
    return reqd_list


# In[76]:


with open(new_dir + 'DB_proto2.csv','r') as inp,open(new_dir + 'temp.csv','w',newline='') as out:
    
    
    
    csvreader = csv.reader(inp)
    csvwriter=csv.writer(out)
    
    fields = next(csvreader)
    
    fields.extend(['Team1_F','Team1_G','Team1_H','Team1_I','Team1_J','Team1_K',
                   'Team2_F','Team2_G','Team2_H','Team2_I','Team2_J','Team2_K'])
    
    csvwriter.writerow(fields)
    
    
    
    for row in csvreader:
        
        new_entry = row
        team1_F = 0
        team1_G = 0
        team1_H = 0
        team1_I = 0
        team1_J = 0
        team1_K = 0
        team2_F = 0
        team2_G = 0
        team2_H = 0
        team2_I = 0
        team2_J = 0
        team2_K = 0
        
        if row[0] in ['2008','2009','2010']:
            new_entry.extend([0,0,0,0,0,0,0,0,0,0])
            csvwriter.writerow(new_entry)
            continue
        
        year = row[0]
        
        squad1_list = list_players(row[11])
        squad2_list = list_players(row[12])
        
        
        
        count = 0
        
        for name in squad1_list:
            flag = True
            index = name.find(' ')
            match = name[0] + ".*" + name[index+1:]
            
            r1 = re.compile(match)
            r2 = re.compile(match[1:])
            #print(match)
            
            for player_name in total_players:
                if r1.match(player_name) or r2.match(player_name):
                    #print("T1 matched with " + player_name)
                    
                    
                    player_stat = get_stats( player_stats, player_name, year)
            
                    #print('hard_hitter :',player_stat[0])
                    team1_F += player_stat[0]
                    team1_G += player_stat[1]
                    team1_H += player_stat[2]
                    team1_I += player_stat[3]
                    team1_J += player_stat[4]
                    team1_K += player_stat[5]
                    
                    break
                else:
                    continue
            
                
        for name in squad2_list:
            flag = True
            index = name.find(' ')
            match = name[0] + ".*" + name[index+1:]
            
            r1 = re.compile(match)
            r2 = re.compile(match[1:])
            #print(match)
            
            for player_name in total_players:
                if r1.match(player_name) or r2.match(player_name):
                    #print("T2 matched with " + player_name)
                    
                    player_stat = get_stats( player_stats, player_name, year)
                    
                    team2_F += player_stat[0]
                    team2_G += player_stat[1]
                    team2_H += player_stat[2]
                    team2_I += player_stat[3]
                    team2_J += player_stat[4]
                    team2_K += player_stat[5]
        
                    break
                else:
                    continue
        
        #print( team1_F,team1_G,team1_H,team1_I,team1_J,team2_F,team2_G,team2_H,team2_I,team2_J )
        
        #print('NEW ENTRY!!',year)
        
        new_entry.extend( [ team1_F,team1_G,team1_H,team1_I,team1_J,team1_K,team2_F,team2_G,team2_H,team2_I,team2_J,team2_K] )
        
        csvwriter.writerow(new_entry)
        


# In[77]:


def rearrange( row ):
    
    #rearrage to have Team1 and Team2 features in sequence and Y as last col
    team2_part1 = row[18:23]
    team1_part2 = row[26:32]
    team2_part2 = row[32:38]
    final_part = row[23:26]
    
    new_row = row[:18]
    new_row.extend(team1_part2)
    new_row.extend(team2_part1)
    new_row.extend(team2_part2)
    new_row.extend(final_part)
    
    return new_row



# In[78]:


with open(new_dir + 'temp.csv','r') as inp,open(new_dir + 'DB_1.csv','w',newline='') as out:
    
    #rearranging cols
    csvreader=csv.reader(inp)
    
    csvwriter=csv.writer(out)
    
    fields = next(csvreader)
    
    new_field = rearrange(fields)
    
    csvwriter.writerow(new_field)
    
    for row in csvreader:
        new_row = rearrange(row)
        csvwriter.writerow(new_row)


# In[79]:


with open(new_dir + 'temp.csv','r') as inp,open('DB_2.csv','w',newline='') as out:

    #rearranging cols
    csvreader=csv.reader(inp)
    csvwriter=csv.writer(out)
    
    
    fields = next(csvreader)
    
    new_field = fields
    
    new_field[-3], new_field[-2], new_field[-1] = new_field[-2],new_field[-1],new_field[-3]
    
    csvwriter.writerow(new_field)
    
    for row in csvreader:
        new_row = row
        new_row[-3], new_row[-2], new_row[-1] = new_row[-2],new_row[-1],new_row[-3]
        csvwriter.writerow(new_row)


# In[ ]:




