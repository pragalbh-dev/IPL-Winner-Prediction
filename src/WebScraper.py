#!/usr/bin/env python
# coding: utf-8

# In[20]:


import bs4 as bs
from urllib.request import Request, urlopen
from http.client import IncompleteRead
import csv
import os


# In[21]:


def parse(url):
    req = Request( url, headers={'User-Agent': 'Chrome/5.0'})

    sauce = urlopen(req).read()

    soup = bs.BeautifulSoup(sauce,'lxml')

    result = soup.find('div',attrs={'class':'cb-col cb-scrcrd-status cb-col-100 cb-text-complete'})
    
    won_index = result.string.find('won')

    winning_team = result.string[:won_index].strip()
    
    words = soup.findAll('div',attrs={'class' : 'cb-col cb-col-73'})

    #for para in words:
    #    print(para.text)

    first = words[0].string

    vs_index = first.find('vs')
    end_index = first.find(',')

    team_1 = first[1:vs_index]
    
    team_2 = first[vs_index+3:end_index]
    
    if( result.string.find('abandoned') != -1):
        return [team_1, team_2, 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', 'NA', [], [], 'NA']

    second = words[2].string[1:].strip()

    won_index = second.find('won')

    toss_won_by = second[:won_index].strip()

    

    last_space_at = second.rindex(' ')

    decision = second[last_space_at:].strip()

    

    venue=words[4].string.strip()

    
    try:
        umpire_1, umpire_2 = words[5].string.strip().split(',')

        umpire_1 = umpire_1.strip()
        umpire_2 = umpire_2.strip()

        umpire_3 = words[6].text.strip()
        referee = words[7].text.strip()

    except:
        umpire_1 = 'NA'
        umpire_2 = 'NA'
        umpire_3 = 'NA'
        referee = 'NA'
    

    

    body = soup.body


    Total = body.get_text()

    #print(Total)

    time_start = Total.find('Date & Time')
    time_end = Total.find('LOCAL')
    
    time = Total[ time_start : time_end ]
    
    time_start=time.find(',')
    
    time = time[time_start + 2 : -1]
    
    squad1_start = Total.find('Playing')
    squad1_end = Total.find('Bench')

    squad2_start = Total.rindex('Playing')
    squad2_end = Total.rindex('Bench')

    squad1 = Total[squad1_start+7 : squad1_end]
    squad2 = Total[squad2_start+7 : squad2_end]

    list_1 = []
    list_2 = []
    for player in squad1.split(','):
        index=player.find('(')
        if( index != -1 ):
            player=player[:index]
        player = player.strip()
        list_1.append(player)

    

    for player in squad2.split(','):
        index=player.find('(')
        if( index != -1 ):
            player=player[:index]
        player = player.strip()
        list_2.append(player)
    
    #print(winning_team)
    #print(team_1)
    #print(team_2)
    #print(toss_won_by)
    #print(decision)
    #print(venue)
    #print(time)
    #print(umpire_1)
    #print(umpire_2)
    #print(umpire_3)
    #print(referee)
    #print(list_1)
    #print(list_2)
    
    return [team_1, team_2, toss_won_by, decision, venue, time, umpire_1, umpire_2, umpire_3, referee, list_1, list_2, winning_team ]

#parse('https://www.cricbuzz.com/live-cricket-scorecard/10633/dc-vs-kkr-26th-match-indian-premier-league-2010')


# In[22]:



req = Request( 'https://www.cricbuzz.com/cricket-series/2059/indian-premier-league-2009/matches', headers={'User-Agent': 'Chrome/5.0'})

sauce = urlopen(req).read()

soup = bs.BeautifulSoup(sauce,'lxml')

body = soup.body

matches = body.findAll('a',href=True,class_='text-hvr-underline')

link = 'https://www.cricbuzz.com/'

link = link + 'live-cricket-scorecard'


counter=1

fields = ['team_1','team_2','toss_won_by','decision','venue', 'time','umpire_1','umpire_2','umpire_3','referee','list_1','list_2','winning_team']
year = '2009'
dir = os.getcwd()
new_dir = dir[:-4] + '\\data\\' + year + '_match_list.csv' 

with open(new_dir, 'w',newline='') as csvfile:  
    
    csvwriter = csv.writer(csvfile)  
    
    csvwriter.writerow(fields)
    
    for m in matches :

        if(m.text not in ['The Times of India', 'Navbharat Times'] ):
            print(counter)
            counter = counter+1
            print(m.text)
            
            next_part = m['href'][1:]
            index = next_part.find('/')
            total_link = link + next_part[index:]
            
            try:
                data_entry = parse(total_link)
            except IncompleteRead:
                continue
            
            csvwriter.writerow(data_entry)
    



# In[ ]:



    
  


# In[ ]:




