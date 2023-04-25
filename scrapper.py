#link with more/better data: 
# apple:    https://markets.businessinsider.com/ajax/Valor_HistoricPriceList/908440/Jan.%2001%202000_Mar.%2009%202023/NDB
# amazon:   https://markets.businessinsider.com/ajax/Valor_HistoricPriceList/645156/Jan.%2001%201990_Mar.%2009%202023/NDB
# to get data, go to the individual stock website and scrap the ChartValor field. That is the 908440 and the 645156 in the above links
# You can then just substitute that valor number for all of the stocks you wantand then get the more detailed data that way. 
#it looks like the tags at the end are different, so NDB, NDN
#https://markets.businessinsider.com/index/dow_jones
from bs4 import BeautifulSoup
import requests
import time
import numpy as np
import pandas as pd
from datetime import date
from datetime import datetime
import os
numberToMonthMap = {'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}
targetDate = str(input('Enter date in the format YYYY-MM-DD:\nFor example, todays date is '+str(datetime.today())[0:11]+'\n')).split('-')
month = numberToMonthMap[targetDate[1]]
day = targetDate[2]
yearPart = targetDate[0][2:4]
#2023-03-17 is an example input

html = requests.get('https://markets.businessinsider.com/index/dow_jones').text
souped = BeautifulSoup(html,'html.parser')

unfiltered = souped.find_all('td', attrs={'class':'table__td'})
#^ all raw html
stock_details= []
for i in range(len(unfiltered)):
    if('<a href="/stocks/' in str(unfiltered[i])):
        #looking for the stock details url
        temp = str(unfiltered[i]).split('\n')
        stock_details.append(temp[1].split(' ')[1])
#full url:https://markets.businessinsider.com
urls=[]
tickers=[]
for obj in stock_details:
    urls.append("https://markets.businessinsider.com"+obj[6:len(obj)-1])
    tickers.append(obj[14:obj.find('-')])
    #adding the general link to the specific link and appending to urls list
print(urls)
print()
print(tickers)
#for windows os: 
tickerFile = open('C://Users//vasantgc//Documents//StockPredictions2.0//tickers.csv','w')
# tickerFile = open('/Users/gcvasanta/Desktop/BetterStockPredictor/StockPredictions2.0/tickers.csv','w')
tickerFile.write('Tickers\n')
for ticker in tickers:
    tickerFile.write(ticker+'\n')
# input('tickers')
tickerToLink={}
allIds = []
for link in urls:
    specSoup = BeautifulSoup(requests.get(link).text,'html.parser')
    var = (specSoup.find_all('script'))
    properIndex = 0
    for i in range(len(var)):
        if('var detailChartViewmodel' in str(var[i])):
            properIndex = i
            #finding the right index for the ChartValor as this index seems to change every so often
            break
    thatVariable = str(var[properIndex])
    uid = thatVariable[thatVariable.find('ChartValor')+15:thatVariable.find('ChartExchange')-4]
    allIds.append(uid)
    #getting the unique ChartValor ID and adding it to allIds list
    print(uid)
    print(link)
print(allIds)
print(len(allIds))
#if <bound method Response.json of <Response [503]>> has the number 503, that means the website doesn't exist
#to fix inconsistencies while reading, try finaggling with this line: (although this seems to be fixed as of 3/17/2023)
#requests.get(url, headers={"user-agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36"})
fullLinks = []
# refactor for i in range(len(allIds)): for loop; have a class that has the url...; but honestly I don't think its that necessary
# this loop makes an initial request to determine whether its NDB or NDN
# affixes the proper (NDB or NDN) request type to the link depending on what the if else outputs
# adds this full link to the fullLinks list

for i in range(len(allIds)):
    print(month)
    print(allIds[i])
    ooga = str(requests.get('https://markets.businessinsider.com/ajax/Valor_HistoricPriceList/'+str(allIds[i])+'/Jan.%2001%202000_'+month+'.%20'+day+'%2020'+yearPart+'/NDB').json)

    if(int(ooga[len(ooga)-6:len(ooga)-3])==200):#means its a valid request
        print("inside NDB is valid case")
        data = requests.get('https://markets.businessinsider.com/ajax/Valor_HistoricPriceList/'+str(allIds[i])+'/Jan.%2001%202000_'+month+'.%20'+day+'%2020'+yearPart+'/NDB').json()
        fullLinks.append('https://markets.businessinsider.com/ajax/Valor_HistoricPriceList/'+str(allIds[i])+'/Jan.%2001%202000_'+month+'.%20'+day+'%2020'+yearPart+'/NDB')
    else:
        print("Inside NDN is valid case")
        data = requests.get('https://markets.businessinsider.com/ajax/Valor_HistoricPriceList/'+str(allIds[i])+'/Jan.%2001%202000_'+month+'.%20'+day+'%2020'+yearPart+'/NDN').json()
        fullLinks.append('https://markets.businessinsider.com/ajax/Valor_HistoricPriceList/'+str(allIds[i])+'/Jan.%2001%202000_'+month+'.%20'+day+'%2020'+yearPart+'/NDN')
    print(allIds[i])
    
    print(len(data))
tickerToLink = dict(zip(tickers,fullLinks))
# the tickerTOLink dictionary maps stock tickers to their corresponding data URLs (for example, 'aapl' maps to the json URL with all of Apple's data)
for ticks in tickerToLink:
    #actually reads the raw numbers in the requested data files
    dataToRead = (requests.get(tickerToLink[ticks]).json())
    #for windows os: 
    fOpen = open('C://Users//vasantgc//Documents//StockPredictions2.0//data//'+ticks+".csv",'w')
    # fOpen = open('/Users/gcvasanta/Desktop/BetterStockPredictor/StockPredictions2.0/data/'+ticks+".csv",'w')
    fOpen.write("Open,Close,High,Low,Date\n") 
    #writes the Open, Close, High, Low, Date fields to the corresponding .csv file (appl.csv for example)
    #excludes the Volume field because the values in this column are inconsistent and the commas mess up the csv file (adds extra columns for commas)
    # descending order (most recent at the EOF) 
    for j in range(len(dataToRead)-1,-1,-1):
    # for j in range(len(dataToRead)):
            fOpen.write(str(dataToRead[j]['Open'])+','+str(dataToRead[j]['Close'])+','+str(dataToRead[j]['High'])+','+str(dataToRead[j]['Low'])+','+str(dataToRead[j]['Date'])+'\n')
#volumes get messed up because they have columns so it stretches across 3 columns in the csv file; also most of the volumes aren't even filled out
#so the volumes column was deleted