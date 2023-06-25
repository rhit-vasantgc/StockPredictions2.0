import pandas as pd
import numpy as np
import requests
#nasdaq: https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Index&tkData=1135,985336,1135,333&from=20140728&to=20230609
#sp500: https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Index&tkData=1059,998434,1059,333&from=20140728&to=20230609
#dow: https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Index&tkData=310,998313,310,333&from=20140728&to=20230609

nas = list(requests.get('https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Index&tkData=1135,985336,1135,333&from=20140728&to=20230615').json())
nas = nas[0:len(nas)-1]

sp500 = list(requests.get('https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Index&tkData=1059,998434,1059,333&from=20140728&to=20230615').json())
sp500 = sp500[0:len(sp500)-1]

dowj = list(requests.get('https://markets.businessinsider.com/Ajax/Chart_GetChartData?instrumentType=Index&tkData=310,998313,310,333&from=20140728&to=20230615').json())
dowj = dowj[0:len(dowj)-1]
# delete 2019-10-27 00:00 from nasdaq
# dow 2022-03-21 00:00 close: 34358.50; open: 34748.84; high: 34648.84; low: 34352.96

for i in range(len(nas)):
    if('2019-10-27 00:00' in nas[i]['Date']):
        # print(i)
        # print(nas[i]['Date'])
        nas[i]['Date'] = '2019-10-28 00:00'
        # print(nas[i])
        # input('stop')
        break
for i in range(len(dowj)):
    if('2022-03-18 00:00' in dowj[i]['Date']):
        # print(i)
        break
dowj.insert(1926, {'Close': 34358.50, 'Open': 34748.84, 'High': 34648.84, 'Low': 34352.96, 'Volume': 0, 'Estimate': 0, 'Date': '2022-03-21 00:00'})
ddates = []
ndates = []
spdates = []
for i in range(len(dowj)):
    ddates.append(dowj[i]['Date'])
for i in range(len(nas)):
    ndates.append(nas[i]['Date'])
for i in range(len(sp500)):
    spdates.append(sp500[i]['Date'])
for i in range(len(ndates)):
    if(spdates[i] not in ndates):
        print(spdates[i])
# print(nas[1323])

arr = ['Open','Close','High','Low']
alldj = {'Open': [], 'Close': [], 'High': [], 'Low': []}
allnasdaq = {'Open': [], 'Close': [], 'High': [], 'Low': []}
allsp500 = {'Open': [], 'Close': [], 'High': [], 'Low': []}
for i in range(len(dowj)):
    for j in range(len(arr)):
        alldj[arr[j]].append(dowj[i][arr[j]])
for i in range(len(nas)):
    for j in range(len(arr)):
        allnasdaq[arr[j]].append(nas[i][arr[j]])
for i in range(len(sp500)):
    for j in range(len(arr)):
        allsp500[arr[j]].append(sp500[i][arr[j]])
ticks = np.array(pd.read_csv('tickers.csv')['Tickers'])
print(ticks)
valid_ticks = []

all_dates = []
for tick in ticks:
    test = list(pd.read_csv('data//' + tick+'.csv')['Date'])
    # print(test)
    all_dates.append(test)
correct_len = len(list(pd.read_csv('data//aapl.csv')['Date']))
counter = 0
for i in range(len(all_dates)):
    if(len(all_dates[i])==correct_len):
        valid_ticks.append(ticks[i])
        counter+=1
    print(len(all_dates[i]))
print(counter)
print(valid_ticks)
cong_file = open('cong_file_dates.csv','w')
names = ['Dow Jones', 'Nasdaq', 'SP500']
cong_file.write('Date,')
for i in range(len(names)):
    for j in range(len(arr)):
        cong_file.write(arr[j] + ' ' + names[i] + ',')

for i in range(len(valid_ticks)-1):
    cong_file.write('Open ' + valid_ticks[i] + ',')
    cong_file.write('Close ' + valid_ticks[i] + ',')
    cong_file.write('High ' + valid_ticks[i] + ',')
    cong_file.write('Low ' + valid_ticks[i] + ',')
cong_file.write('Open ' + valid_ticks[len(valid_ticks)-1]+',')
cong_file.write('Close ' + valid_ticks[len(valid_ticks)-1]+',')
cong_file.write('High ' + valid_ticks[len(valid_ticks)-1]+',')
cong_file.write('Low ' + valid_ticks[len(valid_ticks)-1]+'\n')


aux = []
for i in range(len(arr)):
    aux.append(alldj[arr[i]])
for i in range(len(arr)):
    aux.append(allnasdaq[arr[i]])
for i in range(len(arr)):
    aux.append(allsp500[arr[i]])
print(aux)
print(len(aux))
for i in range(len(valid_ticks)):
    cur_file = pd.read_csv('data//'+valid_ticks[i]+'.csv')
    for j in range(len(arr)):
        aux.append(list(cur_file[arr[j]]))
   
    # print(aux)
    # print()
    # print(0)
    # print(aux[0])
    # print(len(aux))
    # # print()
    # # print()
    # # print(aux[0])
    # input('stop')
    # print(cur_file)

aux = np.ndarray.tolist(np.transpose(np.array(aux)))
#for i in range(len(aux)):

for i in range(len(aux)):
    cong_file.write(str(ddates[i][0:10])+',')
    for j in range(len(aux[i])-1):
        cong_file.write(str(aux[i][j])+',')
    cong_file.write(str(aux[i][len(aux[i])-1])+'\n')
    # input('stop')
print(valid_ticks)
print(len(valid_ticks))
# print(all_dates)