import pandas as pd
import numpy as np
data = np.array(pd.read_csv('C://Users//vasantgc//Documents//StockPredictions2.0//normalized_data//normalized_wba.csv'))
#a positive difference is when the current value is bigger than the previous value
#a negative difference is when the current value is smaller than the previous value

#the order for all of these arrays is [open, close, high, low]
positiveDiffs = [0,0,0,0] #final array for number of positive differences
negativeDiffs = [0,0,0,0] #final array for number of negative differences 
maxStreakPositive = [0,0,0,0] #final array for maximum positive streak
maxStreakNegative = [0,0,0,0] #final array for maximum negative streak
auxPos = [1,1,1,1] #auxilary holder for positive streaks
auxNeg = [1,1,1,1] #auxilary holder for negative streaks
posSwitch = [False,False,False,False] #stores booleans for if it needs to terminate a positive streak
negSwitch = [False,False,False,False] #stores booleans to check if need to terminate a negative streak
posStreakEndIndex = [0,0,0,0]
negStreakEndIndex = [0,0,0,0]
for i in range(1,len(data)-1):
    
    for j in range(len(data[i])):
        if(data[i][j]>data[i-1][j] and data[i+1][j]>data[i][j]): #check if current is greater than previous and next is greater than current
            #this case is for a continuing positive streak
            positiveDiffs[j]+=1 #increment corresponding value in positive diff array
            print("double positive")
            auxPos[j]+=1 #increment auxilary array
            posSwitch[j] = False #set state to false (no need to terminate streak)
        elif(data[i][j]>data[i-1][j] and data[i+1][j]<data[i][j]): #check if current is greater than previous and next is less than current
            #this case is for the last value in an active positive streak (the next value will initiate a negative streak)
            positiveDiffs[j]+=1 #increment corresponding value in positive diff array
            print("positive then negative")
            auxPos[j]+=1 #increment auxilary array one last time
            posSwitch[j] = True #set state to true (need to terminate streak)
            
        if(data[i][j]<data[i-1][j] and data[i+1][j]<data[i][j]): #check if current is less than previous and next is less than current
            #this case is for a continuing negative streak
            negativeDiffs[j]+=1 #increment corresponding value in negative diff array
            print("double negative")
            auxNeg[j]+=1 #increment auxilary array
            negSwitch[j] = False #set state to false (no need to terminate streak)
        elif(data[i][j]<data[i-1][j] and data[i+1][j]>data[i][j]): #check if current is less than previous and next is greater than current
            #this case is for the last value in an active negative streak (the next value will initiate a positive streak)
            negativeDiffs[j]+=1 #increment corresponding value in negative diff array 
            print("negative then positive")
            auxNeg[j]+=1 #increment auxilary array one last time
            negSwitch[j] = True #set state to true (need to terminate streak)
            
        print(data[i-1][j])
        print(data[i][j])
        print(data[i+1][j])
        print()
        print()
    for m in range(len(auxPos)):
        if(posSwitch[m]==True): #checking if need to terminate streak
            auxPos[m] = 1 #actually terminate streak
        if(negSwitch[m]==True): 
            auxNeg[m] = 1
    for k in range(len(auxPos)):
        if(auxPos[k]>maxStreakPositive[k]):
            maxStreakPositive[k] = auxPos[k] #update max positive streak
            posStreakEndIndex[j] = i #set the streak end index
        if(auxNeg[k]>maxStreakNegative[k]):
            maxStreakNegative[k] = auxNeg[k] #update max negative streak
            negStreakEndIndex[j] = i #set the streak end index
    print(auxPos)
    print(auxNeg)
    print(posSwitch)
    print(negSwitch)
    print(maxStreakPositive)
    print(maxStreakNegative)
    # input('stop')
        # # print(data[i][j])
        # # print(data[i-1][j])
        # if(data[i][j]>data[i-1][j]):
        #     positiveDiffs[j]+=1
        #     #print(positiveDiffs)
        # else:
        #     negativeDiffs[j]+=1
        # # input(j)

print(data)
print(positiveDiffs)
print(negativeDiffs)
print()
print()
print(maxStreakPositive)
print(maxStreakNegative)
print()
print()
print(posStreakEndIndex)
print(negStreakEndIndex)