
#switch activated, new is copied to old and new is emptied old=new.copy()

from datetime import date, timedelta

# Get current date and time
today = date.today()

def listcompare (FridgeOld, FridgeNew, detected, var, ExpirationDays):
    today = date.today()
    for row in detected:
        food=row[0]
        if FridgeOld.get(food):
            for iteration in range(len(FridgeOld[food])):
                if row[1]<=FridgeOld[food][iteration][0]+var and row[1]>=FridgeOld[food][iteration][0]-var:
                    if row[2]<=FridgeOld[food][iteration][1]+var and row[2]>=FridgeOld[food][iteration][1]-var:
                        match=iteration
                        break
                else:
                    match=-1 
        else:
            match=-1
            
        if match!=-1:
            if not FridgeNew.get(food):
                FridgeNew[food]=[]
           # days_passed=  ((int)(today.day)+ExpirationDays[food])-FridgeOld[food][match][2]
           # FridgeOld[food][match][3] -= days_passed
            FridgeNew[food].append(FridgeOld[food][match])
        else:
            if not FridgeNew.get(food):
                FridgeNew[food]=[]
            expiry_date = today + timedelta(days=ExpirationDays[food])
            #days_left= (int) (expiry_date.day)- (int) (today.day)
            row.extend([expiry_date.day,expiry_date.month])
            FridgeNew[food].append([row[1:5]]) 
            print('hello')



var=0 #how much variation is allowed from the central point 

FridgeOld={        
    'apple': [[2,1,13,12],[3,5,13,12]],

    'tomato':[[2,4,14,12]],
    }   

ExpirationDays = {}
file = open("ExpirationDays.txt",'r')
for line in file:
    key, value = line.split(':')
    ExpirationDays[key] = (int) (value)
print(ExpirationDays)



FridgeNew=dict()
detected= [['apple',2,1],['apple', 4,3], ['onion', 5,7], ['carrot', 2,4]] 


listcompare (FridgeOld, FridgeNew, detected, var, ExpirationDays)  

print(FridgeNew)
