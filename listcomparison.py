
# List comparison function that gets called with each triggering of the camera

from datetime import date, timedelta

# Get current date and time
today = date.today()

def listcompare (FridgeOld, FridgeNew, detected, var, ExpirationDays):
    today = date.today()
    for row in detected:
        food=row[0]
        if FridgeOld.get(food):
            for iteration in range(len(FridgeOld[food])):
                if row[1]<=int(FridgeOld[food][iteration][0])+var and row[1]>=int(FridgeOld[food][iteration][0])-var:
                    if row[2]<=int(FridgeOld[food][iteration][1])+var and row[2]>=int(FridgeOld[food][iteration][1])-var:
                        match=iteration
                        break
                    else:
                        match=-1
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

            row.insert(3,expiry_date.day)
            row.insert(4,expiry_date.month)
            FridgeNew[food].append(row[1:5])
        print(food)
