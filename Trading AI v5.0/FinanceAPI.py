"""
size = 365
queue = [0,size,[0 for _ in range(size)]] #qindex qsize qitems
#add item
def Qadd(q, item):
    q[2][q[0]] = item
    q[0] = (q[0] + 1) % q[1]
def orderedQ(q):
    return [q[2][((q[0] + i)%q[1])] for i in range(q[1])]
for i in range(366):
    Qadd(queue,i)
print(orderedQ(queue))
"""
import yfinance as yf

from datetime import datetime, timedelta
import random
STOCK = None
def noise():
   return 1 + ((10000 - (random.randrange(51) + 9975)) / 10000)
def getData():
   stock = yf.download(STOCK)
   history = stock["Close"].astype(float)
   return format(history)
def format(data):
   temp = [0.0 for _ in range(len(data))]
   for x in range(1,len(data)):
      temp[x] = float(data.iloc[x])
   return temp
def noisify(data):
   stretch = noise()
   shift = noise()*data[0]
   return [(x*stretch + shift) for x in data]
"""
date = datetime.today().strftime('%Y-%m-%d')
day = pickrandom(date, stock)
lastprice = checkprice(day,stock)
percentage = 1
for _ in range(365):
    newday = nextday(day)
    newprice = checkprice(newday, stock)
    # AI bought = 0 or 1
    percentage *= percentChange(lastprice,newprice)
    print(percentage,lastprice,newprice)
    day = newday
    lastprice = newprice
"""