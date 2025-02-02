import Indicators as ind
import ctypes
def mpthink(args):
   inputs,bots,shape = args
   layers = len(shape)
   #neurons = sum(shape[1:])
   #layers = len(shape[1:])
   # run fitness for each bot
   
   sizeofinputs = len(inputs)
   indicate = [ind.calculate_indicators(inputs[x:51+x]) for x in range(sizeofinputs-49)]
   inputs = inputs[51:] # skip the first 51 because they have no inputs/ indicators and 51 because it is the first day and 
   inp = [indicate[x // 6][x % 6] for x in range((sizeofinputs - 49)*6)] # 6 for each indicator
   mylib = ctypes.CDLL("./Fitness.so")
   mylib.fitness.argtypes = [ctypes.POINTER(ctypes.c_float),ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int,  ctypes.POINTER(ctypes.c_float)]
   mylib.fitness.restype = None
   inpc = (ctypes.c_float * len(inp))(*inp)
   pricesc = (ctypes.c_float * len(inputs))(*inputs)
   shapec = (ctypes.c_int * len(shape))(*shape)
   weightcount = len(bots[0])

   botcount = len(bots)
   outputs = [ctypes.c_float() for _ in range(botcount)]
   for x in range(botcount):
      (mylib.fitness((ctypes.c_float * weightcount)(*bots[x]), inpc,sizeofinputs - 51, pricesc, shapec, layers, ctypes.byref(outputs[x])))
   return outputs
if __name__ == "__main__":
   import FinanceAPI as Fi
   import multiprocessing as mp
   import os
   import math
   import random
   import time
   netshape = [7,14,7,4,2,1]
   pop_per_core = 1000
   percentsaved = .1
   mutationpercent = .25
   CPU_CORES = os.cpu_count()
   population = pop_per_core*CPU_CORES
   def fileExists(path):
      try:
         open(path,"r")
         return True
      except:
         return False
   def clearFile(path):
      file = open(path,"w")
      file.writelines([""])
      file.close()
   def save(agentlist):
      for a in range(len(agentlist)):
         file = open(f"./Agents/{a}WeightsBiases.txt","w")
         file.writelines([str(x)+"\n" for x in agentlist[a]])
         file.close()
   def load():
      outagents = []
      for a in range(population):
         try:
            file = open(f"./Agents/{a}WeightsBiases.txt","r")
            outagents.append([float(x) for x in file.readlines()])
            file.close()
         except:
            print(a,"Failed to Load")
            raise "ERROR"
      return outagents
   print(not(fileExists(f"./Agents/{population - 1}WeightsBiases.txt")),fileExists(f"./Agents/{population}WeightsBiases.txt"))
   if not(fileExists(f"./Agents/{population - 1}WeightsBiases.txt")) or fileExists(f"./Agents/{population}WeightsBiases.txt"):
      print("Deleted")
      os.system(f"rmdir /s /q Agents")
      os.makedirs(f"{os.path.dirname(os.path.realpath(__file__))}/Agents")
      os.makedirs(f"{os.path.dirname(os.path.realpath(__file__))}/Agents/Outputs")
      biases = sum(netshape[1:])
      weights = sum([netshape[x] * netshape[x+1] for x in range(len(netshape) - 1)])
      agents = [[random.uniform(-1, 1) for _ in range(biases + weights)] for _ in range(population)] # initialize ais with values randomly between 1 and -1
      save(agents)
   else:
      agents = load()
   print(len(agents[0]), "Connections")
   def noise(num):
      #return random.random()
      #return num + (num*(2*(random.randrange(variation + 1)/variation) - 1)) + .0001 # ADD RANDOM value between 1 and -1 * current number
      return num*random.uniform(-2, 2) + random.uniform(-1, 1) # ADD RANDOM value between 1 and -1 and stretch the original
   def mutate(array):
      copy = array[:]
      if mutationpercent == 0:
         return copy
      for _ in range(1,random.randrange(int(round(mutationpercent*len(array))))): # how many genes are mutated
         x = random.randrange(len(array)) # which genes
         copy[x] = noise(array[x])
      return copy
   def train(daycount, newagents, randyear,generations):
      botsPerCore = population // CPU_CORES
      # run fitness functions
      #randyear = Fi.noisify(randyear)  
      score = [0.0 for _ in range(population)]
      for _ in range(generations): # train agents for _ runs before sorting and mutating
         if len(randyear) - (daycount+netshape[0]) > 0:
            i = random.randrange(len(randyear) - (daycount+netshape[0])) # random starting position
            trainyear = randyear[i:i + (daycount+netshape[0])]
         else:
            trainyear = randyear
         #for _ in range(5):
         #   (mpthink((trainyear,newagents[botsPerCore*19:botsPerCore*19+botsPerCore],netshape)))
         with mp.Pool(CPU_CORES) as p:
            outputs = p.map(mpthink, [(trainyear,newagents[botsPerCore*num:botsPerCore*num+botsPerCore],netshape) for num in range(CPU_CORES)])
            #compile scores
            output = [outputs[x // botsPerCore][x % botsPerCore] for x in range(population)]
            # fitness
            for y in range(population):
               score[y] += output[y].value
      sortedagents = sorted([(newagents[x], score[x]) for x in range(population)], key=lambda x: x[1])
      newagents = [x[0] for x in sortedagents]
      # MUTATE
      botsSaved = math.floor(population*percentsaved)
      for x in range(population - botsSaved):
         newagents[x] = mutate(sortedagents[random.randrange(botsSaved) + (population-botsSaved)][0])
      avg = sum([sortedagents[x][1] for x in range(len(sortedagents))])/len(sortedagents)
      print(daycount,"AVG SCORE",(avg/generations),"BEST SCORE",(sortedagents[-1][1]/generations), "MEDIAN",(sortedagents[len(sortedagents)//2][1]/generations),"AVG PERFORMANCE OF STOCK",(trainyear[-1]-trainyear[0]))
      return newagents
   def running():
      file = open("Running.txt","r")
      bit = file.read()
      file.close()
      if bit == "1":
         return True
      return False
   def test(testdata,bot):
      # FITNESS FUNCTION
      
      output = [testdata[-365] for _ in range(365)]
      #performance by bot
      feedback = [0.0 for _ in range(1)]
      bought = False
      buys = []
      sells = []
      offset = 51
      #total change in stock
      daysToCalc = 51
      testdata = testdata[-(365+daysToCalc):]
      testout = [1.0 for _ in range(365)]
      for x in range(365):
         testout[x] = testdata[x+daysToCalc]
      confidence = 0
      #ai change in stock
      import ctypes
      mylib = ctypes.CDLL("./Fitness.so")
      mylib.think.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
      mylib.think.restype = None

      for x in range(1,365):
         if bought:
            feedback[0] = 1.0
         else:
            feedback[0] = 0.0
         #print("RAN",testdata[x:offset+x]+ feedback[:],offset)
         inp = ind.calculate_indicators(testdata[x:51+x]) + feedback[:]
         buffer = (ctypes.c_float * netshape[-1])()
         mylib.think((ctypes.c_float * len(inp))(*inp),(ctypes.c_float * len(bot))(*bot),(ctypes.c_int * len(netshape))(*netshape),len(netshape),buffer)
         decision = buffer[0]
         #print(decision)
         confidence += abs(decision)
         if decision > 0: # min estimate to buy
            if bought == False:
               bought = True
               buys.append(x)
            
         else:
            if bought == True:
               bought = False
               sells.append(x)
            
         if bought:
            output[x] = output[x - 1]*(1+((testdata[offset+x]-testdata[offset+(x-1)])/testdata[offset+(x-1)]))
         else:
            output[x] = output[x - 1]
      """
      #TESTING
      for x in range(365):
         test = ind.calculate_indicators(testdata[x:51+x])[1]
         output[x] = test
         testout[x] = test
      """
      print("AVG Confidence of test",confidence/365)
      Graph.update((testout,output,buys,sells))
   file = open("Running.txt","w")
   file.write("1")
   file.close()
   stocks = ["SPY","WEAT","CORN","TSLA","INTC","NVDA"]
   Fi.STOCK = "WEAT"
   import Graph
   Graph.stock = Fi.STOCK
   Graph.init()
   # Get current data to see how AI performs on unknown data
   data = Fi.getData()
   testingdata = data[-(365+51):]
   data = data[1:-365] # 1 because first datapoint is always 0
   test(testingdata,agents[-1])
   #
   runs = 0
   maxscore = 0.0
   while running():
      runs = (runs + 1)%10 # runs 
      start = time.time()
      # len(data) train on all data
      # train on a year to all data random.randrange(365, len(data))
      agents = train(len(data), agents,data, 1) # train the ai on the same random year for 3 generations
      test(testingdata,agents[-1])
      totalChange = 1
      if runs == 0:
         # display test on current year
         print("Saving...")
         save(agents)
         print("Saved")
      print(f"Time Taken: {time.time() - start}")
