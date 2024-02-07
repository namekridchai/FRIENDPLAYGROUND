
import ast
import os,sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import random
from dataclasses import dataclass
from teamAgent import *
function_name = sys.argv[1]

print('hello python')

CARD_DICT = {'Hearts':{2:0,3:1,4:2,5:3,6:4,7:5,8:6,9:7,10:8,'J':9,'Q':10,'K':11,'A':12},
            'Diamonds':{2:13,3:14,4:15,5:16,6:17,7:18,8:19,9:20,10:21,'J':22,'Q':23,'K':24,'A':25},
            'Clubs':{2:26,3:27,4:28,5:29,6:30,7:31,8:32,9:33,10:34,'J':35,'Q':36,'K':37,'A':38},
            'Spades':{2:39,3:40,4:41,5:42,6:43,7:44,8:45,9:46,10:47,'J':48,'Q':49,'K':50,'A':51},

            }
@dataclass
class easyAgent: 
    minimumBidWhenBigAndLong: int = 65
    minimumBidWhenBigAndShort: int = 60
    minimumBidWhenSmallAndLong: int = 60
    minimumBidWhenSmallAndShort: int = 60
    probWhenCurrentBiddingExceedMinumum : int = 0.70
    fileName : str = 'easy_bot' 

@dataclass
class mediumAgent:
    minimumBidWhenBigAndLong: int = 70
    minimumBidWhenBigAndShort: int = 60
    minimumBidWhenSmallAndLong: int = 70
    minimumBidWhenSmallAndShort: int = 60
    probWhenCurrentBiddingExceedMinumum: int = 0.60
    probSelectBestTrump: int = 0.6
    fileName : str = 'medium_bot' 

@dataclass
class hardAgent:
    minimumBidWhenBigAndLong: int = 75
    minimumBidWhenBigAndShort: int = 55
    minimumBidWhenSmallAndLong: int = 70
    minimumBidWhenSmallAndShort: int = 55
    probWhenCurrentBiddingExceedMinumum: int = 0.50
    probSelectBestTrump: int = 1
    fileName : str = 'hard_bot' 

botList = [easyAgent(),mediumAgent(),hardAgent()]


# def get_name(parameter):
#     return 'john' + str(parameter)
# if function_name == 'get_name':
#     name = get_name(parameter)
#     print(name)
#     print()


def evaluate_cards(card_ids):  
    isLong  = False
    isBig = False
   
    for i in range(0,52,13):
        suite = list(filter(lambda x : i<=x<=i+12,card_ids)) 
        if len(suite)>=6:
            isLong  = True
           
        suite = list(filter(lambda x : i+9<=x<=i+12,card_ids))     
        if len(suite)>=3:
            isBig = True   

    return isLong,isBig



#  func bidding accept list of  card id in hand and current max bid and difficulty
#   return bidding number
#  return 0 mean give up

def bidding(card_ids,current_bid,difficulty):
    
    bot  = botList[difficulty]
    isLong,isBig = evaluate_cards(card_ids)
    
    if isLong and isBig:
        maxBidding = bot.minimumBidWhenBigAndLong
    elif not isBig  and isLong:
        maxBidding = bot.minimumBidWhenSmallAndLong
    elif not isBig and not isLong :
        maxBidding = bot.minimumBidWhenSmallAndShort
    else:
        maxBidding  = bot.minimumBidWhenBigAndShort
    
    rand = random.random()

    if rand > bot.probWhenCurrentBiddingExceedMinumum and current_bid > maxBidding:
        return 0
    else:
        return current_bid + 5
    

# func pick trump accept list of  card id in hand and bot difficulty
# return suite name
def calculateTrumpsuite(card_ids,difficulty):
    bot  = botList[difficulty]
    isLong  = False
    
    trumpIndex= 0
    trumpSuite = ['Hearts','Diamonds','Clubs','Spades']
    for i in range(0,52,13):
        suite = list(filter(lambda x : i<=x<=i+12,card_ids)) 
        if len(suite)>=6:
            isLong  = True
            trumpIndex = i / 13

        suite = list(filter(lambda x : i+9<=x<=i+12,card_ids))     
        if len(suite)>=3:
            if not isLong:
                trumpIndex = i / 13  
    rand = random.random()
    if rand < bot.probSelectBestTrump:
        return trumpSuite[trumpIndex]
    else:
        return trumpSuite[0]
    
# func pick friend card accept list of card id in hand and bot diffuculty
# return card id
def calculateFriendCard(card_ids):
    
    aceCardsID = [12,25,38,51]
    allCardsID = [ i for i in range(52)]
    allCardsIDNotInHand = list(filter(lambda x : x not in card_ids,allCardsID)) 
    isLong  = False
    
    trumpIndex= 0
    #trumpSuite = ['Hearts','Diamonds','Clubs','Spades']
    for i in range(0,52,13):
        suite = list(filter(lambda x : i<=x<=i+12,card_ids)) 
        if len(suite)>=6:
            isLong  = True
            trumpIndex = i / 13

        suite = list(filter(lambda x : i+9<=x<=i+12,card_ids))     
        if len(suite)>=3:
            if not isLong:
                trumpIndex = i / 13
    
    friendID  = (trumpIndex * 13) + 12

    if friendID not in card_ids:
        return friendID
    
    for i in range(len(aceCardsID)):
        if aceCardsID[i] not in card_ids:
            return aceCardsID
    return allCardsIDNotInHand[0]


def mapOutPutToCard(output):
    arr = np.array(output)
    arr = arr.reshape(4,7)
    indices = np.where(arr == 1)
    suites = ['Hearts','Diamonds','Clubs','Spades']
    suiteIndex = indices[0][0]
    pointIndex = indices[1][0]
    points = [0,5,10,'J','Q','K','A']
    smallPoints = [2,3,4,6,7,8,9]
    suiteOutput = suites[suiteIndex]
    pointOutput = points[pointIndex]
    
    if(pointOutput ==0):
        cardOutput = [CARD_DICT[suiteOutput][point] for point in smallPoints]
    else:
        cardOutput = [CARD_DICT[suiteOutput][pointOutput]]
    return cardOutput

# func drop card accept list of card id in hand  list of card id in field ,list of state,bot difficulty 
# return card id
def dropCard(card_ids,state,difficulty):
    bot  = botList[difficulty]
    agent = teamAgent()
    agent.loadModel(f'{bot.fileName}_.pth')
    best_act = agent.get_best_action(state)
    card_to_play = mapOutPutToCard(best_act)
    for i in range(len(card_to_play)) :
        if card_to_play[i] in card_ids:
            return card_to_play[i]
    
if function_name == 'bidding':
   
    card_ids = sys.argv[2]
    card_ids = ast.literal_eval(card_ids)
    
    current_bid = sys.argv[3]
    current_bid = ast.literal_eval(current_bid)
   
    difficulty = sys.argv[4]
    difficulty =  ast.literal_eval(difficulty)

    bidding_score = bidding(card_ids,current_bid,difficulty)
    print(bidding_score)


elif function_name == 'calculateTrumpsuite':
    card_ids = sys.argv[2]
    card_ids = ast.literal_eval(card_ids)
    
    difficulty = sys.argv[3]
    difficulty =  ast.literal_eval(difficulty)

    trump_suite = calculateTrumpsuite(card_ids,difficulty)
    print(trump_suite)

elif function_name == 'calculateFriendCard':

    card_ids = sys.argv[2]
    card_ids = ast.literal_eval(card_ids)
    
    card_id = calculateFriendCard(card_ids)
    print(card_id)

elif function_name == 'dropCard':
    card_ids = sys.argv[2]
    card_ids = ast.literal_eval(card_ids)
    
    state = sys.argv[3]
    state = ast.literal_eval(state)
   
    difficulty = sys.argv[4]
    difficulty =  ast.literal_eval(difficulty)

    card_id = dropCard(card_ids,state,difficulty)
    print(card_id)


    
    


