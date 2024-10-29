#Training
MODEL_THRESHOLD = 0.6
GAIN_CLASS_THRESHOLD = 0.03
CLASS_RANGES = [
        (-50/100, -0.3/100, 0), 
        (-0.3/100, -0.2/100, 1),  
        (-0.2/100, 0, 2),
        (0, GAIN_CLASS_THRESHOLD/100, 3),    
        (GAIN_CLASS_THRESHOLD/100, 0.2/100, 4),   
        (0.2/100, 50/100, 5),    
      ]


#Backtesting
TRADE_LEN = 7 
TRANSACTION_COST_PER_STOCK = 0.005
BACKTESTING_THRESHOLD = 0.6

