
"""
1. Fetch data
2. Fetch technical indictors
3. 


"""
from utils.fetch_stock_data import fetch_stock_data
import pandas_ta as pa
# from xgboost import XGBClassifier
# from sklearn.metrics import accuracy_score, log_loss

def read_data():
  stock_name = 'NVDA'
  interval = '30min'
  start_date = ''
  end_date = ''
  data = fetch_stock_data(stock_name, interval, start_date, end_date)

  return data

def get_indicators_data(df):
  df["RSI"] = pa.rsi(df.close, length=16)
  df["CCI"] = pa.cci(df.high, df.low, df.close, length=16)
  df["AO"] = pa.ao(df.high, df.low)
  df["MOM"] = pa.mom(df.close, length=16)
  a = pa.macd(df.close)
  df = df.join(a)


  df.dropna(inplace=True)
  df.reset_index(drop=True, inplace=True)
  return df

pipdiff = 200*1e-4 #for TP
SLTPRatio = 2 #pipdiff/Ratio gives SL

def mytarget(barsupfront, df1):
    length = len(df1)
    high = list(df1['high'])
    low = list(df1['low'])
    close = list(df1['close'])
    open = list(df1['open'])
    trendcat = [None] * length
    for line in range (0,length-barsupfront-2):
        valueOpenLow = 0
        valueOpenHigh = 0
        for i in range(1,barsupfront+2):
            value1 = open[line+1]-low[line+i]
            value2 = open[line+1]-high[line+i]
            valueOpenLow = max(value1, valueOpenLow)
            valueOpenHigh = min(value2, valueOpenHigh)

            if ( (valueOpenLow >= pipdiff) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
                trendcat[line] = 1 #-1 downtrend
                break
            elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= pipdiff) ):
                trendcat[line] = 2 # uptrend
                break
            else:
                trendcat[line] = 0 # no clear trend
            
    return trendcat

def main():
  df = read_data()
  df = get_indicators_data(df)
  df['Target'] = mytarget(20, df)
  df['Target'].hist()
  df.dropna(inplace=True)
  df.reset_index(drop=True, inplace=True)

if __name__ == '__main__':
  main()
