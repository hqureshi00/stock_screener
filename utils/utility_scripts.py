import pandas as pd
import pdb

def test_for_consecutive_buy_signals(data):
    df = pd.DataFrame(data)
    df['Prev_Buy_Sell'] = df['Buy_Sell'].shift(1)
    df['Consecutive_1s'] = (df['Buy_Sell'] == 1) & (df['Prev_Buy_Sell'] == 1)
    consecutive_1s_indices = df.index[df['Consecutive_1s']].tolist()
    pdb.set_trace()