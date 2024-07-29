Today's Tasks
1. Finalize MACD 
2. Finalize RSI 
3. Optimization

####Backlog
1. Go through the SMA strategy, see how it works, and check if the slope functionality works + the removal of weekend time functionality - done
2. Go through the EMA strategy, check if the plot function works alright
3. Go through the MACD strategy, check if it works, and create a plot function
4. Go through the RSI strategy, check if it works, and create a plot function
5. Go through the BB strategy, check if it works, and create a plot funciton
6. Go through the candlesticks strategy, check if it works and create a plot function
7. Optimize the 1. EMA 2. MACD 3. RSI
8. Run a mega function that for a stock_name, start_date, end_date gives the interval it should run at, and the strategy it should use (or maybe just a table) and the parameters those strategies should be used with (that means that I have to optimize the parameters for some, and for some I just need to go normally like BB and candlestick patterns)
9. Final verification and cross-validation processes with TradingView; also how to benchmark and evaluate thoroughly

##############################

What are the overall end-goals this week + nextweek:
1. be able to run strategies with a specific parameter if we so want
###TODO: we should be able to run them separately
2. be able to optimize for MACD, EMA and SMA since they utilize short term and long term moving averages 
3. being able to optimize on the basis of intervals for other strategies
4. Show graphs and tables to be able to analyse final trades 

Features:
1. Run strategies with all parameters and see trades in a graph and table format
2. Optimize for strategies and list down a table for singal as well as multiple strategies comparison
3. Allow us to make decisions for trades 


##############################

###TODOS

1. Run and Optimize Moving CrossOver Strategy
2. Run EMA Strategy and optimize it using parameters
3. Run MACD Strategy and optimize it using parameters
4. Create a function that compares all these three strategies with optimized parameters, and publishes strategies that have max equity returns 

Goal Outputs:

1. A file containing optimized parameters for MA CrossOver
2. A file containing optimized parameters for EMA
3. A file containing optimized parameters for MACD 
4. A file containing best to worst performing strategies for a particular time_period



Extra TODOs:
1. make sure that you have some sort of a GUI to enter all the details so I don't have to add it at different places
2. add error checking
3. add testing 