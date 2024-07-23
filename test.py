import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

def on_submit():
    stock_name = stock_name_entry.get()
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    interval = interval_entry.get()
    strategy_name = strategy_name_entry.get()
    
    # Example of what you might do with these inputs
    # Here we'll just print them
    print(f"Stock Name: {stock_name}")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Interval: {interval}")
    print(f"Strategy Name: {strategy_name}")

    # You can replace the print statements with calls to your actual functions
    # For example:
    # data = fetch_data(stock_name, start_date, end_date, interval)
    # strategy = get_strategy(strategy_name)
    # result = simulate_trading(strategy, data)
    # print(result)

# Initialize the main window
root = tk.Tk()
root.title("Stock Trading Simulator")

# Create labels and entry widgets for each argument
ttk.Label(root, text="Stock Name:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
stock_name_entry = ttk.Entry(root)
stock_name_entry.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(root, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
start_date_entry = ttk.Entry(root)
start_date_entry.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(root, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
end_date_entry = ttk.Entry(root)
end_date_entry.grid(row=2, column=1, padx=10, pady=5)

ttk.Label(root, text="Interval:").grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)
interval_entry = ttk.Entry(root)
interval_entry.grid(row=3, column=1, padx=10, pady=5)

ttk.Label(root, text="Strategy Name:").grid(row=4, column=0, padx=10, pady=5, sticky=tk.W)
strategy_name_entry = ttk.Entry(root)
strategy_name_entry.grid(row=4, column=1, padx=10, pady=5)

# Create a submit button
submit_button = ttk.Button(root, text="Submit", command=on_submit)
submit_button.grid(row=5, columnspan=2, pady=10)

# Run the application
root.mainloop()
