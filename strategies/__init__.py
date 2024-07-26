# strategies/__init__.py

from .ma_crossover import crossover_signal, crossover_signal_with_slope
from .ema import ema_strategy
from .macd import generate_macd_signals
from .rsi import generate_rsi_signals
from .candlestick_patterns import generate_candlestick_signals
from .bollinger_bands import generate_bb_signals

__all__ = ['crossover_signal',
           'crossover_signal_with_slope',
           'ema_strategy',
           'generate_macd_signals',
           'generate_rsi_signals',
           'generate_candlestick_signals',
           'generate_bb_signals'
           ]