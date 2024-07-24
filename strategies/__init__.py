# strategies/__init__.py

from .ma_crossover import crossover_signal, crossover_signal_with_slope
from .ema import ema_strategy

__all__ = ['crossover_signal',
           'crossover_signal_with_slope',
           'ema_strategy'
           ]