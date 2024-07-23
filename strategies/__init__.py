# strategies/__init__.py

from .ma_crossover import crossover_signal, crossover_signal_with_slope

__all__ = ['crossover_signal',
           'crossover_signal_with_slope'
           ]