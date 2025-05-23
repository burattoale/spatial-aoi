from typing import List
from dataclasses import dataclass

@dataclass
class SimulationParameters(object):
    """
    Keep organized all the parameters of the simulation
    """
    q:float
    eta:float
    zeta:float
    epsilon:float
    m:int
    K:int
    alpha:float = 0.02
    R:float = 10
    X_symbols:List[str|int] = None
    Y_symbols:List[str|int] = None