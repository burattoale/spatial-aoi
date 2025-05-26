from typing import List, Any # Changed str|int to Any for broader Python versions
from dataclasses import dataclass

@dataclass
class SimulationParameters(object):
    """
    Keep organized all the parameters of the simulation
    """
    q: float         # DTMC parameter
    eta: float       # DTMC parameter
    zeta: float|List[float]      # Node transmission attempt rate
    epsilon: float   # Node transmission erasure probability
    m_override: int = None # If None, m is calculated from rho. Otherwise, use this value for number of nodes.
    rho: float = 1.0   # Node density (used if m_override is None)
    K: int = 5         # Number of regions/buckets
    alpha: float = 2.0 # Power law exponent for node correctness & entropy calc
    beta: float = 2.0 # Power law exponent for node transmission probability
    R_unit: float = 1.0  # Unit radius for circle sections (was R in original, renamed for clarity)
    
    # Default X_symbols and Y_symbols for binary case
    X_symbols: List[Any] = None # Will be [0, 1]
    Y_symbols: List[Any] = None # Will be [0, 1]

    def __post_init__(self):
        if self.X_symbols is None:
            self.X_symbols = [0, 1]
        if self.Y_symbols is None:
            self.Y_symbols = [0, 1]