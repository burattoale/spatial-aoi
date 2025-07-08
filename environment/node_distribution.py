# node_distribution.py
import numpy as np
import math
from typing import List, Tuple

# Assuming node.py is in the same directory or accessible as a module
# For direct execution, use: from node import Node
# For package structure, use: from .node import Node
try:
    from .node import Node # If part of a package
except ImportError:
    from node import Node # If running as standalone script with node.py in same dir


class NodeDistribution(object):
    """
    Contains the positions and the total number of nodes in a circular radius
    given the density.
    """

    def __init__(self, rho: float, unit_radius: float, K: int, zeta:float|np.ndarray, alpha:float, beta:float, seed: int = None, zeta_bucket:bool=False): # Added alpha
        """
        :param rho: The density of the nodes in the circle.
        :type rho: float
        :param unit_radius: The radius of the sections of which the total circle is composed.
        :type unit_radius: float
        :param K: Number of units the circle is composed of.
        :type K: int
        :param alpha: The exponent of the power law for the correct probability.
        :type alpha: float
        :param beta: The exponent of the power law for the transmission probability.
        :type beta: float
        :param seed: the seed for the RNG.
        :type seed: int
        """
        self.generator = np.random.default_rng(seed)
        self.unit_radius = unit_radius
        self.K = K
        self.radius = self.unit_radius * self.K  # total radius of the circle
        
        # create the list of the subdivisions of the radius
        # Note: NodeDistribution._subdivide_radius is defined but not used here.
        # The current bucket creation is fine.
        buckets = []
        for i in range(K):
            # Ensure buckets are (lower, upper) where lower is inclusive, upper is exclusive
            # bisect_right works with left_bounds, so this is okay.
            # For a point exactly on unit_radius * (i+1), it would fall into bucket i+1
            # if bisect_right is used on left_bounds.
            # Let's adjust slightly for clarity: bucket i is [i*ur, (i+1)*ur)
            # A distance d will fall into bucket_idx such that:
            # bucket_idx * ur <= d < (bucket_idx+1) * ur
            buckets.append((self.unit_radius * i, self.unit_radius * (i + 1)))
        
        # Area of circle is pi * r^2. Number of nodes = density * area.
        self.m = math.floor(rho * np.pi * (self.radius**2)) # Corrected Area
        
        sampled_points = [self._sample_circle() for _ in range(self.m)]
        # Pass alpha to Node constructor
        if isinstance(zeta, float):
            self._nodes = [Node(point, self.unit_radius, buckets, zeta=zeta, alpha=alpha, beta=beta) for point in sampled_points]
            self._tx_probabilities = [n.zeta for n in self.nodes]
            self._tx_prob_bucket = np.array([zeta] * K)
        elif isinstance(zeta, np.ndarray) and not zeta_bucket:
            self._nodes = [Node(point, self.unit_radius, buckets, zeta=z, alpha=alpha, beta=beta) for z, point in zip(zeta,sampled_points)]
            self._tx_probabilities = zeta.tolist()
            self._tx_prob_bucket = np.array([zeta] * K)
        elif isinstance(zeta, np.ndarray) and zeta_bucket:
            self._nodes = [Node(point, self.unit_radius, buckets, zeta=zeta, alpha=alpha, beta=beta) for point in sampled_points]
            for n in self._nodes:
                n.tx_probability = zeta[n.zone_idx]
            self._tx_probabilities = [n.zeta for n in self.nodes]
            self._tx_prob_bucket = zeta
        else:
            raise NotImplementedError("The combination of parameter is not allowed by the current implementation")


    @property
    def nodes(self):
        return self._nodes
    
    @property
    def tx_probabilities(self):
        return self._tx_probabilities
    
    @property
    def tx_prob_bucket(self):
        return self._tx_prob_bucket

    def __len__(self):
        return self.m
    
    def __getitem__(self, index):
        return self._nodes[index]

    def _sample_circle(self, coordinates: str = "radial") -> Tuple[float, float]:
        """
        Sample uniformly in a circle.

        :param coordinates: The coordinate system to be used. Options are radial or cartesian.
        :type coordinates: str
        
        :returns: A tuple containing the radius and angle or the tuple (x, y) if coordinates is "cartesian".
        :rtype: Tuple[float, float]

        :meta private:
        """
        # To sample uniformly in a circle, sample r^2 uniformly, then sqrt for r.
        r_squared = self.generator.uniform(0, self.radius**2)
        r = np.sqrt(r_squared)
        # r = np.sqrt(self.generator.random()) * self.radius # This is correct already
        theta = self.generator.uniform(0, 2 * np.pi) # Correct: self.generator.random() is uniform [0,1)

        if coordinates == "cartesian":
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return (x, y)
        
        return (r, theta)

    def _subdivide_radius(self, radius: float, K: int) -> List[Tuple[float, float]]:
        """
        Calculate the list of ranges for the radius in which nodes fall into
        (This method is currently not used by __init__)
        :param radius: The length of the circle radius.
        :type radius: float
        :param K: Number of units the circle is composed of.
        :type K: int

        :returns: The list with the extremes for the buckets.
        :rtype: List[Tuple[float,float]]
        """
        range_length = radius / K
        sub_ranges = []
        for i in range(K):
            sub_start = i * range_length
            sub_end = sub_start + range_length
            sub_ranges.append((sub_start, sub_end))
        return sub_ranges

    def __str__(self):
        out = ""
        for n_idx, n_obj in enumerate(self.nodes):
            out += f"Node {n_idx}: " + str(n_obj) + "\n\n"
        return out