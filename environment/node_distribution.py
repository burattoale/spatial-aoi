import numpy as np
import math
from typing import List, Tuple

from .node import Node

class NodeDistribution(object):
    """
    Contains the positions and the total number of nodes in a circular radius
    given the density.
    """

    def __init__(self, rho:float, unit_radius:float, K:int, seed:int=None):
        """
        :param rho: The density of the nodes in the circle.
        :type rho: float
        :param unit_radius: The radius of the sections of which the total circle is composed.
        :type unit_radius: float
        :param K: Number of units the circle is composed of.
        :type K: int
        :param seed: the seed for the RNG.
        :type seed: int
        """
        self.generator = np.random.default_rng(seed)
        self.unit_radius = unit_radius
        self.K = K
        self.radius = self.unit_radius * self.K # total radius of the circle
        buckets = []
        # create the list of the subdivisions of the radius
        for i in range(K):
            buckets.append((self.unit_radius * i, self.unit_radius * (i + 1)))
        self.m = math.floor(rho * np.pi * self.radius)
        sampled_points = [self._sample_circle() for _ in range(self.m)]
        self._nodes = [Node(point, self.unit_radius, buckets, alpha=2) for point in sampled_points]

    @property
    def nodes(self):
        return self._nodes
    
    def __len__(self):
        return self.m


    def _sample_circle(self, coordinates:str="radial") -> Tuple[float, float]:
        """
        Sample uniformly in a circle.

        :param coordinates: The coordinate system to be used. Options are radial or cartesian.
        :type coordinates: str
        
        :returns: A tuple containing the radius and angle or the tuple (x, y) if coordinates is "cartesian".
        :rtype: Tuple[float, float]

        :meta private:
        """
        r = np.sqrt(self.generator.random()) * self.radius
        theta = self.generator.random() * 2 * np.pi

        if coordinates == "cartesian":
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            return (x, y)
        
        return (r, theta)
    
    def _subdivide_radius(self, radius:float, K:int) -> List[Tuple[float,float]]:
        """
        Calculate the list of ranges for the radius in which nodes fall into

        :param radius: The length of the circle radius.
        :type radius: float
        :param K: Number of units the circle is composed of.
        :type K: int

        :returns: The list with the extremes for the buckets.
        :rtype: List[Tuple[float,float]]
        """
        # Calculate the length of each sub-range
        range_length = radius / K
    
        # Generate the K equal sub-ranges
        sub_ranges = []
        for i in range(K):
            sub_start = i * range_length
            sub_end = sub_start + range_length
            sub_ranges.append((sub_start, sub_end))
    
        return sub_ranges
    
    def __str__(self):
        out = ""
        for n in self.nodes:
            out += str(n) + "\n\n"
        return out
