import numpy as np
import bisect
from typing import List, Tuple

class Node(object):
    """
    Represents the transmission node with its position, distance to center,
    and probability of having a correct transmission.
    """
    def __init__(self, position:Tuple[float, float], unit_radius:float, buckets:List[Tuple[float,float]], alpha:float, beta:float, coordinates:str="radial"):
        """
        :param position: The position of the node
        :type position: Tuple[float, float]
        :param unit_radius: The radius of the sections of which the total circle is composed.
        :type unit_radius: float
        :param buckets: List of the ranges for the radius.
        :type buckets: List[Tuple[float,float]]
        :param alpha: The exponent of the power law for the correct probability.
        :type alpha: float
        :param beta: The exponent of the power law for the transmission probability.
        :type beta: float
        :param coordinates: Specify the type of coordinates used for position. Can be radial or cartesian.
        :type coordinates: str
        """
        self._position = position
        self.coordinates = coordinates
        self.unit_radius = unit_radius
        self.distance = self._compute_distance()
        self.d_idx = self._get_bucket_idx(buckets)
        self.correct_probability = self._compute_correct_prob(alpha)
        self.tx_probability = self._compute_tx_prob(beta)

    @property
    def lam(self):
        return self.correct_probability
    
    @property
    def position(self):
        return self._position
    
    @property
    def zeta(self):
        return self.tx_probability

    def _compute_distance(self) -> float:
        """
        Compute the distance of the node from the center

        :returns: The distance from the center.
        :rtype: float
        """
        if self.coordinates == "cartesian":
            x, y = self._position
            return np.sqrt(x**2 + y**2)
        
        r, _ = self._position
        return r
    
    def _compute_correct_prob(self, alpha:float) -> float:
        """
        Compute the probability that the information form this node is correct.

        :param alpha: The exponent of the power law for the probability.
        :type alpha: float

        :returns: The probability of the corresponding bucket.
        :rtype: float
        """
        return (1 + self.d_idx * self.unit_radius)**(-alpha)
    
    def _compute_tx_prob(self, beta:float) -> float:
        """
        Compute the probability that the information form this node is correct.

        :param alpha: The exponent of the power law for the probability.
        :type alpha: float

        :returns: The transmission probabilitz of the node.
        :rtype: float
        """
        return (1 + self.d_idx * self.unit_radius)**(-beta)
    
    def _get_bucket_idx(self, bucket_list:List[Tuple[float,float]]) -> int:
        """
        Compute the index of the bucket in which to insert the distance of the node.

        :param bucket_list: List of the ranges for the radius.
        :type bucket_list: List[Tuple[float,float]]

        :returns: The index of the bucket for the node.
        :rtype: int
        """
        left_bounds = [b[0] for b in bucket_list]
        return bisect.bisect_right(left_bounds, self.distance)-1


    def __str__(self):
        return f"Node position: {self.position} \nNode distance: {self.distance} \nBucket index: {self.d_idx} \nTx probability{self.zeta}"