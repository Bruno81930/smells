"""
Expose public exceptions & warnings
"""

class WrongMetricValue(ValueError):
    """
    Error raised when a wrong metric value is 
    passed as input for the cross-project model.
    """
    pass