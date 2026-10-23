"""
Name: Chengjun Lu, Victoria Liu, Yudi Wu, Eve Zhang
Date: 10/22/2026
"""

class InletProfile:
    """
    Base class for the drug concentration entering the vessel over time.
    """
    def __call__(self, t):
        """
        Return the inlet concentration at time t. 

        Args:
            t (float): time in seconds
        """
        raise NotImplementedError
    
    @staticmethod
    def from_parameters(p):
        """
        Build the inlet profile

        Args:
            p (VesselParameters): Parameters cointaining inlet_kind, inlet_concentration_mg_L and bolus_duration_s. 
        
        Returns:
        InletProfile: A ConstantInlet or BolusInlet object.
        """
        c0 = p.inlet_concentration_mg_L
        if p.inlet_kind == "constant":
            return ConstantInlet(c0)
        elif p.inlet_kind == "bolus":
            return BolusInlet(c0, p.bolus_duration_s)
        else:
            raise ValueError

class ConstantInlet(InletProfile):
    """
    Inlet that always delivers the same concentrtaion. 
    """
    def __init__(self, c0):
        self.c0 = c0

    def __call__(self, t):
        """
        Return the inlet concentration at time t. 

        Args: 
            t (float): time in seconds
        """
        return self.c0

class BolusInlet(InletProfile):
    """
    Inlet that delivers drug for a short time, and then stops. 
    """
    def __init__(self, c0, duration):
        self.c0 = c0
        self.duration = duration
        
    def __call__(self, t):
        """
        Return the inlet concentration at time t.
        """
        if t < self.duration:
            return self.c0
        else:
            return 0.0