import numpy as np

class Boundary():
    boundary_ids: np.ndarray[int]
    boundary_type: np.ndarray[int]
    groups: dict[str,np.ndarray]
    num_outputs: int

    def __init__(self,num_outputs,groups) -> None:
        self.num_outputs = num_outputs
        self.groups = groups



