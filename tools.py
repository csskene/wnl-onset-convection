"""
Functions for accessing and manipulating adjoint eigenvectors
"""
import numpy as np

def set_state_adjoint(self, index, subsystem):
    """
    Set state vector to the specified eigenmode.
    Parameters
    ----------
    index : int
        Index of desired eigenmode.
    subsystem : Subsystem object or int
        Subsystem that will be set to the corresponding eigenmode.
        If an integer, the corresponding subsystem of the last specified
        eigenvalue_subproblem will be used.
    """
    # TODO: allow setting left modified eigenvectors?
    subproblem = self.eigenvalue_subproblem
    if isinstance(subsystem, int):
        subsystem = subproblem.subsystems[subsystem]
    # Check selection
    if subsystem not in subproblem.subsystems:
        raise ValueError("subsystem must be in eigenvalue_subproblem")
    # Set coefficients
    for var in self.state:
        var['c'] = 0
    subsystem.scatter(self.modified_left_eigenvectors[:, index], self.state)

def _build_modified_left_eigenvectors(self):
    sp = self.eigenvalue_subproblem
    return (sp.M_min @ sp.pre_right_pinv).H@sp.pre_left@self.left_eigenvectors

def _normalize_left_eigenvectors(self):
    modified_left_eigenvectors = _build_modified_left_eigenvectors(self)
    norms = np.diag(modified_left_eigenvectors.T.conj() @ self.eigenvectors)
    self.left_eigenvectors /= np.conj(norms)
    self.modified_left_eigenvectors /= np.conj(norms)

