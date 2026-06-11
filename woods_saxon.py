import numpy as np
import scipy.special as sp
from scipy.optimize import root_scalar

# Physical constants
HBAR_C = 197.327  # MeV fm
M_N = 939.565     # Neutron mass in MeV/c^2
M_C = 9327.9      # Core 10Be mass in MeV/c^2
MU_DEFAULT = (M_N * M_C) / (M_N + M_C)  # Reduced mass of 10Be + n system

class Potential:
    """Base class for potentials."""
    def evaluate(self, r):
        raise NotImplementedError("Subclasses must implement evaluate(r)")

class WoodsSaxonPotential(Potential):
    """Woods-Saxon potential V(r) = -V0 / (1 + exp((r - R)/a))."""
    def __init__(self, V0, r0=1.15, a=0.50, A_core=10.0):
        self.V0 = V0
        self.r0 = r0
        self.a = a
        self.A_core = A_core
        self.R = r0 * (A_core**(1/3))

    def evaluate(self, r):
        # Prevent overflow in exponential
        arg = (r - self.R) / self.a
        # Use np.where or clip to handle very large values safely
        exp_factor = np.exp(np.clip(arg, -100, 100))
        return -self.V0 / (1.0 + exp_factor)

class CentrifugalPotential(Potential):
    """Centrifugal potential V_cent(r) = hbar^2 * l * (l + 1) / (2 * mu * r^2)."""
    def __init__(self, l, mu=MU_DEFAULT):
        self.l = l
        self.mu = mu

    def evaluate(self, r):
        if self.l == 0:
            return 0.0
        # Prevent division by zero at r=0
        r_safe = np.where(r == 0, 1e-15, r)
        return (HBAR_C**2) * self.l * (self.l + 1) / (2 * self.mu * r_safe**2)

class LagrangeMeshSolver:
    """
    Radial Schrödinger equation solver using the Lagrange-Laguerre mesh method.
    """
    def __init__(self, N=40, h=0.3, mu=MU_DEFAULT):
        self.N = N
        self.h = h
        self.mu = mu
        self.roots, self.weights = sp.roots_laguerre(N)
        
        # Derivatives of Laguerre polynomials at the roots: L'_N(x_i)
        # Using L'_N(x_i) = - N / x_i * L_{N-1}(x_i)
        self.dL = -N * sp.eval_laguerre(N - 1, self.roots) / self.roots
        
        # Regularization normalization factors C_i
        self.C = 1.0 / (np.sqrt(self.weights) * self.roots * self.dL)
        
        # Precompute the unscaled kinetic energy matrix T_ij
        self.T = np.zeros((N, N))
        for i in range(N):
            xi = self.roots[i]
            self.T[i, i] = (4 + (4*N + 2)*xi - xi**2) / (12 * xi**2)
            for j in range(N):
                if i != j:
                    xj = self.roots[j]
                    self.T[i, j] = (-1)**(i-j) * (xi * xj)**(-0.5) * (xi + xj) / (xi - xj)**2

    def solve(self, potential, l=0):
        """
        Diagonalize the Hamiltonian H = T + V for a given potential and orbital angular momentum l.
        Returns eigenvalues (sorted) and the corresponding eigenvectors.
        """
        # Scale kinetic energy: H_T = (hbar_c)^2 / (2 * mu * h^2) * T
        factor = (HBAR_C**2) / (2 * self.mu * self.h**2)
        H = factor * self.T.copy()
        
        # Centrifugal potential
        v_cent = CentrifugalPotential(l, self.mu)
        
        # Add potential energy (diagonal) at the scaled mesh points r_i = h * x_i
        for i in range(self.N):
            r_i = self.h * self.roots[i]
            V_val = potential.evaluate(r_i) + v_cent.evaluate(r_i)
            H[i, i] += V_val
            
        # Diagonalize symmetric matrix
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        return eigenvalues, eigenvectors

    def f_i(self, x, i):
        """Evaluate the i-th regularized Lagrange-Laguerre function at dimensionless coordinate x."""
        xi = self.roots[i]
        Ci = self.C[i]
        if np.abs(x - xi) < 1e-8:
            return Ci * xi * self.dL[i] * np.exp(-0.5 * xi)
        else:
            L = sp.eval_laguerre(self.N, x)
            return Ci * x * L * np.exp(-0.5 * x) / (x - xi)

    def wavefunction(self, eigenvector, r):
        """
        Reconstruct the radial wavefunction u_l(r) at a given radius r (scalar or array)
        from the eigenvector coefficients c_i.
        """
        # Support both scalar and array inputs for r
        r_arr = np.atleast_1d(r)
        u_vals = np.zeros_like(r_arr, dtype=float)
        
        for idx, ri in enumerate(r_arr):
            x = ri / self.h
            val = 0.0
            for i in range(self.N):
                val += eigenvector[i] * self.f_i(x, i)
            u_vals[idx] = val / np.sqrt(self.h)
            
        return u_vals[0] if np.isscalar(r) else u_vals

def adjust_depth_to_binding_energy(solver, target_energy, r0=1.15, a=0.50, A_core=10.0, l=0, initial_guess=70.0):
    """
    Vary the depth V0 of the Woods-Saxon potential to reproduce the target binding energy.
    Note that target_energy should be negative for bound states (e.g., -0.50 MeV).
    For l=0, we target the 2s state (second eigenvalue, index 1).
    """
    def objective(V0):
        pot = WoodsSaxonPotential(V0, r0=r0, a=a, A_core=A_core)
        eigenvalues, _ = solver.solve(pot, l)
        if l == 0:
            # For 11Be ground state, 2s1/2 is the second s-state
            if len(eigenvalues) > 1:
                return eigenvalues[1] - target_energy
            else:
                return eigenvalues[0] - target_energy + 50.0
        else:
            return eigenvalues[0] - target_energy
        
    res = root_scalar(objective, x0=initial_guess, x1=initial_guess * 1.1, method='secant', xtol=1e-6)
    return res.root
