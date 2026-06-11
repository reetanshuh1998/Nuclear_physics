import numpy as np
from scipy.optimize import minimize

# Physical constants
HBAR_C = 197.327  # MeV fm
M_N = 939.565     # Nucleon mass (MeV/c^2)

class Nucleon:
    """Represents a single-particle nucleon wave packet in AMD."""
    def __init__(self, Z, spin, isospin):
        """
        Z: 3-vector of complex coordinates: Z_k = sqrt(nu)*R_k + i*P_k / (2*hbar*sqrt(nu))
        spin: float, +0.5 (up) or -0.5 (down)
        isospin: float, +0.5 (proton) or -0.5 (neutron)
        """
        self.Z = np.array(Z, dtype=complex)
        self.spin = float(spin)
        self.isospin = float(isospin)

class AMDState:
    """Represents a Slater determinant wave function for A nucleons in AMD."""
    def __init__(self, nucleons, nu=0.16):
        """
        nucleons: list of Nucleon objects
        nu: width parameter of Gaussian wave packet (fm^-2)
        """
        self.nucleons = nucleons
        self.A = len(nucleons)
        self.nu = nu
        self.update_matrices()

    def update_matrices(self):
        """Precompute the overlap matrix B and its inverse."""
        self.B = np.zeros((self.A, self.A), dtype=complex)
        for i in range(self.A):
            ni = self.nucleons[i]
            for j in range(self.A):
                nj = self.nucleons[j]
                # Spin and isospin delta functions
                if ni.spin == nj.spin and ni.isospin == nj.isospin:
                    # Overlap: exp( -0.5 * (Zi* - Zj)^2 )
                    diff_sq = np.sum((np.conj(ni.Z) - nj.Z)**2)
                    self.B[i, j] = np.exp(-0.5 * diff_sq)
                else:
                    self.B[i, j] = 0.0
        
        self.norm = np.linalg.det(self.B)
        if np.abs(self.norm) > 1e-12:
            self.B_inv = np.linalg.inv(self.B)
        else:
            self.B_inv = np.zeros_like(self.B)

    def kinetic_energy(self):
        """Compute the expectation value of kinetic energy: <T>."""
        if np.abs(self.norm) < 1e-12:
            return 0.0
        
        T_matrix = np.zeros((self.A, self.A), dtype=complex)
        factor = (HBAR_C**2) * self.nu / (2.0 * M_N)
        
        for j in range(self.A):
            nj = self.nucleons[j]
            for i in range(self.A):
                ni = self.nucleons[i]
                if ni.spin == nj.spin and ni.isospin == nj.isospin:
                    diff_sq = np.sum((np.conj(nj.Z) - ni.Z)**2)
                    # T_ji = factor * (3 - (Zj* - Zi)^2) * B_ji
                    T_matrix[j, i] = factor * (3.0 - diff_sq) * self.B[j, i]
                    
        # Total expectation value: Sum_{i,j} T_ji * (B^-1)_ij
        T_expect = np.sum(T_matrix * self.B_inv.T)
        return np.real(T_expect)

    def potential_energy(self, V0=-60.0, mu_pot=1.5):
        """
        Compute the expectation value of potential energy: <V>
        using a central Gaussian potential V(r) = V0 * exp(-r^2 / mu_pot^2),
        vectorized for performance using tensor operations.
        """
        if np.abs(self.norm) < 1e-12 or self.A < 2:
            return 0.0
        
        pot_factor = (self.nu * mu_pot**2 / (1.0 + self.nu * mu_pot**2))**1.5
        prefactor = V0 * pot_factor
        gamma = 1.0 / (1.0 + self.nu * mu_pot**2)
        
        # Extract nucleon properties
        Z = np.array([n.Z for n in self.nucleons], dtype=complex)
        spin = np.array([n.spin for n in self.nucleons])
        isospin = np.array([n.isospin for n in self.nucleons])
        
        # Pairwise spin/isospin match matrix
        S_ik = (spin[:, None] == spin[None, :]) & (isospin[:, None] == isospin[None, :])
        
        # Coordinate matrices
        Z_conj = np.conj(Z)
        diff_matrix = np.sum((Z_conj[:, None, :] - Z[None, :, :])**2, axis=-1)
        R_matrix = 0.5 * (Z_conj[:, None, :] + Z[None, :, :])
        
        # Direct term matching and calculation
        # match_direct is shape (A, A, A, A) with indices (i, j, k, l)
        match_direct = S_ik[:, None, :, None] & S_ik[None, :, None, :]
        diff_ik = diff_matrix[:, None, :, None]
        diff_jl = diff_matrix[None, :, None, :]
        diff_R = np.sum((R_matrix[:, None, :, None, :] - R_matrix[None, :, None, :, :])**2, axis=-1)
        
        val_direct = np.zeros((self.A, self.A, self.A, self.A), dtype=complex)
        exponent_dir = -0.5 * diff_ik - 0.5 * diff_jl - gamma * diff_R
        val_direct[match_direct] = prefactor * np.exp(exponent_dir[match_direct])
        
        # Exchange term matching and calculation
        match_exchange = S_ik[:, None, None, :] & S_ik[None, :, :, None]
        diff_il = diff_matrix[:, None, None, :]
        diff_jk = diff_matrix[None, :, :, None]
        diff_R_exch = np.sum((R_matrix[:, None, None, :, :] - R_matrix[None, :, :, None, :])**2, axis=-1)
        
        val_exchange = np.zeros((self.A, self.A, self.A, self.A), dtype=complex)
        exponent_exch = -0.5 * diff_il - 0.5 * diff_jk - gamma * diff_R_exch
        val_exchange[match_exchange] = prefactor * np.exp(exponent_exch[match_exchange])
        
        element = val_direct - val_exchange
        
        # Expectation value sum: 0.5 * sum_{i,j,k,l} B_inv[k,i] * B_inv[l,j] * element[i,j,k,l]
        V_expect = 0.5 * np.einsum('ki,lj,ijkl->', self.B_inv, self.B_inv, element)
        
        return np.real(V_expect)

    def total_energy(self):
        """Total energy expectation value: E = <T> + <V>."""
        return self.kinetic_energy() + self.potential_energy()

    def extract_overlap_amplitude(self, core_state, r_grid, spin_val=0.5, isospin_val=-0.5):
        """
        Extract the valence nucleon overlap amplitude (l=0) on r_grid.
        y_l=0(r) = sqrt(A) * <core | projectile>_integrated_over_angles.
        """
        # project core (A-1) and projectile (A) states
        # The overlap when projectile nucleon p is evaluated at r:
        # y_0(r) = sqrt(A) * Sum_{p=1}^A (-1)^(A+p) * det(B^(p)) * phi_p(r)_l=0
        # where B^(p) is the (A-1)x(A-1) overlap matrix between core and projectile (with p removed)
        u_vals = np.zeros_like(r_grid)
        
        # Build core-projectile overlap matrix B_cp of size (A-1) x A
        B_cp = np.zeros((core_state.A, self.A), dtype=complex)
        for i in range(core_state.A):
            ni = core_state.nucleons[i]
            for j in range(self.A):
                nj = self.nucleons[j]
                if ni.spin == nj.spin and ni.isospin == nj.isospin:
                    diff_sq = np.sum((np.conj(ni.Z) - nj.Z)**2)
                    B_cp[i, j] = np.exp(-0.5 * diff_sq)
        
        # Precompute det(B^(p)) for each p
        det_Bp = np.zeros(self.A, dtype=complex)
        for p in range(self.A):
            # remove column p from B_cp
            B_sub = np.delete(B_cp, p, axis=1)
            det_Bp[p] = np.linalg.det(B_sub)
            
        pre_factor = np.sqrt(self.A) * (2.0 * self.nu / np.pi)**0.75
        
        for idx, r in enumerate(r_grid):
            val = 0.0 + 0j
            for p in range(self.A):
                np_nuc = self.nucleons[p]
                # Valence nucleon spin/isospin match check
                if np_nuc.spin == spin_val and np_nuc.isospin == isospin_val:
                    # Analytical angular average of exp( -nu * (r - Zp)^2 )
                    # Zp_mag = sqrt( Zpx^2 + Zpy^2 + Zpz^2 )
                    Zp_sq = np.sum(np_nuc.Z**2)
                    Zp = np.sqrt(Zp_sq)
                    
                    # exp( -nu*r^2 - Zp^2 ) * sinh(2*sqrt(nu)*r*Zp) / (2*sqrt(nu)*r*Zp)
                    exp_term = np.exp(-self.nu * r**2 - Zp_sq)
                    
                    # Handle Zp -> 0 safely
                    arg = 2.0 * np.sqrt(self.nu) * r * Zp
                    if np.abs(arg) < 1e-6:
                        sinh_term = 1.0 + (arg**2) / 6.0
                    else:
                        sinh_term = np.sinh(arg) / arg
                        
                    phi_l0 = pre_factor * exp_term * sinh_term
                    
                    sign = (-1)**(self.A + p + 1)
                    val += sign * det_Bp[p] * phi_l0
            
            # The overlap radial amplitude u(r) = r * y_0(r) * sqrt(4*pi)
            u_vals[idx] = np.real(val) * r * np.sqrt(4.0 * np.pi)
            
        return u_vals

def minimize_energy_state(initial_Z_coords, spins, isospins, nu=0.16, max_iter=100):
    """
    Find the energy-optimized configuration of Z-coordinates for an AMD state.
    """
    A = len(spins)
    
    def objective(coords_flat):
        # coords_flat has 6*A elements (real and imaginary parts for x,y,z of each nucleon)
        Z_flat = coords_flat[:3*A] + 1j * coords_flat[3*A:]
        nucleons = []
        for i in range(A):
            Z = Z_flat[3*i : 3*i+3]
            nucleons.append(Nucleon(Z, spins[i], isospins[i]))
        state = AMDState(nucleons, nu=nu)
        return state.total_energy()
        
    coords_init = np.concatenate([np.real(initial_Z_coords).flatten(), np.imag(initial_Z_coords).flatten()])
    res = minimize(objective, coords_init, method='BFGS', options={'maxiter': max_iter})
    
    Z_opt_flat = res.x[:3*A] + 1j * res.x[3*A:]
    nucleons_opt = []
    for i in range(A):
        Z = Z_opt_flat[3*i : 3*i+3]
        nucleons_opt.append(Nucleon(Z, spins[i], isospins[i]))
        
    return AMDState(nucleons_opt, nu=nu), res.fun
