import numpy as np
import scipy.special as sp

# Physical constants
HBAR_C = 197.327  # MeV fm
E2 = 1.43997      # e^2 in MeV fm
M_N = 939.565     # Neutron mass in MeV/c^2
M_C = 9327.9      # Core 10Be mass in MeV/c^2
M_T_PB = 193730.0 # Pb-208 mass in MeV/c^2

def hyp2f1_complex(a, b, c, z, max_terms=200, tol=1e-16):
    """
    Evaluate the confluent hypergeometric function 2F1(a, b; c; z) for complex a, b, z,
    using analytical continuation / linear transformation when |z| >= 0.9.
    """
    if np.abs(z) < 0.9:
        term = 1.0 + 0j
        sum_val = term
        for n in range(1, max_terms):
            term *= (a + n - 1) * (b + n - 1) / ((c + n - 1) * n) * z
            sum_val += term
            if np.abs(term) < tol:
                break
        return sum_val
    else:
        # Linear transformation for |z| >= 0.9 (Abramowitz & Stegun 15.3.7)
        if np.abs(a - b) < 1e-10:
            a = a + 1e-10
        g_c = sp.gamma(c)
        g_b_a = sp.gamma(b - a)
        g_a_b = sp.gamma(a - b)
        g_b = sp.gamma(b)
        g_a = sp.gamma(a)
        g_c_a = sp.gamma(c - a)
        g_c_b = sp.gamma(c - b)
        
        # Choose the complex branch (-z) = |z| * exp(-i*pi) to cancel the Coulomb normalization factor
        if np.real(z) > 0:
            log_neg_z = np.log(np.abs(z)) - 1j * np.pi
        else:
            log_neg_z = np.log(np.abs(z))
            
        neg_z_to_neg_a = np.exp(-a * log_neg_z)
        neg_z_to_neg_b = np.exp(-b * log_neg_z)
        
        term1_coeff = g_c * g_b_a / (g_b * g_c_a) * neg_z_to_neg_a
        term2_coeff = g_c * g_a_b / (g_a * g_c_b) * neg_z_to_neg_b
        
        f1 = hyp2f1_complex(a, 1.0 - c + a, 1.0 - b + a, 1.0 / z, max_terms, tol)
        f2 = hyp2f1_complex(b, 1.0 - c + b, 1.0 - a + b, 1.0 / z, max_terms, tol)
        
        return term1_coeff * f1 + term2_coeff * f2

class FRDWBAReaction:
    def __init__(self, Z_a=4, A_a=11.0, Z_b=4, A_b=10.0, Z_c=0, A_c=1.0, Z_t=82, A_t=208.0, E_beam_per_A=69.0):
        """
        Z_a, A_a: Projectile (e.g. 11Be)
        Z_b, A_b: Core (e.g. 10Be)
        Z_c, A_c: Valence particle (e.g. neutron)
        Z_t, A_t: Target (e.g. 208Pb)
        E_beam_per_A: Beam energy per nucleon in MeV/nucleon
        """
        self.Z_a = Z_a
        self.A_a = A_a
        self.Z_b = Z_b
        self.A_b = A_b
        self.Z_c = Z_c
        self.A_c = A_c
        self.Z_t = Z_t
        self.A_t = A_t
        
        # Masses in MeV/c^2
        self.m_c = A_c * M_N
        self.m_b = A_b * M_N
        self.m_t = A_t * M_N
        self.m_a = self.m_b + self.m_c
        
        # Mass factors
        self.alpha = self.m_c / (self.m_c + self.m_b)
        self.delta_mass = self.m_t / (self.m_b + self.m_t)
        self.gamma = 1.0 - self.alpha * self.delta_mass
        
        # Incident kinematics
        self.E_beam = E_beam_per_A * A_a  # Projectile kinetic energy in Lab (MeV)
        # Projectile velocity and wave number
        self.p_a_lab = np.sqrt(2.0 * self.m_a * self.E_beam)
        self.v_a_lab = self.p_a_lab / self.m_a
        
        # In the c.m. frame of projectile and target
        self.mu_at = (self.m_a * self.m_t) / (self.m_a + self.m_t)
        self.E_cm = self.E_beam * self.m_t / (self.m_a + self.m_t)
        # Projectile wave number in the Lab frame (where core & valence are solved)
        self.q_a_mag = self.p_a_lab / HBAR_C
        
        # Incident wavevector along z-axis
        self.q_a = np.array([0.0, 0.0, self.q_a_mag])
        
        # Incident Coulomb parameter
        # eta_a = Z_a * Z_t * e^2 / (hbar * v_rel)
        # v_rel is relative velocity between projectile and target
        self.eta_a = (Z_a * Z_t * E2) / (HBAR_C * self.v_a_lab)
        
    def solve_final_state(self, E_b_lab, theta_b, phi_b, theta_c, phi_c):
        """
        Solve final state 3-body kinematics for a given core energy in Lab and angles.
        Returns a dictionary of kinematic variables.
        """
        # Core momentum in Lab
        p_b_mag = np.sqrt(2.0 * self.m_b * E_b_lab)
        
        # Core unit vector
        u_b = np.array([
            np.sin(theta_b) * np.cos(phi_b),
            np.sin(theta_b) * np.sin(phi_b),
            np.cos(theta_b)
        ])
        p_b = p_b_mag * u_b
        
        # Valence unit vector
        u_c = np.array([
            np.sin(theta_c) * np.cos(phi_c),
            np.sin(theta_c) * np.sin(phi_c),
            np.cos(theta_c)
        ])
        
        # We solve the quadratic equation for p_c_mag:
        # A_c * p_c^2 + B_c * p_c + C_c = 0
        p_diff = self.p_a_lab * np.array([0, 0, 1]) - p_b
        
        A_c = (self.m_c + self.m_t) / (2.0 * self.m_c * self.m_t)
        B_c = - np.dot(u_c, p_diff) / self.m_t
        # Binding energy / Q-value. Here S_n is one-neutron separation energy (0.50 MeV)
        # E_final = E_lab - S_n = p_b^2 / 2m_b + p_c^2 / 2m_c + p_t^2 / 2m_t
        S_n = 0.50
        C_c = np.dot(p_diff, p_diff) / (2.0 * self.m_t) + (p_b_mag**2) / (2.0 * self.m_b) - self.E_beam + S_n
        
        discriminant = B_c**2 - 4.0 * A_c * C_c
        if discriminant < 0:
            return None  # Kinematically forbidden
            
        p_c_mag = (-B_c + np.sqrt(discriminant)) / (2.0 * A_c)
        if p_c_mag < 0:
            return None
            
        p_c = p_c_mag * u_c
        p_t = p_diff - p_c
        
        # Convert to wavevectors
        q_a = self.q_a
        q_b = p_b / HBAR_C
        q_c = p_c / HBAR_C
        
        # Core-target relative motion in final channel
        # Relative velocity for final state Coulomb parameter eta_b
        mu_bt = (self.m_b * self.m_t) / (self.m_b + self.m_t)
        # Relative momentum of core-target
        p_bt_rel = (self.m_t * p_b - self.m_b * p_t) / (self.m_b + self.m_t)
        v_b_rel = np.linalg.norm(p_bt_rel) / mu_bt
        
        eta_b = (self.Z_b * self.Z_t * E2) / (HBAR_C * v_b_rel)
        
        # Local momentum K of core at R=10 fm
        # E_bt = p_bt_rel^2 / 2mu_bt
        E_bt = np.sum(p_bt_rel**2) / (2.0 * mu_bt)
        R_local = 10.0  # fm
        V_c_local = (self.Z_b * self.Z_t * E2) / R_local
        
        K_val_sq = (2.0 * mu_bt / HBAR_C**2) * (E_bt - V_c_local)
        K_val = np.sqrt(max(0.0, K_val_sq))
        # Direction of K is along q_b
        q_b_mag = np.linalg.norm(q_b)
        K = K_val * (q_b / q_b_mag if q_b_mag > 0 else np.array([0, 0, 1]))
        
        # Momentum transfer in structural factor: k1 = |gamma * q_c - alpha * K|
        k1_vec = self.gamma * q_c - self.alpha * K
        k1 = np.linalg.norm(k1_vec)
        
        # Relative energy between core and valence neutron
        p_rel = (self.m_c * p_b - self.m_b * p_c) / (self.m_b + self.m_c)
        mu_bc = (self.m_b * self.m_c) / (self.m_b + self.m_c)
        E_rel = np.sum(p_rel**2) / (2.0 * mu_bc)
        
        return {
            'q_a': q_a,
            'q_b': q_b,
            'q_c': q_c,
            'eta_b': eta_b,
            'k1': k1,
            'E_rel': E_rel,
            'p_b_mag': p_b_mag,
            'p_c_mag': p_c_mag,
            'p_t': p_t
        }

    def solve_final_state_from_Ec(self, E_c_lab, theta_b, phi_b, theta_c, phi_c):
        """
        Solve final state 3-body kinematics for a given valence energy in Lab and angles.
        Returns a dictionary of kinematic variables.
        """
        # Valence momentum in Lab
        p_c_mag = np.sqrt(2.0 * self.m_c * E_c_lab)
        
        # Valence unit vector
        u_c = np.array([
            np.sin(theta_c) * np.cos(phi_c),
            np.sin(theta_c) * np.sin(phi_c),
            np.cos(theta_c)
        ])
        p_c = p_c_mag * u_c
        
        # Core unit vector
        u_b = np.array([
            np.sin(theta_b) * np.cos(phi_b),
            np.sin(theta_b) * np.sin(phi_b),
            np.cos(theta_b)
        ])
        
        # We solve the quadratic equation for p_b_mag:
        # A_b * p_b^2 + B_b * p_b + C_b = 0
        p_diff = self.p_a_lab * np.array([0, 0, 1]) - p_c
        
        A_b = (self.m_b + self.m_t) / (2.0 * self.m_b * self.m_t)
        B_b = - np.dot(u_b, p_diff) / self.m_t
        S_n = 0.50
        C_b = np.dot(p_diff, p_diff) / (2.0 * self.m_t) + E_c_lab - self.E_beam + S_n
        
        discriminant = B_b**2 - 4.0 * A_b * C_b
        if discriminant < 0:
            return None  # Kinematically forbidden
            
        p_b_mag = (-B_b + np.sqrt(discriminant)) / (2.0 * A_b)
        if p_b_mag < 0:
            return None
            
        p_b = p_b_mag * u_b
        p_t = p_diff - p_b
        
        # Convert to wavevectors
        q_a = self.q_a
        q_b = p_b / HBAR_C
        q_c = p_c / HBAR_C
        
        # Core-target relative motion in final channel
        mu_bt = (self.m_b * self.m_t) / (self.m_b + self.m_t)
        p_bt_rel = (self.m_t * p_b - self.m_b * p_t) / (self.m_b + self.m_t)
        v_b_rel = np.linalg.norm(p_bt_rel) / mu_bt
        
        eta_b = (self.Z_b * self.Z_t * E2) / (HBAR_C * v_b_rel)
        
        # Local momentum K of core at R=10 fm
        E_bt = np.sum(p_bt_rel**2) / (2.0 * mu_bt)
        R_local = 10.0  # fm
        V_c_local = (self.Z_b * self.Z_t * E2) / R_local
        
        K_val_sq = (2.0 * mu_bt / HBAR_C**2) * (E_bt - V_c_local)
        K_val = np.sqrt(max(0.0, K_val_sq))
        q_b_mag = np.linalg.norm(q_b)
        K = K_val * (q_b / q_b_mag if q_b_mag > 0 else np.array([0, 0, 1]))
        
        k1_vec = self.gamma * q_c - self.alpha * K
        k1 = np.linalg.norm(k1_vec)
        
        # Relative energy between core and valence neutron
        p_rel = (self.m_c * p_b - self.m_b * p_c) / (self.m_b + self.m_c)
        mu_bc = (self.m_b * self.m_c) / (self.m_b + self.m_c)
        E_rel = np.sum(p_rel**2) / (2.0 * mu_bc)
        
        return {
            'q_a': q_a,
            'q_b': q_b,
            'q_c': q_c,
            'eta_b': eta_b,
            'k1': k1,
            'E_rel': E_rel,
            'p_b_mag': p_b_mag,
            'p_c_mag': p_c_mag,
            'p_t': p_t
        }

    def bremsstrahlung_integral(self, q_b, q_c, eta_b):
        """
        Evaluate the Bremsstrahlung integral I.
        k = q_a - q_b - delta * q_c
        """
        q_a = self.q_a
        k_vec = q_a - q_b - self.delta_mass * q_c
        k2 = np.sum(k_vec**2)
        k = np.sqrt(k2)
        
        q_a_mag = np.linalg.norm(q_a)
        q_b_mag = np.linalg.norm(q_b)
        
        # Definitions of u1 and u2 at x=0 (positive definite definitions to match the physical branch cut of Nordsieck)
        u1 = 2.0 * np.dot(k_vec, q_a) - k2
        u2 = 2.0 * np.dot(k_vec, q_b) + k2
        
        # Derivatives at x=0
        u1_prime = 2.0 * q_a_mag
        u2_prime = 2.0 * q_b_mag
        
        # B(0) and its derivative B'(0)
        # Power factor: i*eta_a, i*eta_b, etc.
        pow_a = 1j * self.eta_a
        pow_b = 1j * eta_b
        
        # Convert bases to complex to prevent float conversion warnings or negative bases issues
        k2_c = k2 + 0j
        u1_c = u1 + 0j
        u2_c = u2 + 0j
        
        C_0 = 4.0 * np.pi / (k2_c**(pow_a + pow_b + 1.0))
        B_0 = C_0 * (u1_c**pow_a) * (u2_c**pow_b)
        
        # B'(0) = B(0) * ( i*eta_a * u1'/u1 + i*eta_b * u2'/u2 )
        B_prime_0 = B_0 * (pow_a * u1_prime / u1 + pow_b * u2_prime / u2)
        
        # D(0) (which is xi(0)) and its derivative D'(0)
        # N(0) = 2*k^2*(qa*qb + qa.qb) - 4*(k.qa)*(k.qb)
        N_0 = 2.0 * k2 * (q_a_mag * q_b_mag + np.dot(q_a, q_b)) - 4.0 * np.dot(k_vec, q_a) * np.dot(k_vec, q_b)
        # N'(0) = 4*qb*(k.qa) - 4*qa*(k.qb)
        N_prime_0 = 4.0 * q_b_mag * np.dot(k_vec, q_a) - 4.0 * q_a_mag * np.dot(k_vec, q_b)
        
        D_0 = u1 * u2
        D_prime_0 = u1_prime * u2 + u1 * u2_prime
        
        xi_0 = N_0 / D_0
        xi_prime_0 = (N_prime_0 * D_0 - N_0 * D_prime_0) / (D_0**2)
        
        # Hypergeometric functions
        # 1. 2F1(-i*eta_a, -i*eta_b; 1; xi_0)
        F1 = hyp2f1_complex(-pow_a, -pow_b, 1, xi_0)
        # 2. 2F1(1 - i*eta_a, 1 - i*eta_b; 2; xi_0)
        F2 = hyp2f1_complex(1.0 - pow_a, 1.0 - pow_b, 2, xi_0)
        
        # I = -i * [ B(0) * D'(0) * (-eta_a * eta_b) * 2F1(...) + B'(0) * 2F1(...) ]
        term1 = B_0 * xi_prime_0 * (-self.eta_a * eta_b) * F2
        term2 = B_prime_0 * F1
        I = -1j * (term1 + term2)
        
        return I

    def structure_integral(self, k1, u_r, V_r, r_grid):
        """
        Evaluate Z_l = \int_0^\infty r_1^2 dr_1 j_0(k_1 r_1) V_{bc}(r_1) u_0(r_1)
        Here we assume s-wave (l=0) where j_0(x) = sin(x)/x.
        """
        # If k1 is very small, j_0(k1*r) -> 1
        if k1 < 1e-8:
            j0_vals = np.ones_like(r_grid)
        else:
            j0_vals = np.sin(k1 * r_grid) / (k1 * r_grid)
            
        integrand = r_grid**2 * j0_vals * V_r * u_r
        # Integrate using Simpson's rule or trapezoidal rule
        Z_l = np.trapz(integrand, r_grid)
        return Z_l

    def triple_differential_cross_section(self, E_b_lab, theta_b, phi_b, theta_c, phi_c, u_r, V_r, r_grid):
        """
        Compute the triple differential cross section d^3\sigma / (dE_b d\Omega_b d\Omega_c) in mb / (MeV sr^2).
        """
        kin = self.solve_final_state(E_b_lab, theta_b, phi_b, theta_c, phi_c)
        if kin is None:
            return 0.0
            
        # 1. Structure part: Z_l
        Z_l = self.structure_integral(kin['k1'], u_r, V_r, r_grid)
        
        # 2. Dynamics part: Bremsstrahlung integral I
        I = self.bremsstrahlung_integral(kin['q_b'], kin['q_c'], kin['eta_b'])
        
        # 3. Phase space factor
        # \rho = m_b * m_c * m_t * p_b * p_c / ( (2*pi*hbar)^6 * (m_t + m_c - m_c * q_c. (q_a - q_b)/q_c^2) )
        # Using hbar = HBAR_C
        h_bar = HBAR_C
        p_b_mag = kin['p_b_mag']
        p_c_mag = kin['p_c_mag']
        
        q_c_unit = kin['q_c'] / np.linalg.norm(kin['q_c'])
        q_diff = kin['q_a'] - kin['q_b']
        den = self.m_t + self.m_c - self.m_c * np.dot(q_c_unit, q_diff) / np.linalg.norm(kin['q_c'])
        
        rho = (self.m_b * self.m_c * self.m_t * p_b_mag * p_c_mag) / (((2.0 * np.pi * h_bar)**6) * den)
        
        # 4. Cross section formula:
        # d3sigma = (2*pi / (hbar * v_a)) * \rho * [ 4*pi^2 * eta_a * eta_b / ((exp(2*pi*eta_b) - 1)*(exp(2*pi*eta_a) - 1)) ] * |I|^2 * (1 / 4*pi) * |Z_l|^2
        v_a = self.v_a_lab * 2.99792458e23  # relative velocity in fm/s or work in MeV units:
        # Relative velocity va in c:
        va_c = self.v_a_lab / (2.99792458e23 / HBAR_C)  # va in units of c
        # Wait, va is already calculated as self.v_a_lab in units of c if we divide by c.
        # Actually, self.v_a_lab = self.p_a_lab / self.m_a in units of c (since p is in MeV/c and m is in MeV/c^2).
        # So self.v_a_lab is already in units of c!
        # In this case: hbar * va = HBAR_C * self.v_a_lab (MeV fm).
        # This is extremely clean!
        
        # Prefactor
        prefactor = (2.0 * np.pi) / (HBAR_C * self.v_a_lab)
        
        # Coulomb factor
        coul_num = 4.0 * np.pi**2 * self.eta_a * kin['eta_b']
        coul_den = (np.exp(2.0 * np.pi * kin['eta_b']) - 1.0) * (np.exp(2.0 * np.pi * self.eta_a) - 1.0)
        coulomb_factor = coul_num / coul_den
        
        # d3sigma/dEb dOmega_b dOmega_c in fm^2/MeV = 10 mb/MeV (since 1 fm^2 = 10 mb)
        d3sigma = prefactor * rho * coulomb_factor * np.abs(I)**2 * (1.0 / (4.0 * np.pi)) * np.abs(Z_l)**2
        
        # Convert fm^2 / MeV to mb / MeV (1 fm^2 = 10 mb)
        d3sigma_mb = d3sigma * 10.0
        
        return d3sigma_mb

    def triple_differential_cross_section_from_Ec(self, E_c_lab, theta_b, phi_b, theta_c, phi_c, u_r, V_r, r_grid):
        """
        Compute the triple differential cross section d^3\sigma / (dE_b d\Omega_b d\Omega_c) in mb / (MeV sr^2)
        but specified in terms of the valence particle (neutron) energy E_c_lab.
        """
        kin = self.solve_final_state_from_Ec(E_c_lab, theta_b, phi_b, theta_c, phi_c)
        if kin is None:
            return 0.0
            
        # 1. Structure part: Z_l
        Z_l = self.structure_integral(kin['k1'], u_r, V_r, r_grid)
        
        # 2. Dynamics part: Bremsstrahlung integral I
        I = self.bremsstrahlung_integral(kin['q_b'], kin['q_c'], kin['eta_b'])
        
        # 3. Phase space factor
        h_bar = HBAR_C
        p_b_mag = kin['p_b_mag']
        p_c_mag = kin['p_c_mag']
        
        q_c_unit = kin['q_c'] / np.linalg.norm(kin['q_c'])
        q_diff = kin['q_a'] - kin['q_b']
        den = self.m_t + self.m_c - self.m_c * np.dot(q_c_unit, q_diff) / np.linalg.norm(kin['q_c'])
        
        rho = (self.m_b * self.m_c * self.m_t * p_b_mag * p_c_mag) / (((2.0 * np.pi * h_bar)**6) * den)
        
        # 4. Cross section formula
        prefactor = (2.0 * np.pi) / (HBAR_C * self.v_a_lab)
        
        coul_num = 4.0 * np.pi**2 * self.eta_a * kin['eta_b']
        coul_den = (np.exp(2.0 * np.pi * kin['eta_b']) - 1.0) * (np.exp(2.0 * np.pi * self.eta_a) - 1.0)
        coulomb_factor = coul_num / coul_den
        
        d3sigma = prefactor * rho * coulomb_factor * np.abs(I)**2 * (1.0 / (4.0 * np.pi)) * np.abs(Z_l)**2
        d3sigma_mb = d3sigma * 10.0
        
        return d3sigma_mb
