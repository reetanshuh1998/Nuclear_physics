import numpy as np
import scipy.special as sp
from woods_saxon import WoodsSaxonPotential, LagrangeMeshSolver, adjust_depth_to_binding_energy
from frdwba import FRDWBAReaction

HBAR_C = 197.327

def hyp2f1_complex(a, b, c, z, max_terms=200, tol=1e-16):
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
        
        # Choose the branch (-z) = |z| * exp(-i*pi) to get the exp(pi*eta) factor
        if np.real(z) > 0:
            log_neg_z = np.log(np.abs(z)) - 1j * np.pi
        else:
            log_neg_z = np.log(np.abs(z))
            
        neg_z_to_neg_a = np.exp(-a * log_neg_z)
        neg_z_to_neg_b = np.exp(-b * log_neg_z)
        
        term1_coeff = g_c * g_b_a / (g_b * g_c_a) * neg_z_to_neg_a
        term2_coeff = g_c * g_a_b / (g_a * g_c_b) * neg_z_to_neg_b
        
        f1 = hyp2f1_complex(a, 1 - c + a, 1 - b + a, 1.0 / z, max_terms, tol)
        f2 = hyp2f1_complex(b, 1 - c + b, 1 - a + b, 1.0 / z, max_terms, tol)
        
        return term1_coeff * f1 + term2_coeff * f2

# Setup solver
solver = LagrangeMeshSolver(N=40, h=0.35, mu=(939.565 * 9327.9)/(939.565 + 9327.9))
V0_opt = adjust_depth_to_binding_energy(solver, -0.50, r0=1.15, a=0.50, A_core=10.0, l=0, initial_guess=70.0)
opt_pot = WoodsSaxonPotential(V0_opt, r0=1.15, a=0.50, A_core=10.0)
eigenvalues, eigenvectors = solver.solve(opt_pot, l=0)
r_grid = np.linspace(0.01, 35.0, 500)
u_ws = solver.wavefunction(eigenvectors[:, 1], r_grid)
if u_ws[10] < 0:
    u_ws = -u_ws
V_ws_grid = opt_pot.evaluate(r_grid)

reaction_au = FRDWBAReaction(Z_a=4, A_a=11.0, Z_b=4, A_b=10.0, Z_c=0, A_c=1.0, Z_t=79, A_t=197.0, E_beam_per_A=44.0)

E_c_grid = [20.0, 30.0, 40.0, 50.0, 60.0]
th_n = np.radians(1.0)
th_b = np.radians(1.0)

for Ec in E_c_grid:
    kin = reaction_au.solve_final_state_from_Ec(Ec, th_b, 0.0, th_n, np.pi)
    if kin is None:
        continue
    
    q_b = kin['q_b']
    q_c = kin['q_c']
    eta_b = kin['eta_b']
    
    q_a = reaction_au.q_a
    k_vec = q_a - q_b - reaction_au.delta_mass * q_c
    k2 = np.sum(k_vec**2)
    
    q_a_mag = np.linalg.norm(q_a)
    q_b_mag = np.linalg.norm(q_b)
    
    # Positive definitions of u1 and u2:
    u1 = 2.0 * np.dot(k_vec, q_a) - k2
    u2 = 2.0 * np.dot(k_vec, q_b) + k2
    
    pow_a = 1j * reaction_au.eta_a
    pow_b = 1j * eta_b
    
    C_0 = 4.0 * np.pi / (k2**(pow_a + pow_b + 1.0))
    B_0 = C_0 * ((u1 + 0j)**pow_a) * ((u2 + 0j)**pow_b)
    
    u1_prime = 2.0 * q_a_mag
    u2_prime = 2.0 * q_b_mag
    B_prime_0 = B_0 * (pow_a * u1_prime / u1 + pow_b * u2_prime / u2)
    
    N_0 = 2.0 * k2 * (q_a_mag * q_b_mag + np.dot(q_a, q_b)) - 4.0 * np.dot(k_vec, q_a) * np.dot(k_vec, q_b)
    D_0 = u1 * u2
    xi_0 = N_0 / D_0
    
    N_prime_0 = 4.0 * q_b_mag * np.dot(k_vec, q_a) - 4.0 * q_a_mag * np.dot(k_vec, q_b)
    D_prime_0 = u1_prime * u2 + u1 * u2_prime
    xi_prime_0 = (N_prime_0 * D_0 - N_0 * D_prime_0) / (D_0**2)
    
    F1 = hyp2f1_complex(-pow_a, -pow_b, 1, xi_0)
    F2 = hyp2f1_complex(1.0 - pow_a, 1.0 - pow_b, 2, xi_0)
    
    term1 = B_0 * xi_prime_0 * (-reaction_au.eta_a * eta_b) * F2
    term2 = B_prime_0 * F1
    I = -1j * (term1 + term2)
    
    # Prefactors
    Z_l = reaction_au.structure_integral(kin['k1'], u_ws, V_ws_grid, r_grid)
    p_b_mag = kin['p_b_mag']
    p_c_mag = kin['p_c_mag']
    
    q_c_unit = q_c / np.linalg.norm(q_c)
    q_diff = q_a - q_b
    den = reaction_au.m_t + reaction_au.m_c - reaction_au.m_c * np.dot(q_c_unit, q_diff) / np.linalg.norm(q_c)
    
    rho = (reaction_au.m_b * reaction_au.m_c * reaction_au.m_t * p_b_mag * p_c_mag) / (((2.0 * np.pi * HBAR_C)**6) * den)
    
    prefactor = (2.0 * np.pi) / (HBAR_C * reaction_au.v_a_lab)
    
    coul_num = 4.0 * np.pi**2 * reaction_au.eta_a * eta_b
    coul_den = (np.exp(2.0 * np.pi * eta_b) - 1.0) * (np.exp(2.0 * np.pi * reaction_au.eta_a) - 1.0)
    coulomb_factor = coul_num / coul_den
    
    d3sigma = prefactor * rho * coulomb_factor * np.abs(I)**2 * (1.0 / (4.0 * np.pi)) * np.abs(Z_l)**2
    d3sigma_mb = d3sigma * 10.0
    
    print(f"Ec = {Ec}:")
    print(f"  u1 = {u1:.4f}, u2 = {u2:.4f}")
    print(f"  xi_0 = {xi_0:.4f}")
    print(f"  F1_abs = {np.abs(F1):.4e}, F2_abs = {np.abs(F2):.4e}")
    print(f"  I_abs = {np.abs(I):.4e}")
    print(f"  coulomb_factor = {coulomb_factor:.4e}")
    print(f"  dsig_mb = {d3sigma_mb * 0.82:.4f} mb/MeV.sr")
