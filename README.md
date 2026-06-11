# Nuclear Physics Frameworks: Woods-Saxon, AMD, and FRDWBA

This repository implements three fundamental nuclear physics frameworks using Object-Oriented Python to study nuclear structures and reaction dynamics for halo nuclei (e.g., $^{11}\text{Be}$):
1. **Woods-Saxon Potential & Radial Schrödinger Solver** (using the Lagrange-mesh method).
2. **Antisymmetrized Molecular Dynamics (AMD) Structure Solver** (microscopic Slater determinant, kinetic/potential energy expectation values, and overlap wavefunction projection).
3. **Finite Range Distorted Wave Born Approximation (FRDWBA) Reaction Solver** (three-body final state kinematics, distorted waves, and Nordsieck's analytical Bremsstrahlung integral evaluation).

---

## 1. Physics Background & Formulations

### A. Woods-Saxon Potential & Lagrange-Mesh Method
The ground state structure of the halo nucleus $^{11}\text{Be}$ is modeled as a two-body bound state consisting of a inert $^{10}\text{Be}$ core and a valence neutron ($^{10}\text{Be} + n$).
*   **Woods-Saxon Potential**:
    $$V(r) = - \frac{V_0}{1 + \exp\left(\frac{r - R}{a}\right)}$$
    where the nuclear radius is $R = r_0 A_{\text{core}}^{1/3}$ ($r_0 = 1.15 \text{ fm}$, $a = 0.50 \text{ fm}$, $A_{\text{core}} = 10$).
*   **Schrödinger Equation**: Solved on a Lagrange-Laguerre mesh. Mesh points $x_i$ are roots of the Laguerre polynomial $L_N(x)$ with scaled coordinates $r_i = h x_i$. The kinetic energy matrix elements are given by:
    $$T_{ii} = \frac{1}{12 x_i^2} [4 + (4N + 2)x_i - x_i^2]$$
    $$T_{ij} = (-1)^{i-j} (x_i x_j)^{-1/2} \frac{x_i + x_j}{(x_i - x_j)^2} \quad (i \neq j)$$
    The Hamiltonian matrix $H_{ij} = \frac{\hbar^2}{2\mu h^2} T_{ij} + V(r_i) \delta_{ij}$ is diagonalized to find the radial wavefunction $u(r)$.
*   **Binding Energy Calibration**: The potential depth $V_0$ is automatically optimized (to $V_0 \approx 71.0\text{ MeV}$) to match the experimental one-neutron separation energy $S_n = 0.50\text{ MeV}$ for the $2s_{1/2}$ state (which has 1 node in the nuclear interior).

### B. Antisymmetrized Molecular Dynamics (AMD)
AMD provides a microscopic description of the nucleus without assuming a core-valence cluster structure.
*   **Single-Particle Wave Packets**: Nucleons are represented by Gaussian wave packets:
    $$\phi_i(\vec{r}) = \left(\frac{2\nu}{\pi}\right)^{3/4} \exp\left(-\nu\left(\vec{r} - \frac{\vec{Z}_i}{\sqrt{\nu}}\right)^2\right) \otimes \chi_{\sigma_i} \otimes \chi_{\tau_i}$$
    where $\vec{Z}_i = \sqrt{\nu}\vec{R}_i + i \frac{\vec{P}_i}{2\hbar\sqrt{\nu}}$ are complex coordinates representing position and momentum centroids.
*   **Slater Determinant**: The multi-nucleon wavefunction is the antisymmetrized product:
    $$\Phi = \frac{1}{\sqrt{A!}} \det[\phi_i(\vec{r}_j)]$$
    The overlap matrix is $B_{ij} = \langle \phi_i | \phi_j \rangle = \exp\left(-\frac{1}{2}(\vec{Z}_i^* - \vec{Z}_j)^2\right) \delta_{\sigma_i \sigma_j} \delta_{\tau_i \tau_j}$.
*   **Energy Expectation Values**:
    *   **Kinetic Energy**:
        $$\langle T \rangle = \sum_{i,j} \langle \phi_i | T | \phi_j \rangle B^{-1}_{ji}$$
        where $T_{ij} = \frac{\hbar^2 \nu}{2M} \left( 3 - (\vec{Z}_i^* - \vec{Z}_j)^2 \right) B_{ij}$.
    *   **Central Gaussian Potential** ($V(r) = V_0 e^{-r^2/\mu^2}$):
        $$\langle V \rangle = \frac{1}{2} \sum_{i,j,k,l} B^{-1}_{ki} B^{-1}_{lj} \left( \langle i j | V | k l \rangle - \langle i j | V | l k \rangle \right)$$
        where the matrix elements are evaluated analytically using the centroid coordinates.
*   **Overlap Wavefunction Extraction**: The core-valence overlap wavefunction is:
    $$y_0(r) = \sqrt{A} \langle \Phi_{A-1} | \Phi_A \rangle$$
    This is projected onto the $l=0$ s-wave by analytical angular integration, yielding the radial overlap amplitude $u_{\text{AMD}}(r) = r \sqrt{4\pi} y_0(r)$.

### C. Finite Range Distorted Wave Born Approximation (FRDWBA)
FRDWBA calculates the cross sections for the three-body Coulomb breakup reaction $a + t \rightarrow b + c + t$ (e.g., $^{11}\text{Be} + {}^{197}\text{Au} \rightarrow {}^{10}\text{Be} + n + {}^{197}\text{Au}$).
*   **Bremsstrahlung Integral**: Evaluates the transition matrix element involving Coulomb distorted waves:
    $$I = \int e^{-\lambda r} e^{i\vec{k}\cdot\vec{r}} {}_1F_1(i\eta_b, 1, i(q_b r + \vec{q}_b\cdot\vec{r})) {}_1F_1(i\eta_a, 1, i(q_a r - \vec{q}_a\cdot\vec{r})) d\vec{r}$$
    This is calculated analytically using Nordsieck's confluent hypergeometric derivative formulation:
    $$I = -i \left[ B(0) D'(0) (-\eta_a \eta_b) {}_2F_1(1-i\eta_a, 1-i\eta_b; 2; D(0)) + B'(0) {}_2F_1(-i\eta_a, -i\eta_b; 1; D(0)) \right]$$
*   **Piecewise Hypergeometric Continuation**: When $|z| \geq 0.9$ (divergence region for standard series), we apply the analytical continuation formula:
    $${}_2F_1(a, b; c; z) = \frac{\Gamma(c)\Gamma(b-a)}{\Gamma(b)\Gamma(c-a)} (-z)^{-a} {}_2F_1\left(a, 1-c+a; 1-b+a; \frac{1}{z}\right) + \frac{\Gamma(c)\Gamma(a-b)}{\Gamma(a)\Gamma(c-b)} (-z)^{-b} {}_2F_1\left(b, 1-c+b; 1-a+b; \frac{1}{z}\right)$$
    The base $(-z)$ is evaluated on the physical complex branch cut $\ln(-z) = \ln|z| - i\pi$ to cancel the Coulomb suppression factors.
*   **Triple Differential Cross Section**:
    $$\frac{d^3\sigma}{dE_b d\Omega_b d\Omega_c} = \frac{2\pi}{\hbar v_a} \rho(E_b, \Omega_b, \Omega_c) \left[ \frac{4\pi^2 \eta_a \eta_b}{(e^{2\pi\eta_b}-1)(e^{2\pi\eta_a}-1)} \right] |I|^2 \frac{1}{4\pi} |Z_l|^2$$
    where $Z_l = \int_0^\infty r_1^2 dr_1 j_l(k_1 r_1) V_{bc}(r_1) u_l(r_1)$ is the structural overlap integral containing the local momentum transfer $k_1$.

---

## 2. Directory Structure

*   `woods_saxon.py`: Implements the `WoodsSaxonPotential` class, Lagrange-mesh solver (`LagrangeMeshSolver`), and bound-state binding energy calibration.
*   `amd.py`: Implements the microscopic `Nucleon`, `AMDState` classes, matrix calculations for norms, kinetic energy, potential energy, energy optimization, and the radial overlap wavefunction extraction.
*   `frdwba.py`: Implements three-body kinematics, phase space factors, Nordsieck's Bremsstrahlung integral, and the piecewise confluent hypergeometric solver.
*   `run_calculations.py`: Orchestrates all structure and reaction calculations, saving plots for structure, AMD wavefunction comparison, triple differential cross sections, parallel momentum distributions, relative energy spectra, and E1 dipole strength.

---

## 3. Installation & Dependencies

To run the simulation suite, you will need `python3` along with standard scientific computing packages:
```bash
pip install numpy matplotlib scipy
```

---

## 4. Running the Simulations

You can execute the entire suite of calculations (which will solve the bound state, extract the AMD overlap, run reaction dynamics, and generate the plots) by executing:
```bash
python3 run_calculations.py
```

### Outputs & Generated Figures:
The script generates and saves the following plots to the project directory:
1.  `woods_saxon_structure.png`: Woods-Saxon potential and the bound radial wavefunction $u(r)$ for the $2s_{1/2}$ state of $^{11}\text{Be}$.
2.  `wavefunction_comparison.png`: Comparison between the phenomenological Woods-Saxon wavefunction and the microscopic AMD overlap wavefunction showing the physical interior node.
3.  `triple_differential_cross_section.png`: Triple differential cross sections for four angle combinations reproducing Figure 8 of the paper.
4.  `parallel_momentum_distribution.png`: Parallel momentum distribution of the core ($^{10}\text{Be}$) in Coulomb breakup on $^{181}\text{Ta}$ at $63\text{ MeV}/u$, verifying a FWHM of $\approx 29.9\text{ MeV}/c$.
5.  `relative_energy_and_dipole.png`: Dipole response strength $dB(E1)/dE_{\text{rel}}$ and relative energy spectrum $d\sigma/dE_{\text{rel}}$ for breakup on $^{208}\text{Pb}$ at $69\text{ MeV}/u$.
