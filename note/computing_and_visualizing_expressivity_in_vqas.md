# Computing and Visualizing Expressivity in VQAs

A useful starting point is that expressivity in variational quantum algorithms (VQAs) can be defined in several different ways depending on what object one wants the circuit to represent. In practice, expressivity may refer to the richness of a function class, the diversity of reachable quantum states, the coverage of reachable unitaries, the induced geometry of a quantum feature map, or the flexibility of observable outputs. Because of this, there is no single universal equation for expressivity. Instead, different definitions are useful for different research questions.

## 1. Function-Space Expressivity

If a VQA defines a model

$$
f(x;\theta)=\operatorname{Tr}\!\left[O\,U(x,\theta)\rho_0 U^\dagger(x,\theta)\right],
$$

then the associated function class is

$$
\mathcal{F}=\{f(\cdot;\theta): \theta \in \Theta\}.
$$

One natural notion of expressivity is the ability of this class to approximate a target family of functions $\mathcal{G}$. This can be measured through

$$
\mathrm{Expr}_{\mathcal{G}}
:=
\sup_{g\in \mathcal{G}}
\inf_{\theta}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
|f(x;\theta)-g(x)|^2
\right].
$$

Smaller values indicate that the ansatz can represent a larger portion of the target family.

An empirical approximation at depth $L$ is

$$
\widehat{\mathrm{Expr}}(L)
=
\frac{1}{M}\sum_{j=1}^M
\min_{\theta}
\frac{1}{N}\sum_{i=1}^N
|f(x_i;\theta)-g_j(x_i)|^2,
$$

where $\{g_j\}_{j=1}^M$ is a test family of target functions.

### Useful visualizations

- Approximation error versus depth $L$
- Approximation error versus number of qubits
- Approximation error versus entangling-layer count

## 2. State-Space Expressivity

If the focus is on the diversity of reachable quantum states, define

$$
\rho(\theta)=U(\theta)\rho_0 U^\dagger(\theta).
$$

Then a common notion of expressivity is how broadly the ansatz-induced state distribution covers Hilbert space. One standard way to assess this is to compare pairwise fidelities between states sampled from the ansatz to those expected under the Haar distribution.

For pure states, the overlap-based fidelity can be written as

$$
F(\theta,\theta')
=
\operatorname{Tr}\!\left[\rho(\theta)\rho(\theta')\right].
$$

If the ansatz is highly expressive, the induced fidelity distribution

$$
P_{\mathrm{ansatz}}(F)
$$

should approach the Haar reference distribution

$$
P_{\mathrm{Haar}}(F).
$$

A quantitative discrepancy measure is

$$
\mathrm{Expr}_{\mathrm{state}}
=
D\!\left(
P_{\mathrm{ansatz}}(F),\,
P_{\mathrm{Haar}}(F)
\right),
$$

where $D$ may be KL divergence, total variation distance, or Wasserstein distance.

### Useful visualizations

- Histogram of pairwise fidelities
- Overlay of ansatz fidelity distribution and Haar fidelity distribution
- Divergence-to-Haar versus depth

## 3. Unitary Expressivity

If one wants to quantify the richness of reachable unitaries rather than states, a standard tool is the frame potential. For sampled parameter values $\theta,\theta'$, define

$$
\mathcal{F}^{(t)}
=
\mathbb{E}_{\theta,\theta'}
\left[
\left|
\operatorname{Tr}\!\left(U(\theta)^\dagger U(\theta')\right)
\right|^{2t}
\right].
$$

If the ansatz approximates a unitary $t$-design, then this quantity approaches the corresponding Haar value. Lower discrepancy from the Haar frame potential indicates higher unitary expressivity.

### Useful visualizations

- Frame potential $\mathcal{F}^{(t)}$ versus depth
- Ratio of frame potential to Haar frame potential
- Deviation from the Haar reference across different $t$

## 4. Feature-Map or Kernel Expressivity

If the VQA acts as a feature map, let

$$
|\phi(x)\rangle=\Phi(x)|0\rangle.
$$

Then the associated quantum kernel is

$$
K(x,x')
=
|\langle \phi(x)\mid \phi(x')\rangle|^2.
$$

In this setting, expressivity is tied to the geometry induced over the input data. Given a dataset $\{x_i\}_{i=1}^N$, define the kernel matrix

$$
K_{ij}=K(x_i,x_j).
$$

Several summary statistics are useful:

$$
\mathrm{rank}(K), \qquad
\mathrm{tr}(K), \qquad
\lambda_1,\dots,\lambda_N,
$$

where $\lambda_i$ are the eigenvalues of $K$.

A common compact measure is the effective rank

$$
r_{\mathrm{eff}}
=
\frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}.
$$

Higher effective rank usually indicates richer feature separation over the dataset.

### Useful visualizations

- Heatmap of the kernel matrix
- Eigenspectrum of the kernel matrix
- Effective rank versus depth or qubit count

## 5. Observable-Output Expressivity

Sometimes one only cares about the flexibility of the circuit's output observables rather than the full internal state family. In this case, expressivity can be studied through the covariance of outputs over parameter samples.

Define the mean output

$$
\mu(x)=\mathbb{E}_{\theta}[f(x;\theta)]
$$

and the covariance kernel

$$
\Sigma_f(x,x')
=
\mathbb{E}_{\theta}
\left[
\big(f(x;\theta)-\mu(x)\big)
\big(f(x';\theta)-\mu(x')\big)
\right].
$$

This object measures how flexibly the circuit can vary its outputs across the input domain.

One may also inspect the output variance

$$
\mathrm{Var}_{\theta}[f(x;\theta)].
$$

### Useful visualizations

- Covariance matrix over sampled inputs
- Eigenvalue spectrum of $\Sigma_f$
- Output variance versus depth

## 6. Teacher-Relative Expressivity

For distillation problems, an especially relevant notion is not abstract expressivity, but expressivity relative to a fixed teacher. If the student depth is $L$, define

$$
\gamma_L(\mathcal{D})
=
\inf_{\theta}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
\ell\!\left(
f_S^{(L)}(x;\theta),
f_T(x)
\right)
\right].
$$

This quantity measures the irreducible teacher-student gap at depth $L$. It can be interpreted as a depth-indexed expressivity curve for the student with respect to the teacher.

### Useful visualizations

- Mismatch margin versus depth
- Train-test mismatch curves
- Normalized mismatch margin across different student architectures

## 7. Practical Interpretation

These different definitions capture different aspects of expressivity:

- Function-space expressivity asks what input-output maps the ansatz can approximate.
- State-space expressivity asks what region of Hilbert space the ansatz can reach.
- Unitary expressivity asks how rich the reachable transformation family is.
- Kernel expressivity asks how richly the circuit separates data geometrically.
- Observable-output expressivity asks how flexibly the ansatz can vary its measured outputs.
- Teacher-relative expressivity asks how well a student family can approximate a specific teacher.

For quantum-to-quantum distillation, the most informative visualizations are often:

1. The mismatch margin curve

$$
L \mapsto \hat{\gamma}_L
$$

2. The pairwise-fidelity distribution

$$
P_{\mathrm{ansatz}}(F)
\quad \text{vs.} \quad
P_{\mathrm{Haar}}(F)
$$

3. The kernel eigenspectrum

$$
\lambda_1,\dots,\lambda_N
$$

Together, these show how much of the teacher the student can imitate, how broadly the ansatz covers state space, and how rich the induced feature geometry is.

## Conclusion

Expressivity in VQAs is not a single scalar property but a family of related notions tied to functions, states, unitaries, kernels, and outputs. As a result, computing or visualizing expressivity requires first deciding what kind of representational richness matters for the problem at hand. For distillation, teacher-relative measures such as the mismatch margin are often the most directly useful, while fidelity distributions, frame potentials, and kernel spectra provide complementary structural views.
