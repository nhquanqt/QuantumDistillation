# Expressivity Mismatch in Quantum-to-Quantum Distillation

Expressivity mismatch is one of the strongest reasons quantum-to-quantum distillation can fail. The basic issue is that the student may be asked to imitate a function that the teacher can represent but the student cannot, at least not with its available qubits, depth, entangling pattern, gate set, or measurement scheme.

In a variational quantum algorithm (VQA), expressivity is not just a matter of parameter count. It depends on several tightly coupled resources, including ansatz structure, circuit depth, entanglement topology, available observables, data encoding method, number of qubits, and hardware connectivity. Because of this, two quantum models that appear similar in size may still differ substantially in the family of functions they can represent.

## Core Idea

If the teacher's realized function lies outside the effective hypothesis class of the student, then distillation becomes an impossible approximation problem rather than a simple compression problem.

Under this view, failure is not necessarily caused by poor optimization alone. Instead, the student may be structurally incapable of reproducing the behavior that the teacher has learned.

Let the teacher and student define quantum models

$$
f_T(x) = \operatorname{Tr}\!\left[ O_T \, U_T(x,\theta_T)\rho_0 U_T^\dagger(x,\theta_T) \right]
$$

and

$$
f_S(x) = \operatorname{Tr}\!\left[ O_S \, U_S(x,\theta_S)\rho_0 U_S^\dagger(x,\theta_S) \right].
$$

Here, $x$ is the input, $\rho_0$ is the initial state, $U_T$ and $U_S$ are the teacher and student parameterized circuits, and $O_T, O_S$ are the measured observables. Distillation assumes there exists some $\theta_S^\star$ such that

$$
f_S(x;\theta_S^\star) \approx f_T(x;\theta_T^\star)
\quad \text{for } x \sim \mathcal{D}.
$$

Expressivity mismatch means this approximation may be impossible or badly limited:

$$
\inf_{\theta_S}\; \mathbb{E}_{x\sim\mathcal{D}}
\left[
\ell\!\left(f_S(x;\theta_S),f_T(x;\theta_T^\star)\right)
\right]
> \varepsilon
$$

for some nontrivial $\varepsilon > 0$. In words, even the best student inside its ansatz family cannot drive the teacher-matching loss arbitrarily low.

## Why This Matters in VQAs

In classical distillation, it is often reasonable to assume that a smaller neural network can approximate a larger one sufficiently well on the relevant task distribution. In VQAs, that assumption is much more fragile. A teacher may benefit from depth-induced interference effects, multipartite entanglement, or expressive feature maps that a student cannot reproduce once the circuit is compressed.

This means the student may not simply be undertrained. It may be fundamentally restricted to a smaller and qualitatively different function class. Distillation then pushes the student toward a target it cannot realize, which can lead to underfitting, unstable convergence, or only partial imitation of the teacher's outputs.

## Forms of Expressivity Mismatch

### 1. Depth mismatch

A deep teacher may realize correlations that require repeated layers of entangling gates and nonlinear interference effects. A shallow student may only access lower-complexity transformations, even if the two models have a similar number of trainable parameters.

If the teacher circuit has depth $L_T$ and the student has depth $L_S$ with $L_S \ll L_T$, then the accessible unitary families satisfy

$$
\mathcal{U}_{S}^{(L_S)} \subsetneq \mathcal{U}_{T}^{(L_T)}
$$

in the practical sense that the student can only realize a restricted subset of the transformations available to the teacher. If the teacher's target unitary $U_T^\star$ lies far from this family, then

$$
\inf_{U \in \mathcal{U}_{S}^{(L_S)}} \|U - U_T^\star\| \ge \delta_L
$$

for some $\delta_L > 0$ under an appropriate operator norm or task-induced metric.

At the function level, this gives

$$
\inf_{\theta_S}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
|f_S(x;\theta_S)-f_T(x;\theta_T^\star)|^2
\right]
\ge c_L,
$$

meaning depth reduction induces an irreducible approximation floor.

### 2. Entanglement mismatch

The teacher may prepare multipartite entangled states or long-range correlations that the student ansatz cannot generate. If task performance depends on those correlations, the student cannot faithfully reproduce the teacher's mechanism.

Let

$$
\rho_T(x)=U_T(x,\theta_T^\star)\rho_0U_T^\dagger(x,\theta_T^\star), \qquad
\rho_S(x)=U_S(x,\theta_S)\rho_0U_S^\dagger(x,\theta_S).
$$

Suppose the teacher prepares states with entanglement entropy across a bipartition $A|B$

$$
S_A(\rho_T(x)) = -\operatorname{Tr}\big(\rho_{T,A}(x)\log \rho_{T,A}(x)\big)
$$

that exceeds what the student ansatz can realize:

$$
\sup_{\theta_S} S_A(\rho_S(x)) < S_A(\rho_T(x))
$$

for relevant inputs $x$. Then the student cannot reproduce the same correlation structure.

Equivalently, if the task depends on a teacher correlation function such as

$$
C_T^{(ij)}(x)=
\langle Z_i Z_j \rangle_T - \langle Z_i\rangle_T \langle Z_j\rangle_T,
$$

and the student family obeys

$$
\sup_{\theta_S}|C_S^{(ij)}(x)| < |C_T^{(ij)}(x)|,
$$

then the student cannot match the teacher's nonclassical correlations even if their final labels sometimes agree.

### 3. Encoding mismatch

If the teacher and student use different data-encoding circuits or feature maps, then they do not even begin from the same input geometry. The student is not just compressing the teacher's function, but trying to mimic behavior produced by a different representation of the data.

Let the teacher and student use feature maps $\Phi_T(x)$ and $\Phi_S(x)$, producing encoded states

$$
|\phi_T(x)\rangle = \Phi_T(x)|0\rangle, \qquad
|\phi_S(x)\rangle = \Phi_S(x)|0\rangle.
$$

Their induced kernels are

$$
K_T(x,x') = |\langle \phi_T(x)\mid \phi_T(x')\rangle|^2,
\qquad
K_S(x,x') = |\langle \phi_S(x)\mid \phi_S(x')\rangle|^2.
$$

If

$$
K_T(x,x') \not\approx K_S(x,x')
$$

across the data distribution, then teacher and student embed the same inputs into different geometries in Hilbert space. Distillation then tries to fit

$$
f_T(x)=g_T(\Phi_T(x))
$$

with a student of the form

$$
f_S(x)=g_S(\Phi_S(x)),
$$

which may be impossible unless $g_S$ compensates for the geometric mismatch.

### 4. Observable mismatch

Even if the student approximates some of the teacher's internal behavior, it may not have access to the same measurement operators or output channels. As a result, the student may be unable to expose the same predictive structure at the output level.

Suppose the teacher prediction is derived from observable $O_T$ and the student from a restricted observable family $\mathcal{O}_S$. Then even with similar internal states, the student must satisfy

$$
f_S(x)=\operatorname{Tr}[O_S \rho_S(x)], \qquad O_S \in \mathcal{O}_S.
$$

If no observable in $\mathcal{O}_S$ can reproduce the teacher readout,

$$
\inf_{O_S \in \mathcal{O}_S,\theta_S}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
\left|
\operatorname{Tr}[O_S\rho_S(x)]-\operatorname{Tr}[O_T\rho_T(x)]
\right|^2
\right]
> 0,
$$

then the output map itself is mismatched.

This can also be phrased as an information bottleneck: the teacher may expose a vector of observables

$$
\mathbf{f}_T(x)=
\big(
\langle O_T^{(1)}\rangle,\dots,\langle O_T^{(m)}\rangle
\big),
$$

while the student may only output

$$
\mathbf{f}_S(x)=
\big(
\langle O_S^{(1)}\rangle,\dots,\langle O_S^{(r)}\rangle
\big),
\qquad r<m,
$$

with restricted $O_S^{(k)}$, making faithful output imitation impossible.

### 5. Gate-set and connectivity mismatch

A student designed for hardware efficiency may be constrained by native gates and device connectivity. These restrictions can prevent it from realizing the same unitary family as the teacher, especially if the teacher was trained in ideal simulation or with a less restricted circuit architecture.

Let $\mathcal{G}_T$ and $\mathcal{G}_S$ be the teacher and student gate libraries, and let $G_T=(V,E_T)$, $G_S=(V,E_S)$ denote their interaction graphs. The teacher may implement

$$
U_T^\star \in \langle \mathcal{G}_T, E_T \rangle,
$$

while the student is restricted to

$$
U_S(\theta_S) \in \langle \mathcal{G}_S, E_S \rangle.
$$

If $\mathcal{G}_S$ and $E_S$ are more restrictive, then compiling the teacher behavior into the student family incurs error:

$$
\inf_{\theta_S}
\|U_S(\theta_S)-U_T^\star\|_{\diamond \text{ or op}}
\ge \delta_G.
$$

Under noise, the relevant comparison may be between implemented channels:

$$
\mathcal{E}_T^\star \neq \mathcal{E}_S(\theta_S),
$$

and the mismatch becomes

$$
\inf_{\theta_S}
\|\mathcal{E}_S(\theta_S)-\mathcal{E}_T^\star\|_\diamond
\ge \delta_{\text{hw}}.
$$

## Why Parameter Count Is Not Enough

One subtle issue is that parameter count alone can be misleading. Two VQAs with similar numbers of parameters may still have very different expressive power. A hardware-efficient student may have many tunable angles, but if those angles live inside a restrictive ansatz, they may not span the same family of states or observables as the teacher.

Formally, let $p_T$ and $p_S$ be the parameter counts. Even if

$$
p_S \approx p_T,
$$

it does not follow that

$$
\mathcal{F}_S := \{f_S(\cdot;\theta_S)\}_{\theta_S}
\approx
\mathcal{F}_T := \{f_T(\cdot;\theta_T)\}_{\theta_T}.
$$

The relevant comparison is between function classes, not scalar parameter totals. Distillation fails when

$$
\mathcal{F}_T \not\subseteq \overline{\mathcal{F}_S}
$$

over the distribution and metric of interest.

This makes expressivity mismatch more structural than numerical. The key question is not just whether the student is smaller, but whether its accessible function class aligns with the family of behaviors that gave the teacher its performance.

## Effect on Distillation

Distillation usually assumes that the teacher provides a meaningful target that the student can at least approximately realize. When that assumption fails, the distillation objective may become harmful rather than helpful.

Instead of guiding the student toward a good solution, the KD loss may push it toward unreachable targets. This can produce several problems:

- Persistent approximation error even after optimization converges
- Conflicting gradient directions during training
- Apparent imitation on a narrow training set but weak generalization
- A tendency to match only coarse output statistics rather than the real underlying mechanism

In other words, the student may learn a projection or caricature of the teacher instead of its real decision rule.

If the KD objective is, for example,

$$
\mathcal{L}_{\mathrm{KD}}(\theta_S)
=
\mathbb{E}_{x\sim\mathcal{D}}
\left[
\ell\!\left(f_S(x;\theta_S),f_T(x;\theta_T^\star)\right)
\right],
$$

then expressivity mismatch implies

$$
\min_{\theta_S}\mathcal{L}_{\mathrm{KD}}(\theta_S)
=
\mathcal{L}_{\mathrm{approx}}
>
0.
$$

The nonzero floor $\mathcal{L}_{\mathrm{approx}}$ is the approximation error induced by the student ansatz family itself, not merely by failed optimization.

## Global vs. Local Mismatch

It is also useful to distinguish between global and local expressivity mismatch.

Global mismatch means the student cannot reproduce the teacher's function across the relevant input space. Local mismatch means the student cannot match the teacher in general, but may still mimic it reasonably well on the training distribution or on a small benchmark.

Global mismatch can be written as

$$
\inf_{\theta_S}
\sup_{x \in \mathcal{X}}
|f_S(x;\theta_S)-f_T(x;\theta_T^\star)|
\ge \varepsilon_{\mathrm{global}}.
$$

Local mismatch is weaker:

$$
\inf_{\theta_S}
\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{train}}}
|f_S(x;\theta_S)-f_T(x;\theta_T^\star)|
\ll
\inf_{\theta_S}
\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{test}}}
|f_S(x;\theta_S)-f_T(x;\theta_T^\star)|.
$$

So the student may appear to distill successfully on a narrow training distribution while still failing to reproduce the teacher more broadly.

This distinction matters because apparent empirical success on small datasets does not necessarily imply true transfer of quantum knowledge. The student may only be fitting a narrow slice of the teacher's behavior without capturing the mechanism that produced it.

## Research Framing

A useful way to state the issue is:

> Expressivity mismatch in quantum-to-quantum distillation is not merely a compression gap; it is often a structural incompatibility between the state families, correlation patterns, and measurement maps available to teacher and student.

This framing shifts the discussion away from simple model size and toward representational compatibility.

## Testable Predictions

Several predictions follow from this view:

- Distillation should fail more often when student depth is aggressively reduced.
- Students with hardware-efficient ansatze should struggle more when the teacher uses a task-specific or problem-inspired ansatz.
- Output-only distillation should be weakest when the teacher relies on richer entanglement structure than the student can prepare.
- Distillation may improve if the teacher is constrained in advance to a student-compatible ansatz family.
- Success on small benchmarks may overstate transfer if the student only matches the teacher locally rather than globally.

## Conclusion

Expressivity mismatch suggests that some failures of quantum-to-quantum distillation are not optimization accidents, but consequences of structural incompatibility. A student VQA may be too restricted to represent the same function, state family, or measurement behavior as the teacher. When this happens, distillation becomes an ill-posed transfer problem rather than a straightforward form of model compression.
