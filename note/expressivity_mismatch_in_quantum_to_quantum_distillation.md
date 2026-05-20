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

## Mismatch Margin Computation

A useful quantitative way to formalize expressivity mismatch is through a mismatch margin. This is the smallest achievable discrepancy between the trained teacher and the best realizable student within the student's ansatz family.

Let the mismatch margin over a distribution $\mathcal{D}$ be

$$
\gamma(\mathcal{D})
:=
\inf_{\theta_S}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
\ell\!\left(f_S(x;\theta_S),f_T(x;\theta_T^\star)\right)
\right].
$$

This quantity measures irreducible teacher-student mismatch after optimizing over all student parameters. If $\gamma(\mathcal{D})=0$, then the student family can in principle match the teacher on the distribution of interest. If $\gamma(\mathcal{D})>0$, then there is a nonzero approximation floor induced by the student architecture.

### 1. Expectation-value mismatch margin

If teacher and student both output scalar expectation values, then a natural definition is

$$
\gamma_{\mathrm{exp}}(\mathcal{D})
=
\inf_{\theta_S}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
|f_S(x;\theta_S)-f_T(x;\theta_T^\star)|^2
\right].
$$

This is often the simplest choice in VQAs with a single measured observable.

### 2. Vector-output mismatch margin

If the models output vectors of observables,

$$
\mathbf{f}_T(x),\mathbf{f}_S(x)\in\mathbb{R}^m,
$$

then the mismatch margin can be defined as

$$
\gamma_{\mathrm{vec}}(\mathcal{D})
=
\inf_{\theta_S}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
\|\mathbf{f}_S(x;\theta_S)-\mathbf{f}_T(x;\theta_T^\star)\|_2^2
\right].
$$

This is useful when the teacher exposes several expectation values or logits simultaneously.

### 3. Output-distribution mismatch margin

If the teacher and student output bitstring distributions $p_T(z|x)$ and $p_S(z|x;\theta_S)$, then one can define a distribution-level mismatch margin such as

$$
\gamma_{\mathrm{KL}}(\mathcal{D})
=
\inf_{\theta_S}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
D_{\mathrm{KL}}\!\left(p_T(\cdot|x)\,\|\,p_S(\cdot|x;\theta_S)\right)
\right].
$$

Other choices are also possible, for example total variation distance,

$$
\gamma_{\mathrm{TV}}(\mathcal{D})
=
\inf_{\theta_S}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
\frac{1}{2}\sum_z |p_T(z|x)-p_S(z|x;\theta_S)|
\right].
$$

These definitions are more informative when output probabilities themselves are important, not just low-dimensional expectation values.

### 4. State-level mismatch margin

If one wants a more structural quantum notion of mismatch, then one can compare the teacher and student quantum states directly:

$$
\gamma_{\rho}(\mathcal{D})
=
\inf_{\theta_S}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
1-F\!\left(\rho_T(x),\rho_S(x;\theta_S)\right)
\right],
$$

where $F(\rho,\sigma)$ denotes the quantum fidelity. This margin is conceptually appealing because it probes mismatch before measurement compression, but it is usually much harder to estimate on real hardware.

### 5. Observable-family constrained margin

Sometimes the student is not only restricted by its circuit family but also by its allowed observable set $\mathcal{O}_S$. In that case, the mismatch margin should explicitly optimize over both parameters and measurement choices:

$$
\gamma_{\mathcal{O}}(\mathcal{D})
=
\inf_{\theta_S,\; O_S \in \mathcal{O}_S}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
\left|
\operatorname{Tr}[O_S\rho_S(x;\theta_S)]
-
\operatorname{Tr}[O_T\rho_T(x)]
\right|^2
\right].
$$

This formulation isolates mismatch due to restricted readout rather than only restricted state preparation.

## Decomposition of the Observed Distillation Loss

In practice, the training loss observed after optimization mixes two effects:

1. optimization failure
2. irreducible mismatch

Conceptually, one can write

$$
\mathcal{L}_{\mathrm{KD}}^{\mathrm{obs}}
=
\gamma(\mathcal{D}) + \mathcal{E}_{\mathrm{opt}},
$$

where $\mathcal{E}_{\mathrm{opt}}$ is the residual error due to imperfect training. This decomposition is schematic rather than exact in all settings, but it is a useful interpretation. A large final loss does not by itself prove expressivity mismatch; one must separate architectural limitations from optimization failure.

## Practical Estimation

The exact mismatch margin is usually intractable because it requires global optimization over the full student family. In practice, one estimates it empirically by running the best available training procedure:

$$
\hat{\gamma}(\mathcal{D})
=
\min_{r \in \mathcal{R}}
\frac{1}{N}\sum_{i=1}^N
\ell\!\left(
f_S(x_i;\theta_S^{(r)}),
f_T(x_i;\theta_T^\star)
\right),
$$

where $\mathcal{R}$ indexes repeated training runs, restarts, or optimization strategies, and $\theta_S^{(r)}$ is the student obtained in run $r$.

This empirical estimate should be interpreted carefully:

- If $\hat{\gamma}$ is small and stable across runs, the student family is likely compatible with the teacher.
- If $\hat{\gamma}$ is large but highly variable across runs, optimization may still be the dominant issue.
- If $\hat{\gamma}$ remains large even under strong optimization, that is evidence for a true architectural mismatch.

Since training rarely finds the global optimum, the empirical quantity is best viewed as an upper bound on the ideal mismatch floor achievable by the student family under the chosen loss.

## Train-Test Margin Gap

It is also useful to compare mismatch margins across training and test distributions:

$$
\gamma_{\mathrm{train}}
:=
\inf_{\theta_S}
\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{train}}}
\left[
\ell\!\left(f_S(x;\theta_S),f_T(x;\theta_T^\star)\right)
\right],
$$

$$
\gamma_{\mathrm{test}}
:=
\inf_{\theta_S}
\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{test}}}
\left[
\ell\!\left(f_S(x;\theta_S),f_T(x;\theta_T^\star)\right)
\right].
$$

If

$$
\gamma_{\mathrm{train}} \ll \gamma_{\mathrm{test}},
$$

then the student may only match the teacher locally on the training region rather than globally on the broader task distribution.

## Normalized Mismatch Margin

To compare mismatch across tasks or output scales, one can define a normalized quantity such as

$$
\tilde{\gamma}
=
\frac{\gamma(\mathcal{D})}
{\mathbb{E}_{x\sim\mathcal{D}}[\|f_T(x)\|_2^2] + \epsilon},
$$

where $\epsilon > 0$ prevents division by zero. This makes the mismatch margin easier to compare across different teacher architectures and datasets.

## Depth-Based Mismatch Margin

To isolate mismatch caused specifically by circuit-depth limitations, one can define a depth-indexed student function class

$$
\mathcal{F}_S^{(L)}
:=
\left\{
f_S^{(L)}(\cdot;\theta)
\right\}_{\theta},
$$

where $L$ is the student circuit depth. The corresponding depth-based mismatch margin is

$$
\gamma_L(\mathcal{D})
:=
\inf_{\theta}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
\ell\!\left(
f_S^{(L)}(x;\theta),
f_T(x;\theta_T^\star)
\right)
\right].
$$

This quantity measures the smallest teacher-student discrepancy achievable by any student restricted to depth $L$.

### Scalar-output form

If the teacher and student both output a scalar observable, then a natural specialization is

$$
\gamma_L(\mathcal{D})
=
\inf_{\theta}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
\left|
f_S^{(L)}(x;\theta)-f_T(x;\theta_T^\star)
\right|^2
\right].
$$

This directly captures the irreducible output mismatch caused by restricting the student to depth $L$.

### Monotonicity with depth

In many ansatz families, increasing depth enlarges the student function class:

$$
\mathcal{F}_S^{(L)}
\subseteq
\mathcal{F}_S^{(L+1)}.
$$

As a result, the optimal mismatch margin is nonincreasing with depth:

$$
\gamma_{L+1}(\mathcal{D}) \le \gamma_L(\mathcal{D}).
$$

This expresses the intuition that extra depth can only improve the best achievable approximation or leave it unchanged.

### Irreducible depth floor

If the teacher relies on transformations unavailable to a shallow student, then for sufficiently small $L$,

$$
\gamma_L(\mathcal{D}) > 0.
$$

This nonzero value is the depth-induced approximation floor. It shows that even perfect optimization cannot remove the mismatch if the student's circuit is too shallow.

### Critical depth

For a tolerance level $\varepsilon > 0$, one can define the critical depth required to approximate the teacher:

$$
L_{\mathrm{crit}}
:=
\min
\left\{
L : \gamma_L(\mathcal{D}) \le \varepsilon
\right\}.
$$

This gives a compact way to summarize how much depth the student needs before distillation becomes accurate enough for a chosen standard.

### Unitary-level depth mismatch

The same idea can be phrased at the unitary or state level. Let $\mathcal{U}_S^{(L)}$ denote the family of unitaries reachable by the student ansatz at depth $L$. Then one may define

$$
\gamma_L^{U}(\mathcal{D})
:=
\inf_{U \in \mathcal{U}_S^{(L)}}
\mathbb{E}_{x\sim\mathcal{D}}
\left[
d\!\left(
U\rho_x U^\dagger,
\rho_T(x)
\right)
\right],
$$

where $\rho_T(x)$ is the teacher state and $d(\cdot,\cdot)$ is a state-distance measure such as infidelity or trace distance. This version captures mismatch before measurement compression.

### Practical empirical estimator

In experiments, the true depth-based margin is typically approximated by training students at several depths and taking the best observed loss:

$$
\hat{\gamma}_L
=
\min_{r \in \mathcal{R}}
\frac{1}{N}\sum_{i=1}^N
\ell\!\left(
f_S^{(L)}(x_i;\theta^{(r)}),
f_T(x_i;\theta_T^\star)
\right),
$$

where $\mathcal{R}$ indexes multiple training runs, random restarts, or optimizers.

The resulting depth profile

$$
L \mapsto \hat{\gamma}_L
$$

is useful for diagnosis:

- A steep decrease suggests that insufficient depth is a major cause of mismatch.
- A shallow decrease suggests that added depth helps only marginally.
- A persistent plateau above zero suggests that other architectural mismatches remain even as depth grows.

### Optimization-sensitive interpretation

The empirical quantity should be interpreted as

$$
\hat{\gamma}_L
=
\gamma_L + \mathcal{E}_{\mathrm{opt}}^{(L)},
$$

where $\mathcal{E}_{\mathrm{opt}}^{(L)}$ is the optimization residual at depth $L$. Thus, a large observed margin does not by itself prove expressivity mismatch. Strong evidence for depth-induced mismatch comes when $\hat{\gamma}_L$ remains stably large across restarts and training strategies.

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
