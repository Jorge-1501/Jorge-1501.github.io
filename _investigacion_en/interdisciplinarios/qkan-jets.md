---
layout: proyecto-layout
title: "Quantum Sine-Kolmogorov-Arnold Networks for High Energy Physics Analysis"
proyecto: "Quantum KAN & Jets"
seccion: "Technical summary"
area: "Interdisciplinary"
tipo: "macro"
estatus: "Completed"
description: "Quantum Kolmogorov-Arnold Networks for jet tagging, my Google Summer of Code 2026 project with ML4SCI."
orden: 1
---
# Quantum Kolmogorov-Arnold Networks for top quark jet tagging

*A technical summary of the QKAN project, Google Summer of Code 2026 at ML4SCI*

This project was developed by Jorge Toral and can be found in this [GitHub repository](https://github.com/Jorge-1501/QKANs-ML4SCI_2026).

---

## 1. The problem and the core idea

At the Large Hadron Collider, a top quark decays almost instantaneously and produces a **jet** (a collimated spray of particles) with a characteristic internal substructure, distinct from that of an ordinary jet originating from a light quark or a gluon (QCD). Distinguishing these two types of jet, a task known as **top tagging**, is a well-studied binary classification problem, with classical reference architectures reaching AUCs above 0.96-0.98 on the public dataset used in this project ([Kasieczka et al., 2019](#ref-kasieczka-2019)): jets simulated at 14 TeV with Pythia8 and an ATLAS-like Delphes detector card, reconstructed with anti-$k_T$ and $R=0.8$ in the range $p_T \in [550, 650]$ GeV.

The goal of the project is to answer a question other than outperforming those architectures: **can a variational quantum neural network (VQC) perform this task, and what role can a classical network play in making it feasible?**

The underlying limitation is the cost of simulating a quantum circuit, which grows as $O(2^Q)$ with the number of qubits $Q$. Feeding a jet with dozens of variables directly into a VQC is impractical. The solution explored here uses a classical **Kolmogorov-Arnold Network (KAN)** as a preprocessor and qubit filter: it is trained, pruned down to a small, interpretable topology, and that pruned topology determines how many qubits the quantum circuit needs and how they are connected. In this way, the classical model sets the quantum resource systematically, without resorting to trial and error.

The full pipeline, documented across three notebooks (`EDA_top.ipynb`, `Training_process.ipynb`, `Results.ipynb`), preprocesses the jets into balanced replicas, trains and prunes the KAN, extracts each surviving edge into a compact basis (Chebyshev or sine) as a *warm start*, fine-tunes the QKAN on an ideal simulator, on one with finite-shot noise, and on one with hardware noise, and compares everything against a classical Random Forest.

## 2. From the original proposal to the final pipeline

The original GSoC 2026 proposal for this project, *"Quantum Sine-Kolmogorov-Arnold Networks for High Energy Physics Analysis"*, set out a goal that was both narrower and more ambitious: implementing a **Q-SineKAN** in PennyLane in which the univariate activation functions of a KAN were replaced directly by parameterized quantum circuits (PQCs) acting as native sine functions, using *data re-uploading* to match the expressivity of the learnable frequencies, amplitudes, and phases of the classical SineKAN ($A\sin(\omega x + \phi)$, encoded as $R_y(\omega x + \phi)$ rotations). The initial motivation came from the work of [Ivashkov et al. (2024)](#ref-ivashkov-2024), which had proposed a theoretical QKAN built on fault-tolerant subroutines such as the Quantum Singular Value Transformation (QSVT), unfeasible on current noisy hardware. The proposal put forward sinusoidal re-uploading as the route to something similar that would be usable in the NISQ era.

That approach remained almost intact: the sine basis compared in section 7 corresponds to the proposed Q-SineKAN, implemented with the same fixed frequency-and-phase grid as the reference SineKAN. What changed as the work progressed was the structure surrounding that central piece.

The first change was structural. The original proposal trained the quantum circuit directly on the jet's kinematic variables, without a prior classical pruning step acting as a qubit filter. The plan did include training a reference classical KAN, but only as a comparison *baseline*, without any role in deciding the quantum topology. In practice, feeding 22 unfiltered variables into a re-uploading circuit would have required too many qubits to simulate in reasonable time. This is the same obstacle Ria Khatoniar documents in her GSoC 2025 project (section 11), where her hybrid QKAN "could not scale beyond a certain point" on the quark-gluon dataset because of simulator memory failures. Structured KAN pruning was therefore adopted as a mandatory step before any quantization, as a direct response to a limitation that only became evident when scaling the original approach.

The second change was adding the Chebyshev basis as a point of comparison, which the proposal did not include because it deliberately committed to the sinusoidal basis. Building that comparison produced the most relevant finding of section 7: the sinusoidal basis fits each edge worse ($R^2 \approx 0.49$-$0.61$) than Chebyshev ($R^2$ close to 1), because a fixed frequency grid with no constant term cannot represent static offsets. The original proposal, written before any data was available, could not anticipate this result. It appears because the project ended up testing the proposal's central hypothesis, "the sinusoidal formulation is exceptionally natural for the quantum domain", against an alternative.

Finally, the proposal identified the risk of **barren plateaus** when using deep re-uploading, as part of the convergence criteria to monitor (together with the gradient norm $\|\nabla_\theta \mathcal{L}\| \to 0$ and *early stopping*). That risk was built directly into an architectural decision: the final circuit supports only depth-2 networks (section 6), partly to avoid stacking unnecessary re-uploading layers. The paper ["Is Data-Reuploading Really a Cheat Code?" (Arias Alamo et al., 2025)](#ref-ariasalamo-2025) documents this problem with experimental evidence: an excess of re-uploading layers produces barren plateaus without expressivity gains that would justify them.

In summary, the executed project answers the question posed by the original proposal with a broader architecture than planned: two bases instead of one and an explicit classical pruning step. This broadening results from testing each of the risks the proposal itself had already anticipated.

## 3. Motivation for the KAN architecture

Kolmogorov-Arnold Networks, proposed by [Liu et al. (2024)](#ref-liu-2024-kan) building on the Kolmogorov-Arnold representation theorem (KAT), invert the design of a traditional multilayer perceptron (MLP). In an MLP, the activation functions are fixed (ReLU, tanh, ...) and live on the nodes; the learnable weights are scalars on the edges. In a KAN the relationship is inverted: **each edge carries its own learnable univariate function**, parameterized as a spline, and the nodes only sum. The KAT theorem guarantees that any continuous multivariate function can be written as

$$f(x_1, \dots, x_n) = \sum_{q=1}^{2n+1} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

that is, as a composition of univariate functions and sums. Two properties follow that this project exploits directly. The first is **interpretability**: once training is finished, each edge can be fit symbolically against a library of candidate functions, giving a closed-form expression for the transformation it performs. The second, crucial for this project, is **structured prunability**: since each edge is an independent functional unit, an attribution (how much it contributes to the output) can be measured and the edge removed if negligible, without breaking the interpretation of the rest of the network.

A follow-up work by the same authors, [KAN 2.0 (Liu et al., 2024)](#ref-liu-2024-kan2), also introduces **multiplication nodes**. The KAT theorem in its classical form only guarantees composition through sums; allowing some hidden nodes to multiply their inputs expresses variable interactions directly. This project's classical architecture (`HEPKAN`, a subclass of [pykan](https://github.com/kindxiaoming/pykan)) adopts this idea and uses a hidden layer with sum *and* multiplication nodes in parallel.

## 4. Data exploration: what distinguishes a top jet

Before training, the EDA notebook characterizes the dataset. The HDF5 file does not document the order of its columns, so it was inferred from the data: each block of four columns corresponds to one constituent particle of the jet, with energy in the first position and the three momentum components afterward. The pattern is recognizable in the column means, which are much larger in the first column of each block of four.

Four observations emerged from the histograms and radial profiles and guided the subsequent design:

- **Invariant mass discriminates, but isn't enough.** $m_{jet} = \sqrt{E^2 - p_x^2 - p_y^2 - p_z^2}$ has a clear peak in the signal around the top quark mass (~173 GeV), while the background is wider. The two distributions overlap by sections, so the **145-205 GeV** window was selected for the main experiments: within it, the trivial mass cue is largely removed and classifiers must rely on subtler information.
- **Top jets are more populated**: multiplicity (number of constituents) is systematically higher and more spread out than in QCD, consistent with a three-body decay, for this selected mass window.
- **Top jets are more diffuse**: the radial energy profile and the cumulative $p_T$ fraction grow more gradually in tops than in QCD, where energy is more concentrated near the jet axis.
- $\eta$-$\phi$ scatter plots were inconclusive due to noise and scale.

An artifact was also detected: the $\Delta R$ distribution does not show the expected sharp cutoff at 0.8, due to a small constant added to avoid division by zero. Constituents outside the jet radius were filtered out to correct it.

From this the input representation was built, combining **global** variables (jet mass and multiplicity, counting only constituents with energy above $10^{-8}$) with **local** variables for each retained constituent ($\Delta R$ to the jet axis and $p_{T,\text{rel}}$). Using the 80% cumulative $p_T$ criterion, around 15 constituents were estimated to suffice to capture most of the energy. The final pipeline uses only **the 10 most energetic constituents per jet**, for a total of **22 inputs** (2 global + 10 pairs of local variables). Each original jet carries up to 200 stored constituents; the cut to 10 is a deliberate compression decision revisited in the results and conclusions, since the model never sees 95% of the particles recorded by the detector.

The local variables already live in $(0,1)$. The global ones undergo a logarithmic transform followed by tanh normalization, deliberately keeping the outliers: the centers of the two class distributions are similar and the discriminative information lies in the tails.

<figure>
  <img src="/assets/img/investigacion/qkan/dispersion.png" alt="Constituent dispersion">
  <figcaption>Dispersion of the jet constituents in top and QCD signals. Range before any mass cut.</figcaption>
</figure>

<figure>
  <img src="/assets/img/investigacion/qkan/Invariant_mass.png" alt="Invariant mass">
  <figcaption>Invariant mass of the jets in top and QCD signals, before any mass cut. The signal peak at ~173 GeV and the 145-205 GeV window chosen for the main regime are visible.</figcaption>
</figure>

Finally, since the number of events in the mass window differs between classes (there are more tops), the majority class is subsampled to balance it, and the result is split into **5 disjoint, class-balanced subsets**. Each seed selects one (`seed % 5`), so several seeds produce independent end-to-end replicas rather than a single point estimate.

## 5. The classical architecture and its pruning

The classical model has architecture `[22, [9, 9], 1]`: 22 inputs, one hidden layer with up to 9 sum nodes and 9 multiplication nodes in parallel, and a scalar output. The B-splines use degree $k=3$ and grid size 5, for a total of **8,568 parameters**. In the reference run (seed 10), the base model reached a test AUC of **0.792** in the point evaluation reported in the notebook, consistent with the mean of **0.789 ± 0.002** obtained by aggregating the five replicas (section 9). Training stopped early, around epoch 20 of the planned 60.

`HEPKAN` introduces three practical modifications over `pykan`, motivated by the project's limited computational resources: a fix for a bug in `prune_input` (it passed a module instead of a string name when rebuilding the pruned model, which broke checkpoint serialization), a plotting routine that reuses a single Matplotlib figure for all edges and skips already-pruned ones, and a no-op history logger that avoids writing a checkpoint on every model mutation during pruning and symbolic search.

Pruning combines an input threshold of 0.01 with attribution thresholds of 0.04 (nodes) and 0.06 (edges), and adds a hard constraint aimed at the quantum limit: a **fan-in cap of two**, which keeps only the two strongest input edges per hidden neuron. After pruning, the structure is retrained for 20 epochs, during which validation AUC rises from 0.714 to ~0.754.

The result for the reference run is very compact: of 22 input variables, only **2** survive (jet mass and multiplicity), organized into **1 sum node and 5 multiplication nodes**. This finding agrees with the feature importance (MDI/Gini) of the reference Random Forest (section 8), which independently also points to mass and multiplicity as the most predictive variables among the 22 available. This cross-check indicates that aggressive pruning preserves the relevant signal.

The notebook also runs a symbolic fit on each surviving edge, matching it against a library of candidate functions to obtain a formula-based interpretation. This is the main interpretability advantage of KANs and belongs to the project's purely classical branch. The quantum branch uses the **numerical response** of each edge instead of the symbolic formula.

<figure>
  <img src="/assets/img/investigacion/qkan/retrained_model.png" alt="Pruned and retrained KAN">
  <figcaption>Pruned and retrained KAN model: of 22 inputs, 2 survive (jet mass and multiplicity), organized into 1 sum node and 5 multiplication nodes.</figcaption>
</figure>

## 6. From the classical graph to the quantum circuit

An extractor isolates the response of each active edge by disconnecting the other inputs to its destination node, and exports a serialized graph of sum and multiplication nodes. For the reference run, that graph has **11 qubits**, **18 input edges**, **5 `IsingZZ` transfers**, and 6 output edges, a direct result of compressing 22 variables into **2 surviving inputs** distributed across **1 sum node and 5 multiplication nodes**. Qubits are counted **per surviving accumulator node**: the same input variable can be re-uploaded onto several wires if it feeds several different hidden nodes. This is why 2 variables occupy 11 qubits.

The circuit follows five design principles:

1. **Data re-uploading**: each edge's univariate function is modeled through repeated $R_y$/$R_z$ rotations of the input data, parameterized by the coefficients fitted in the warm start, rather than encoding the data once.
2. **Inherited topology**: the classical hidden layer decides which variables matter and how many qubits are used; the circuit reproduces the topology of the pruned graph.
3. **Summation**: each edge feeding a sum node chains consecutive $R_Y$ and $R_Z$ rotation gates on the same wire, so sum nodes require no two-qubit gate at all. The number of re-uploads depends on the degree of the polynomial fitted on the edge. For this work, a fixed degree of 4 was used.
4. **Multiplication**: implemented with an `IsingZZ` gate combined with a `CNOT`, the only point in the circuit that introduces entanglement between wires.
5. **Single-qubit readout**: all information collapses onto one output wire, and the prediction is the Pauli-Z expectation value of that qubit, passed through a sigmoid to obtain a class probability.

<figure>
  <img src="/assets/img/investigacion/qkan/quantum-circuit.png" alt="Quantum circuit">
  <figcaption>Quantum circuit corresponding to the pruned classical graph: 11 qubits, 18 input edges encoded by re-uploading, and 5 IsingZZ gates for the multiplication nodes.</figcaption>
</figure>

The hidden-to-output stage is a variational readout and does not literally reproduce a second KAN layer, because a hidden node's value lives in a qubit's phase and cannot be re-uploaded without an intermediate measurement. For this reason **only depth-2 networks are supported**. This is an explicit design limitation, also motivated by the *barren plateau* risk discussed in section 2, and it remains as future work.

The model can run on three simulators, representing successively more realistic versions of the same circuit: `ideal` (`lightning.qubit`, with no noise or finite sampling), `shots` (`default.qubit` with a finite number of shots, which introduces the statistical noise of a real measurement), and `noisy` (Qiskit Aer with a noise model derived from `FakeManilaV2`, which additionally simulates the decoherence and gate error of a specific IBM device). `FakeManilaV2` model `ibmq_manila`, a real device with 5 qubits, the reference circuit use 11, the first 5 are modelated with the FakeManilaV2 noise and the other are ideals.

Training uses binary cross-entropy with logits, the Adam optimizer, and a `ReduceLROnPlateau` scheduler that halves the learning rate when validation stalls. Simulation is expensive: the `noisy` backend takes on average about 2,460 s per full evaluation, versus ~130 s on `ideal`, a difference of almost 19 times. Each epoch therefore trains on a fresh random subset (~1,000 samples) and validates on a fixed subset.

## 7. Circuit initialization: the warm start

Before fine-tuning the circuit with gradient descent, the initial angles must be set. Three strategies were compared.

**Chebyshev.** Each isolated edge response is fit as $y \approx \sum_{i=0}^{N} c_i T_i(x)$ over $[-1,1]$, following the design of [Chebyshev-KAN (Sidharth et al., 2024)](#ref-sidharth-2024), and the resulting coefficients are converted into the initial rotation angles. The degree is fixed at $N=4$ for all edges. The choice of a fixed degree instead of an adaptive one comes from a bug diagnosed during the project. The original approach searched for the smallest degree that exceeded an $R^2$ threshold, but that criterion almost always chose low degrees and produced inconsistent metrics, because the circuit lost the classical structure's information. The symptom was an abrupt drop in reference AUC (from ~0.80 to 0.26-0.36) as the training set got smaller. The fit is not nested by degree: lowering the degree removes the high-order coefficients and also perturbs the low-order ones that remain. Reverting to a fixed degree restored the expected behavior, and the case is documented as one of the project's findings.

**Sine basis.** As an alternative, a fixed-frequency sinusoidal basis was implemented, following the design of [**SineKAN** (Reinhardt et al., 2025)](#ref-reinhardt-2025): $y \approx \sum_k A_k \sin(\text{freq}_k \, x + \text{phase}_k)$, with the amplitudes $A_k$ obtained via least squares over a *fixed* grid of frequencies and phases. In SineKAN, the notion of an edge's "degree" corresponds exactly to the number of sine terms summed in that grid, that is, the number of accumulated sinusoidal harmonics, in contrast with the growing-degree polynomial of Chebyshev. This basis is also the one used by Ria Khatoniar in the classical-readout branch of her GSoC 2025 project ([Khatoniar, 2025a](#ref-khatoniar-2025a), section 11), and the reference script used to faithfully port the frequency-and-phase grid construction (constants $A=0.9724$, $K=0.9884$, $C=0.9994$ from the original `SineKANLayer`) comes directly from her code.

With this fixed basis, the Chebyshev experiment was replicated at small scale, fitting real edges extracted from the pipeline with both bases and comparing their $R^2$. The result quantitatively confirms what theory suggests: the sine basis fit is moderate, with a **mean $R^2$ of approximately 0.49-0.61**, well below Chebyshev's near-perfect fit (close to 1). Without a constant term, the sine basis cannot represent static offsets and produces negative $R^2$ on some edges.

To further verify this hypothesis, we repeated only the least-squares fit, not the entire circuit training pipeline, while adding a constant term to the sine basis. The result confirms the cause: the fit's $R^2$ improves, surpassing even that of the Chebyshev basis. This isolates the cause of the original sine basis's poor fit to the missing constant term, rather than to some other issue with the least-squares procedure itself (such as ill-conditioning of the fixed frequency grid). That said, this verification was limited to the edge-level fit; we did not re-run the full circuit with this extended basis, so we do not know whether the improved $R^2$ translates into a better initial AUC for the warm start. A more accurate fit for each edge does not automatically guarantee better classification downstream in the circuit, so this remains an open question.

This finding is consistent with a recent theoretical paper on the same basis: the ["Sinusoidal Approximation Theorem for KANs" (Gleyzer et al., 2025)](#ref-gleyzer-2025) gives a constructive universal-approximation proof for sine-basis KANs, in the spirit of the original KAT. The theorem guarantees that, with enough terms and freedom to fit frequency and phase, a sine basis *can* approximate any continuous function. The result of this project shows that a fixed grid, without a constant term and with only $N=4$ harmonics, does not yet exploit that capacity.

**Random initialization.** As a control, the pruned topology is kept exactly (same qubits and connections) and each angle is sampled from $\mathcal{N}(0,1)$. This separates the value of the transferred classical knowledge from the value of the topology alone.

**Clamp in cosine.** Both the Chebyshev coefficients and the randomly initialized angles are converted into the corresponding $R_Z$ gate rotation angle by passing through an arccosine, i.e., $\theta = \arccos(\text{value})$, ensuring that the angles remain within the valid rotation range. Since arccosine is defined over $[-1,1]$, any value outside this range is clamped to the limits between [-0.9999, 0.9999]. The sine basis has its values within [-1,1], so the clamp is not applied.

## 8. Classical baseline: Random Forest

To calibrate the distance between the quantum approach and what is classically achievable, a 500-tree Random Forest with balanced class weights was added. Unlike the KAN pipeline, it receives the **full 22 variables**, unpruned. It uses the same metric keys as the KAN trainers, so all models are aggregated into a single results table.

## 9. Results

All per-run metrics are collected into a single Parquet table (74 rows across 6 different seeds). Two regimes are analyzed.

### 9.1 With mass cut, five replicas (seeds 10-14)

The following figure and table summarize the mean test AUC ($\pm$ standard deviation over 5 seeds) for the whole model chain, from the Random Forest to the untrained, randomly initialized QKAN.

<figure>
  <img src="/assets/img/investigacion/qkan/auc_mass_cut.png" alt="AUC by model, mass-cut regime">
  <figcaption>Test AUC by model in the mass-cut regime (mean $\pm$ standard deviation, $\sigma$, over 5 seeds).</figcaption>
</figure>

| Model | AUC (mean $\pm$ σ) | Accuracy (mean $\pm$ σ) | Background rejection at $\varepsilon_S=0.5$ |
|---|---|---|---|
| Random Forest (22 features) | **0.823 $\pm$ 0.004** | 0.746 $\pm$ 0.004 | 10.04 $\pm$ 0.57 |
| Classical KAN, base | 0.789 $\pm$ 0.002 | 0.712 $\pm$ 0.004 | 7.20 $\pm$ 0.37 |
| KAN, pruned + fine-tuned | 0.770 $\pm$ 0.014 | 0.696 $\pm$ 0.013 | 6.35 $\pm$ 0.75 |
| KAN, retrained / symbolic | 0.756 $\pm$ 0.004 | 0.687 $\pm$ 0.005 | 5.79 $\pm$ 0.29 |
| QKAN, trained (ideal) | 0.736 $\pm$ 0.006 | 0.563 $\pm$ 0.013 | 5.46 $\pm$ 0.14 |
| QKAN, trained (shots) | 0.732 $\pm$ 0.006 | 0.571 $\pm$ 0.015 | 5.42 $\pm$ 0.13 |
| QKAN, trained (noisy) | 0.730 $\pm$ 0.006 | 0.571 $\pm$ 0.016 | 5.40 $\pm$ 0.13 |
| QKAN warm start, Chebyshev | 0.698 $\pm$ 0.004 | 0.551 $\pm$ 0.015 | 4.37 $\pm$ 0.23 |
| QKAN warm start, Sine | 0.637 $\pm$ 0.015 | 0.531 $\pm$ 0.008 | 3.26 $\pm$ 0.15 |
| QKAN warm start, random | 0.488 $\pm$ 0.021 | 0.492 $\pm$ 0.014 | 1.81 $\pm$ 0.21 |

Three observations emerge from this table. First, pruning and symbolic simplification cost ~0.03 AUC relative to the base KAN, and the trained QKAN sits an additional 0.02 below the retrained KAN: the circuit reaches AUC ~0.73-0.74 using only two variables and 11 qubits. Second, **fine-tuning has a measurable effect**: training the circuit raises the Chebyshev warm start from 0.698 to 0.736 on the ideal simulator. Third, the three backends (ideal, finite-shot, and noisy) differ from each other by no more than ~0.006 AUC; within the noise model used, the circuit does not visibly degrade.

The comparison across warm-start bases confirms the order Chebyshev > Sine > Random, both in AUC and in background rejection. The next figure shows background rejection at a signal-efficiency working point of 50% ($\varepsilon_S = 0.5$).

<figure>
  <img src="/assets/img/investigacion/qkan/bkg_rejection_warmstart.png" alt="Background rejection by warm-start basis">
  <figcaption>Background rejection by warm-start basis, with no further training of the circuit.</figcaption>
</figure>

Initializing the circuit with the Chebyshev basis, with no further training, rejects roughly **2.4 times more background** than random initialization at the same signal efficiency (4.37 versus 1.81), with the sine basis at an intermediate point (3.26), consistent with its weaker edge fit. The gap grows at the more demanding working point, $\varepsilon_S=0.3$, where Chebyshev rejects about **3.6 times more background** than the random control (10.4 versus 2.9). At $\varepsilon_S=0.9$ the gap closes (1.4 versus 1.1): this is the maximum signal-efficiency regime, where any classifier, including the random one, lets almost all background through.

To verify that these differences exceed the statistical noise of only five seeds, paired $t$-tests ($\alpha=0.05$) were run, with the null hypothesis that random initialization and each warm start give the same accuracy and AUC:

| Comparison | Accuracy | AUC |
|---|---|---|
| Random vs Chebyshev | $t=-4.86$, $p=0.0082$ | $t=-20.17$, $p=3.6\times10^{-5}$ |
| Random vs Sine | $t=-5.00$, $p=0.0075$ | $t=-17.95$, $p=5.7\times10^{-5}$ |

Both null hypotheses are clearly rejected, although the result should be read with caution since it rests on only five seeds.

The quantum circuit's accuracy hovers around 0.56-0.57, versus ~0.71 for the classical KAN. The reference confusion matrix collapses toward the positive class (recall ~0.97, precision ~0.53). The ranking signal, as measured by AUC, is preserved, but the fixed threshold of 0.5 is poorly calibrated for the circuit's output. For this reason AUC is reported as the main metric instead of accuracy.

### 9.2 Almost the full dataset, no mass cut

In a second regime, practically all available events are used (with the same 10 constituents per jet, without restricting the mass window), in a single block with seed 42. With no replicas, error bars and hypothesis tests cannot be computed: the result corresponds to **a single run**.

<figure>
  <img src="/assets/img/investigacion/qkan/auc_full_dataset.png" alt="AUC by model, no-mass-cut regime">
  <figcaption>Test AUC by model in the no-mass-cut regime (single run, seed 42).</figcaption>
</figure>

| Model | AUC (single run) | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Random Forest | **0.965** | 0.909 | 0.873 | 0.957 |
| Classical KAN, base | 0.959 | 0.902 | 0.868 | 0.948 |
| KAN, pruned + fine-tuned | 0.959 | 0.903 | 0.865 | 0.954 |
| KAN, retrained / symbolic | 0.953 | 0.899 | 0.859 | 0.956 |
| QKAN, trained (ideal) | 0.904 | 0.725 | 0.651 | 0.967 |
| QKAN, trained (noisy) | 0.902 | 0.736 | 0.662 | 0.967 |
| QKAN warm start, Chebyshev (ideal, untrained) | 0.708 | 0.719 | 0.693 | 0.787 |
| QKAN warm start, Chebyshev (noisy, untrained) | 0.677 | 0.617 | 0.591 | 0.762 |

Without the mass cut, classifiers can directly exploit jet mass, so all models reach their highest discrimination. The trained QKAN maintains an AUC above 0.90, noticeably higher than in the cut regime. Its accuracy is again lower (0.72-0.74, recall ~0.97, precision ~0.65), repeating the calibration issue observed before.

## 10. Comparison with the literature

The reference paper for this benchmark, [Kasieczka et al. (2019)](#ref-kasieczka-2019), and the most-cited comparison notebook that reproduces and extends its figures, [SebastianMacaluso/TopTagComparison](#ref-macaluso), gather 12 classical and deep-learning taggers (ParticleNet, TreeNiN, ResNeXt, PFN, CNN, NSub, LBN, P-CNN, LoLa, EFN, EFP, TopoDNN) plus the GoaT meta-tagger, evaluated on the full dataset of 2 million jets with no mass restriction. In the original table ("single model"), AUC ranges from **0.955** (LDA, the weakest tagger, explicitly excluded from the meta-tagger for contributing no signal) to **0.985** (ParticleNet); excluding LDA, the range is 0.972 (TopoDNN) to 0.985 (ParticleNet). Background rejection at $\varepsilon_S=0.3$ ranges, in the same table, from 295 ± 14 (LDA) to 1412 ± 46 (ParticleNet).

This comparison requires an explicit caveat: **those numbers use the full dataset, with no cuts**, while most of this project's results with replicas and error bars correspond to the aggressive mass-cut regime (145-205 GeV), designed to remove the easiest cue and force models to rely on substructure. The figures are therefore not comparable point by point. The most reasonable common ground is the no-mass-cut regime (section 9.2): there, the base classical KAN reaches AUC 0.959 and the Random Forest 0.965, in the neighborhood of the lower end of the literature, though still below specialized architectures such as ParticleNet. The trained QKAN in this regime reaches AUC 0.904, a notable value for an 11-qubit circuit derived from just two variables, though clearly below both the project's classical KAN and the taggers in the literature.

Beyond the mass cut, two factors widen this gap. The first is variable compression: pruning reduces 22 variables to just 2, while architectures such as ParticleNet or IAFormer consume the full constituent cloud with graphs or sparse attention designed for that high dimensionality. The second, identified when revisiting the preprocessing after obtaining the main results, is that **each jet carries up to 200 constituents and the pipeline only uses the 10 most energetic**. This deliberate simplification keeps the circuit simulable, but it leaves out almost all low-energy substructure, precisely where architectures such as [IAFormer (Esmail et al., 2026)](#ref-esmail-2026) or the Lorentz-equivariant [L-GATr (Brehmer et al., 2025)](#ref-brehmer-2025) report gains. Together with the mass cut and the compression to two variables, it is a third reason for the gap with the state of the art.

## 11. Related work: other quantum-KAN approaches

Other efforts combine KANs with quantum computing. This section situates the project against three of them, each of which addresses the same bottleneck (how to quantize a learnable univariate function) with a different strategy.

**QKAN ([Ivashkov et al., 2024](#ref-ivashkov-2024))** is the theoretical proposal that motivated, from the outset, the search for an alternative that could run on noisy hardware (section 2). It implements a KAN's univariate functions through *block encoding* and the Quantum Singular Value Transformation (QSVT). This gives it strong expressivity guarantees, but ties it to fault-tolerant primitives that no current NISQ device can run natively. It represents the theoretical ceiling against which any QKAN designed for current hardware is measured: it offers stronger guarantees, but it cannot be executed today.

**Ria Khatoniar's QKAN project ([Khatoniar, 2025a](#ref-khatoniar-2025a), [2025b](#ref-khatoniar-2025b); GSoC 2025, also at ML4SCI)** is the closest precedent in time and the one that directly inspired the choice of the sine basis in this work. Her first report describes a hybrid QKAN combining QSVT encoding, a quantum linear combination of unitaries (LCU), and a Hadamard test for summation, with a classical KAN/SineKAN-style readout; it explicitly reports that the architecture could not scale beyond a certain point on the quark-gluon dataset, due to simulator memory failures (*kernel crashes*). Her second report presents a fully quantum KAN based on a Quantum Circuit Born Machine (QCBM), with label and position qubits, an entangling layer (*LabelMixer*), and Pauli-Z/X readout; there she states that extending the approach to quark-gluon tagging or jet-mass prediction "could not yet be achieved, mainly because the current architecture is computationally slow", limiting it to simulators. Both reports conclude by signaling an intent to continue in the future. This project picks up that direction in choosing the sinusoidal basis for the warm start, using the `SineKANLayer` code from her repository as a direct reference.

**[QuKAN (Werner et al., 2025)](#ref-werner-2025)** explores an approach different from that of Ivashkov et al. despite the similar name: instead of encoding each univariate function through QSVT, it directly uses a Quantum Circuit Born Machine as a generative mechanism to represent them, quantum from the start and without distilling an already-trained classical KAN.

Taken together, the four projects (Ivashkov et al., Khatoniar, Werner et al., and this work) show a clear pattern: scaling a quantum KAN beyond a handful of variables is, in 2025-2026, a shared open problem. The causes are the reliance on fault-tolerant primitives (Ivashkov et al.), simulation memory (Khatoniar), the cost of a generative quantum mechanism (Werner et al.), or, in this case, the need to prune aggressively before the circuit can be simulated. None of the four has yet demonstrated a QKAN running without compromises on a full-scale HEP dataset.

## 12. Conclusions

Five conclusions summarize the project.

1. **Classical pruning works as a qubit filter.** The KAN reduces 22 inputs to two variables and 11 qubits, and the circuit still reaches an AUC of ~0.73 (with mass cut) and ~0.90 (without).
2. **The warm start carries real information.** The order Chebyshev > Sine > Random is consistent across seeds and statistically significant over five replicas. Random initialization performs at chance level (AUC ~0.49), showing that topology alone is not enough. The Chebyshev basis is superior because its per-edge fit is much closer to the real classical response ($R^2$ close to 1, versus 0.49-0.61 for the sine basis).
3. **Simulated noise has a small effect.** The gap between the `ideal` and `noisy` backends is at most ~0.006 AUC in the mass-cut regime, and ~0.002 in the no-cut regime. The result is encouraging, but it comes from a simulated noise model and **still needs to be validated on real hardware**. In addition, simulating it is costly: evaluating the `noisy` backend takes on average ~19 times longer than `ideal`.
4. **The mass cut defines the hardest problem.** With it, AUC drops from ~0.96 to 0.82 for the Random Forest, and from 0.90 to 0.73 for the QKAN: removing the easy mass cue forces the models to rely on substructure, which the two surviving variables capture only partially.
5. **The classical ceiling is set by the Random Forest.** It has access to all 22 variables and leads in both regimes; the quantum model does not surpass it. The value of the hybrid approach lies in producing a compact, interpretable circuit that preserves a considerable fraction of that performance.

## 13. Contributions

- A reproducible, regime-aware pipeline: the different regimes are encoded in the directory structure, replica subsets are class-balanced, each stage is idempotent via checkpoints, and results across several seeds are collected into a single Parquet table.
- `HEPKAN`, a subclass of `pykan` that fixes a serialization bug in input pruning and reduces plotting and logging overhead.
- A pruning rule designed specifically for quantum limits, combining attribution thresholds with a hard fan-in cap.
- A classical-to-quantum bridge: the extractor converts a pruned KAN into a sum/multiplication graph, and the QKAN builder converts that graph into a PennyLane circuit.
- Two interchangeable bases plus a control, allowing the value of the warm start to be measured across three simulation backends, evaluated before and after training.
- A diagnosed and fixed bug: the adaptive Chebyshev degree search was silently collapsing the reference AUC on smaller datasets; the cause was identified and replaced with a fixed degree.
- A classical benchmark, the Random Forest, evaluated with the same metrics and the same data split.
- An exploratory analysis documenting the dataset layout, the $\Delta R$ artifact, and the choice of 10 constituents and 22 inputs.

## 14. Limitations and future work

- **Few replicas**: the statistical tests use five seeds, and the no-mass-cut regime is a single run with no estimated uncertainty.
- **Extreme compression**: pruning leaves ~2 input variables; relaxing it pushes the qubit count beyond what is simulable in reasonable time.
- **Only the 10 most energetic constituents out of up to 200 per jet**: an additional restriction that likely discards relevant low-energy substructure.
- **Calibration**: quantum accuracy and precision are weak even where AUC is good; a tuned decision threshold remains to be studied.
- **Sine basis**: adding a constant term to the least-squares fit raises its $R^2$ above that of Chebyshev, confirming that the lack of this term was the cause of the original poor $R^2$. However, this verification was limited to the edge-level fit: the full circuit with the extended basis still needs to be run to determine whether this better fit actually translates into a higher warm start AUC than Chebyshev, or if something is lost in the rotation angle encoding.
- **Simulated and partial noise only**: `FakeManilaV2` approximates a real device, but it is not one; and since that device has 5 qubits compared to the 11 in the circuit, the calibrated noise only covers the first 5, the rest of the circuit runs ideally even on the noisy backend. Extending the noise model to all 11 qubits, or repeating the experiment with a reference device of comparable size, is an obvious step before drawing firm conclusions about noise tolerance, considering the significant increase in time that this fully noisy and simulated approach would entail, which remains pending.
- **Depth-2 networks only**: deeper KANs would require intermediate measurement and re-encoding. As discussed in section 2, stacking more re-uploading layers without that redesign is technically difficult and increases the barren plateau risk that the original proposal had already anticipated, so this direction depends on first solving intermediate measurement.
- **Other datasets**: a preprocessing pipeline exists for quark-gluon tagging.

In sum, the project shows that a classical KAN can decide the shape of a quantum circuit, that knowledge transferred through a good basis has a measurable effect, and that the resulting compact model preserves a useful part of the classification signal. The model does not yet match the best classical baseline, and that limit is reported alongside the results.

## Acknowledgements

I want to thank my friend Eduardo Villamil for providing computational resources for this project, ML4SCI for the opportunity to take part in Google Summer of Code 2026, and all the admins, mentors, and members of the program for their useful feedback throughout the project.

---

## References

1. <a id="ref-ariasalamo-2025"></a>Arias Alamo, D., Hernández López, S., & Lázaro González, J. (2025). Is data-reuploading really a cheat code? An experimental analysis. In *Proceedings of the 1st International Conference on Quantum Software (IQSOFT 2025)*. https://doi.org/10.5220/0013555000004525
1. <a id="ref-brehmer-2025"></a>Brehmer, J., Bresó, V., de Haan, P., Plehn, T., Qu, H., Spinner, J., & Thaler, J. (2025). *A Lorentz-equivariant transformer for all of the LHC*. SciPost Physics. https://arxiv.org/abs/2411.00446
1. <a id="ref-esmail-2026"></a>Esmail, W., Hammad, A., & Nojiri, M. (2026). IAFormer: Interaction-aware transformer network for collider data analysis. *SciPost Physics, 20*, Article 108. https://arxiv.org/abs/2505.03258
1. <a id="ref-gleyzer-2025"></a>Gleyzer, S., Nguyen, H., Ramakrishnan, D. P., & Reinhardt, E. A. F. (2025). Sinusoidal approximation theorem for Kolmogorov–Arnold networks. *Mathematics, 13*(19), Article 3157. https://doi.org/10.3390/math13193157
1. <a id="ref-ivashkov-2024"></a>Ivashkov, P., Huang, P.-W., Koor, K., Pira, L., & Rebentrost, P. (2024). *QKAN: Quantum Kolmogorov-Arnold networks with applications in machine learning and multivariate state preparation*. arXiv. https://arxiv.org/abs/2410.04435. Published in 2026 in *npj Quantum Information*. https://doi.org/10.1038/s41534-026-01202-5
1. <a id="ref-kasieczka-2019"></a>Kasieczka, G., Plehn, T., Thompson, J., & Russell, M. (2019). *Top quark tagging reference dataset* (Version v0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2603256. See also the associated paper: Kasieczka, G., et al. (2019). The Machine Learning landscape of top taggers. *SciPost Physics, 7*, 014. https://arxiv.org/abs/1902.09914
1. <a id="ref-khatoniar-2025a"></a>Khatoniar, R. (2025a). *GSoC 2025 \| Quantum Kolmogorov-Arnold networks for high energy physics analysis at the LHC (Part I)* [Blog post]. Medium. https://medium.com/@riakhatoniar1234/gsoc-2025-quantum-kolmogorov-arnold-networks-for-high-energy-physics-analysis-at-the-lhc-a98207bf6d4c
1. <a id="ref-khatoniar-2025b"></a>Khatoniar, R. (2025b). *GSoC 2025 \| Quantum Kolmogorov-Arnold networks for high energy physics analysis at the LHC (Part II)* [Blog post]. Medium. https://medium.com/@riakhatoniar1234/gsoc-2025-quantum-kolmogorov-arnold-networks-for-high-energy-physics-analysis-at-the-lhc-part-8b44f5616e6f
1. <a id="ref-liu-2024-kan"></a>Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024). *KAN: Kolmogorov-Arnold networks*. arXiv. https://arxiv.org/abs/2404.19756
1. <a id="ref-liu-2024-kan2"></a>Liu, Z., Ma, P., Wang, Y., Matusik, W., & Tegmark, M. (2024). *KAN 2.0: Kolmogorov-Arnold networks meet science*. arXiv. https://arxiv.org/abs/2408.10205
1. <a id="ref-macaluso"></a>Macaluso, S. (n.d.). *TopTagComparison* [Code repository]. GitHub. Retrieved September 21, 2026, from https://github.com/SebastianMacaluso/TopTagComparison
1. <a id="ref-reinhardt-2025"></a>Reinhardt E, Ramakrishnan D and Gleyzer S (2025) SineKAN: Kolmogorov-Arnold Networks using sinusoidal activation functions. Front. Artif. Intell. 7:1462952. doi: 10.3389/frai.2024.1462952
1. <a id="ref-sidharth-2024"></a>Sidharth, S. S., Gokul, R., Anas, K. P., & Keerthana, A. R. (2024). *Chebyshev polynomial-based Kolmogorov-Arnold networks: An efficient architecture for nonlinear function approximation*. arXiv. https://arxiv.org/abs/2405.07200
1. <a id="ref-werner-2025"></a>Werner, Y., Malemath, A., Liu, M., Fortes Rey, V., Palaiodimopoulos, N., Lukowicz, P., & Kiefer-Emmanouilidis, M. (2025). QuKAN: A quantum circuit Born machine approach to quantum Kolmogorov Arnold networks. *Scientific Reports, 15*. https://doi.org/10.1038/s41598-025-22705-9
{: .references}

*Dataset: [Zenodo record 2603256](https://zenodo.org/records/2603256) (top tagging, Pythia8 + Delphes ATLAS). This project builds on the original proposal submitted to GSoC 2026 / ML4SCI, "Quantum Sine-Kolmogorov-Arnold Networks for High Energy Physics Analysis" (Toral, J., 2026), described in section 2.*
