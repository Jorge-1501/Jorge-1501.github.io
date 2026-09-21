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
# Quantum Kolmogorov-Arnold Networks for Top Quark Jet Tagging

*A technical summary of the QKAN project, Google Summer of Code 2026 at ML4SCI*

This project was developed by Jorge Toral and can be found in this [GitHub repository](https://github.com/Jorge-1501/QKANs-ML4SCI_2026)

---

## 1. The problem and the core idea

At the Large Hadron Collider, a top quark decays almost instantaneously and produces a **jet** (a collimated spray of particles) with a characteristic internal substructure, distinct from that of an ordinary jet originating from a light quark or a gluon (QCD). Distinguishing these two types of jet, **top tagging**, is a well-studied binary classification problem, with classical reference architectures reaching AUCs above 0.96-0.98 on the public dataset used in this project ([Kasieczka et al., 2019](#ref-kasieczka-2019)): jets simulated at 14 TeV with Pythia8 and an ATLAS-like Delphes detector card, reconstructed with anti-$k_T$ and $R=0.8$ in the range $p_T \in [550, 650]$ GeV.

This project does not compete to beat those architectures. Its question is different: **can a variational quantum neural network (VQC) perform this task, and what role can a classical network play in making it feasible?**

The underlying limitation is simple: the cost of simulating a quantum circuit grows as $O(2^Q)$ with the number of qubits $Q$. Feeding a jet with dozens of variables directly into a VQC is impractical. The solution explored here uses a classical **Kolmogorov-Arnold Network (KAN)** as a preprocessor and qubit filter: it is trained, pruned down to a small, interpretable topology, and that pruned topology, not an arbitrary design, decides how many qubits the quantum circuit needs and how they are connected. The classical model determines the quantum resource, rather than leaving it to trial and error.

The full pipeline, documented across three notebooks (`EDA_top.ipynb`, `Training_process.ipynb`, `Results.ipynb`), preprocesses the jets into balanced replicas, trains and prunes the KAN, extracts each surviving edge into a compact basis (Chebyshev or sine) as a *warm start*, fine-tunes the QKAN on an ideal simulator, on one with finite-shot noise, and on one with hardware noise, and compares everything against a classical Random Forest.

## 2. Why a KAN and not an MLP?

Kolmogorov-Arnold Networks, proposed by [Liu et al. (2024)](#ref-liu-2024-kan) building on the Kolmogorov-Arnold representation theorem (KAT), invert the design of a traditional multilayer perceptron. In an MLP, the activation functions are fixed (ReLU, tanh, ...) and live on the nodes; the weights, which are learned, are scalars on the edges. In a KAN it's the other way around: **each edge carries its own learnable univariate function**, parameterized as a spline, and the nodes only sum. The KAT theorem guarantees that any continuous multivariate function can be written as

$$f(x_1, \dots, x_n) = \sum_{q=1}^{2n+1} \Phi_q\left(\sum_{p=1}^{n} \phi_{q,p}(x_p)\right)$$

that is, as a composition of univariate functions and sums. This has two consequences we exploit directly. First, **interpretability**: each edge, once training is finished, can be fit symbolically against a library of candidate functions, giving a closed-form expression for what that edge "does." Second, and crucial for this project, **structured prunability**: since each edge is an independent functional unit, an attribution (how much it contributes to the output) can be measured and the edge removed if negligible, without breaking the interpretation of the rest of the network.

A follow-up work by the same authors, [KAN 2.0 (Liu et al., 2024)](#ref-liu-2024-kan2), also introduces **multiplication nodes**: the KAT theorem in its classical form only guarantees composition through sums, but letting some hidden nodes multiply their inputs instead of summing them allows variable interactions to be expressed directly. This is the idea we adopt: this project's classical architecture (`HEPKAN`, a subclass of [pykan](https://github.com/kindxiaoming/pykan)) uses a hidden layer with sum *and* multiplication nodes in parallel.

## 3. Exploring the data: what makes a top jet different

Before training anything, the EDA notebook characterizes the dataset. The HDF5 file does not document the order of its columns, so we infer it from the data: each block of four columns corresponds to one constituent particle of the jet, with energy in the first position and the three momentum components afterward, a pattern recognizable in the column means, much larger in the first column of each block of four.

Four observations emerged from the histograms and radial profiles that guided all subsequent design choices:

- **Invariant mass discriminates, but isn't enough.** $m_{jet} = \sqrt{E^2 - p_x^2 - p_y^2 - p_z^2}$ has a clear peak in the signal around the top quark mass (~173 GeV), while the background is wider. The two distributions overlap by sections, so we selected the **145-205 GeV** window for the main experiments: within it, the trivial mass cue is largely removed and classifiers must rely on subtler information.
- **Top jets are more populated**: multiplicity (number of constituents) is systematically higher and more spread out than in QCD, consistent with a three-body decay for this selected mass window.
- **Top jets are more diffuse**: the radial energy profile and the cumulative $p_T$ fraction grow more gradually in tops than in QCD, where energy is more concentrated near the jet axis.
- $\eta$-$\phi$ scatter plots were inconclusive due to noise and scale.

We also detected an artifact: the $\Delta R$ distribution does not show the sharp cutoff at 0.8 that it should, due to a small constant added to avoid division by zero. We filtered out constituents outside the jet radius to correct this.

From this we built the input representation, combining **global** variables (jet mass and multiplicity, counting only constituents with energy above $10^{-8}$) with **local** variables for each retained constituent ($\Delta R$ to the jet axis and $p_{T,\text{rel}}$). Using the 80% cumulative $p_T$ criterion, we estimated that around 15 constituents would suffice to capture most of the energy; the final pipeline, however, uses only **the 10 most energetic constituents per jet**, for a total of **22 inputs** (2 global + 10 pairs of local variables). Each original jet carries up to 200 stored constituents; the cut to 10 is a deliberate compression decision we revisit in the results and conclusions, because the model never sees 95% of the particles the detector recorded.

The local variables already live in $(0,1)$; for the global ones we apply a logarithmic transform followed by tanh normalization, deliberately keeping the outliers because the centers of the two class distributions are similar and it is the tail that carries discriminative information.

Finally, since the number of events in the mass window differs between classes (there are more tops), we subsample the majority class to balance it, and split the result into **5 disjoint, class-balanced subsets**. Each seed selects one (`seed % 5`), so several seeds give independent end-to-end replicas rather than a single point estimate.

## 4. The classical architecture and its pruning

The classical model has architecture `[22, [9, 9], 1]`: 22 inputs, one hidden layer with up to 9 sum nodes and 9 multiplication nodes in parallel, and a scalar output. The B-splines use degree $k=3$ and grid size 5, for a total of **8,568 parameters**. In the reference run (seed 10), the base model reached a test AUC of **0.792**, stopping early around epoch 20 of the planned 60.

`HEPKAN` introduces three practical modifications over `pykan`, motivated by the project's limited computational resources: a fix for a bug in `prune_input` (it passed a module instead of a string name when rebuilding the pruned model, breaking checkpoint serialization), a plotting routine that reuses a single Matplotlib figure for all edges and skips already-pruned ones, and a no-op history logger that avoids writing a checkpoint on every model mutation during pruning and symbolic search.

Pruning combines an input threshold of 0.01 with attribution thresholds of 0.04 (nodes) and 0.06 (edges), and adds a hard constraint aimed at the quantum limit: a **fan-in cap of two**, which keeps only the two strongest input edges per hidden neuron. After pruning, the structure is retrained for 20 epochs, during which validation AUC rises from 0.714 to ~0.754.

The result, for the reference run, is surprisingly compact: of 22 input variables, only **2** survive (jet mass and multiplicity), organized into **1 sum node and 5 multiplication nodes**. This finding feeds back into the feature importance (MDI/Gini) of the reference Random Forest (section 7), which independently also points to mass and multiplicity as the most predictive among the 22 available: cross-validation that aggressive pruning does not discard relevant signal.

The notebook also runs a symbolic fit on each surviving edge, matching it against a library of candidate functions to obtain a formula-based interpretation: the main interpretability advantage of KANs, and part of the project's purely classical branch. The quantum branch, by contrast, does not use this symbolic formula but rather the **numerical response** of each edge.

## 5. From the classical graph to the quantum circuit

An extractor isolates the response of each active edge by disconnecting the other inputs to its destination node, and exports a serialized graph of sum and multiplication nodes. For the reference run, that graph has **11 qubits**, 18 input edges, 5 `IsingZZ` transfers, and 6 output edges. It's important to note that qubits are counted **per surviving accumulator node, not per input**: the same input variable can be re-uploaded onto several wires if it feeds several different hidden nodes.

The circuit follows five design principles:

1. **Data re-uploading**: each edge's univariate function is modeled through repeated rotations of the input data, rather than encoding it once.
2. The classical hidden layer decides which variables matter and how many qubits are used, the circuit inherits the topology.
3. **Summation is free**: consecutive $R_Z$ rotations on the same wire accumulate their angles, so sum nodes require no two-qubit gate at all.
4. **Multiplication** is implemented with an `IsingZZ` gate combined with a `CNOT`.
5. All information collapses onto a single output wire, and the prediction is the Pauli-Z expectation value of a single qubit.

The hidden-to-output stage is a variational readout, not a literal second KAN layer, because a hidden node's value lives in a qubit's phase and cannot be re-uploaded without an intermediate measurement. For this reason, **only depth-2 networks are supported**, an explicit design limitation, which we leave as future work.

The model can run on three simulators: `ideal` (`lightning.qubit`), `shots` (`default.qubit` with a finite number of shots), and `noisy` (Qiskit Aer with a noise model derived from `FakeManilaV2`). Training uses binary cross-entropy with logits, the Adam optimizer, and a `ReduceLROnPlateau` scheduler that halves the learning rate when validation stalls. Since simulation is expensive, each epoch trains on a fresh random subset (~1,000 samples) and validates on a fixed subset.

## 6. Three ways to initialize the circuit: the warm start

Before fine-tuning the circuit with gradient descent, we need to decide what angles to start from. We compare three strategies.

**Chebyshev**, following the design of [Chebyshev-KAN (Sidharth et al., 2024)](#ref-sidharth-2024). Each isolated edge response is fit as $y \approx \sum_{i=0}^{N} c_i T_i(x)$ over $[-1,1]$, and the resulting coefficients are converted into the initial rotation angles. The degree is fixed at $N=4$ for all edges. This decision, a fixed degree instead of an adaptive one, comes from a real bug diagnosed during the project: we originally searched for the smallest degree that exceeded an $R^2$ threshold, but that criterion almost always chose low degrees and produced inconsistent metrics, because the circuit lost the classical structure's information. The symptom was an abrupt drop in reference AUC (from ~0.80 to 0.26-0.36) as the training set got smaller; reverting to a fixed degree restored the expected behavior.

**Sine basis.** As an alternative we implemented a fixed-frequency sinusoidal basis, following the design of [**SineKAN** (Reinhardt et al., 2024)](#ref-reinhardt-2024): $y \approx \sum_k A_k \sin(\text{freq}_k \, x + \text{phase}_k)$, with the amplitudes $A_k$ obtained via least squares over a *fixed*, not learned, grid of frequencies and phases. In SineKAN, the notion of an edge's "degree" corresponds exactly to the number of sine terms summed in that grid: there's no growing-degree polynomial as in Chebyshev, just more accumulated sinusoidal harmonics. This basis is, moreover, the one used by Ria Khatoniar herself in the classical-readout branch of her own GSoC 2025 project ([Khatoniar, 2025a](#ref-khatoniar-2025a), section 8), and the reference script we used to faithfully port the frequency-and-phase grid construction (constants $A=0.9724$, $K=0.9884$, $C=0.9994$ from the original `SineKANLayer`) comes directly from her code.

With this fixed basis we replicated the Chebyshev experiment at small scale, fitting real edges extracted from the pipeline with both bases and comparing their $R^2$. The result quantitatively confirms what theory suggests: the sine basis fit is moderate, with a **mean $R^2$ of approximately 0.49-0.61**, well below Chebyshev's near-perfect fit (close to 1). Without a constant term, the sine basis cannot represent static offsets and produces negative $R^2$ on some edges. Adding a constant term to the basis is, given this evidence, the obvious next step.

It's worth situating this finding against a more recent theoretical paper on this same basis: [the "Sinusoidal Approximation Theorem for KANs" (Gleyzer et al., 2025)](#ref-gleyzer-2025) gives a constructive universal-approximation proof for sine-basis KANs, in the spirit of the original KAT. This does not contradict what we observed: it guarantees that, with enough terms and freedom to fit frequency and phase, a sine basis *can* approximate any continuous function; our result shows that a fixed grid, without a constant term and with only $N=4$ harmonics, does not yet exploit that capacity.

**Random initialization.** As a control, we keep exactly the pruned topology (same qubits and connections) but sample each angle from $\mathcal{N}(0,1)$. This isolates the value of the transferred classical knowledge from the value of the topology alone.

## 7. The classical baseline: Random Forest

To calibrate how far the quantum approach is from what's classically achievable, we add a 500-tree Random Forest with balanced class weights. Unlike the KAN pipeline, it receives the **full 22 variables**, unpruned. It uses the same metric keys as the KAN trainers, so all models are aggregated into a single results table.

## 8. Results

All per-run metrics are collected into a single Parquet table (74 rows across 6 different seeds). We analyze two regimes.

### 8.1 With mass cut, five replicas (seeds 10-14)

The following figure summarizes the mean test AUC and its standard deviation over 5 seeds for the whole model chain, from the Random Forest to the untrained, randomly initialized QKAN:

![AUC by model, mass-cut regime](/assets/img/investigacion/qkan/auc_mass_cut.png)

Three observations emerge from this table. First, pruning and symbolic simplification cost ~0.03 AUC relative to the base KAN, and the trained QKAN sits an additional 0.02 below the retrained KAN: the circuit reaches AUC ~0.73-0.74 using only two variables and 11 qubits. Second, **fine-tuning does matter**: training the circuit raises the Chebyshev warm start from 0.698 to 0.736 on the ideal simulator. Third, the three backends (ideal, finite-shot, and noisy) differ from each other by no more than ~0.006 AUC; within our noise model, the circuit does not visibly degrade.

The comparison across warm-start bases confirms the order Chebyshev > Sine > Random, both in AUC and in background rejection. At a signal-efficiency working point of 50% ($\varepsilon_S = 0.5$), we measured:

![Background rejection by warm-start basis](/assets/img/investigacion/qkan/bkg_rejection_warmstart.png)

That is, initializing the circuit with the Chebyshev basis, with no further training, rejects roughly 3 times more background than random initialization at the same signal efficiency, with the sine basis landing at an intermediate point, consistent with its weaker edge fit.

To check that these differences are not statistical noise over just five seeds, we ran paired $t$-tests ($\alpha=0.05$), with the null hypothesis that random initialization and each warm start give the same accuracy and AUC:

| Comparison | Accuracy | AUC |
|---|---|---|
| Random vs Chebyshev | $t=-4.86$, $p=0.0082$ | $t=-20.17$, $p=3.6\times10^{-5}$ |
| Random vs Sine | $t=-5.00$, $p=0.0075$ | $t=-17.95$, $p=5.7\times10^{-5}$ |

Both null hypotheses are clearly rejected, though we read this result with caution since it rests on only five seeds.

One more point: the quantum circuit's accuracy hovers around only 0.56-0.57, versus ~0.71 for the classical KAN. The reference confusion matrix collapses toward the positive class (recall 0.98, precision 0.53). The ranking signal, as measured by AUC, survives, but the fixed threshold of 0.5 is poorly calibrated for the circuit's output, which is why we report AUC, not accuracy, as the main metric.

### 8.2 Almost the full dataset, no mass cut

In a second regime we use practically all available events (with the same 10 constituents per jet, but without restricting the mass window), in a single block, seed 42. With no replicas here, we cannot compute error bars or hypothesis tests: the result should be read as **a single run**.

![AUC by model, full-dataset regime](/assets/img/investigacion/qkan/auc_full_dataset.png)

Without the mass cut, classifiers can directly exploit jet mass, so all models reach their highest discrimination, and the trained QKAN maintains an AUC above 0.90, noticeably higher than in the cut regime, although, again, its accuracy is lower (0.72-0.74, recall 0.97, precision ~0.65), repeating the same calibration issue observed before.

## 9. How far are we from the literature?

It's natural to ask how these numbers compare to the published state of the art for the same dataset. The most-cited comparison notebook for this benchmark is [SebastianMacaluso/TopTagComparison](#ref-macaluso), which gathers 14 classical and deep-learning taggers (ParticleNet, TreeNiN, ResNeXt, PFN, CNN, NSub, LBN, P-CNN, LoLa, EFN, EFP, TopoDNN, among others) on the full 404k-event dataset, with no mass restriction, with AUCs between 0.967 and 0.985 and background rejection at $\varepsilon_S=0.3$ ranging from 295.2 (TopoDNN) to 1298.5 (ParticleNet).

This comparison requires an explicit caveat: **those numbers use the full dataset, with no cuts**, while most of our results with replicas and error bars correspond to the aggressive mass-cut regime (145-205 GeV), designed to remove the easiest cue and force models to rely on substructure. They are, therefore, not directly comparable point by point. The most reasonable common ground is our no-mass-cut regime (section 8.2): there, the base classical KAN reaches AUC 0.959 and the Random Forest 0.965, in the neighborhood of the lower end of the literature, though still below specialized architectures like ParticleNet. The trained QKAN in this regime reaches AUC 0.904, notable for an 11-qubit circuit derived from just two variables, but clearly below both our classical KAN and the taggers in the literature.

Two additional factors, beyond the mass cut, widen this gap. The first is variable compression: we go from 22 to just 2 after pruning, while architectures like ParticleNet or IAFormer consume the full constituent cloud with graphs or sparse attention designed for that high dimensionality. The second, evident only when revisiting the preprocessing after having the main results, is that **each jet carries up to 200 constituents and our pipeline only uses the 10 most energetic**. This is a deliberate simplification to keep the circuit simulable, but it leaves out almost all low-energy substructure, precisely where architectures like [IAFormer (Esmail et al., 2026)](#ref-esmail-2026) or [L-GATr (Brehmer et al., 2025)](#ref-brehmer-2025), Lorentz-equivariant, report gains. This is, together with the mass cut and the compression to two variables, a third legitimate reason for the gap with the state of the art.

## 10. Related work: other quantum-KAN approaches

This project is not the only effort combining KANs with quantum computing, and it's worth situating it against two others.

**Ria Khatoniar's QKAN project ([Khatoniar, 2025a](#ref-khatoniar-2025a), [2025b](#ref-khatoniar-2025b); GSoC 2025, also at ML4SCI)** is the closest precedent, and the one that directly inspired the choice of the sine basis in this work. Her first report describes a hybrid QKAN combining QSVT encoding, a quantum linear combination of unitaries (LCU), and a Hadamard test for summation, with a classical KAN/SineKAN-style readout; it explicitly reports that the architecture could not scale beyond a certain point on the quark-gluon dataset, due to simulator memory failures (*kernel crashes*). Her second report presents a fully quantum KAN based on a Quantum Circuit Born Machine (QCBM), with label and position qubits, an entangling layer (*LabelMixer*), and Pauli-Z/X readout; there she states that extending the approach to quark-gluon tagging or jet-mass prediction "could not yet be achieved, mainly because the current architecture is computationally slow," limiting it to simulators. Both reports conclude the program by signaling an intent to continue in the future. This project picks up that direction in choosing the sinusoidal basis for the warm start, using the `SineKANLayer` code from her own repository as a direct reference.

**[QuKAN (Werner et al., 2025)](#ref-werner-2025)** explores a different approach: instead of distilling an already-trained classical KAN into a circuit, it directly uses a Quantum Circuit Born Machine as a generative mechanism for the KAN's univariate functions, quantum from the start.

Taken together, the three projects leave a clear pattern: scaling a quantum KAN beyond a handful of variables is, in 2025-2026, a shared open problem, whether due to simulation memory (Khatoniar, [2025a](#ref-khatoniar-2025a), [2025b](#ref-khatoniar-2025b)), the cost of a generative quantum mechanism ([Werner et al., 2025](#ref-werner-2025)), or, here, the need to prune aggressively before the circuit can even be simulated.

## 11. Conclusions

Five conclusions summarize the project.

1. **Classical pruning works as a qubit filter.** The KAN reduces 22 inputs to two variables and 11 qubits, and the circuit still reaches an AUC of ~0.73 (with mass cut) and ~0.90 (without).
2. **The warm start carries real information.** The order Chebyshev > Sine > Random is consistent across seeds and statistically significant over five replicas. Random initialization performs at chance level (AUC ~0.49), showing that topology alone is not enough. The Chebyshev basis is better because its per-edge fit is much closer to the real classical response ($R^2$ close to 1, versus 0.49-0.61 for the sine basis).
3. **Simulated noise has a small effect.** The gap between the `ideal` and `noisy` backends is at most ~0.006 AUC in the mass-cut regime, and ~0.001 after training in the no-cut regime. Encouraging, but this is a simulated noise model, **not a result on real hardware**.
4. **The mass cut is the hardest problem.** With it, AUC drops from ~0.96 to 0.82 for the Random Forest, and from 0.90 to 0.73 for the QKAN: removing the easy mass cue forces the models to rely on substructure, which only two surviving variables can partially capture.
5. **The classical ceiling is set by the Random Forest.** It has access to all 22 variables and leads in both regimes; the quantum model does not surpass it. The value of the hybrid approach lies in producing a compact, interpretable circuit that preserves a considerable fraction of that performance, not in an accuracy gain over the classical model.

## 12. Contributions

- A reproducible, regime-aware pipeline: the different regimes are encoded in the directory structure, replica subsets are class-balanced, each stage is idempotent via checkpoints, and results across several seeds are collected into a single Parquet table.
- `HEPKAN`, a subclass of `pykan` that fixes a serialization bug in input pruning and reduces plotting/logging overhead.
- A pruning rule designed specifically for quantum limits, combining attribution thresholds with a hard fan-in cap.
- A classical-to-quantum bridge: the extractor converts a pruned KAN into a sum/multiplication graph, and the QKAN builder converts that graph into a PennyLane circuit.
- Two interchangeable bases plus a control, allowing the value of the warm start to be measured across three simulation backends, evaluated before and after training.
- A diagnosed and fixed bug: the adaptive Chebyshev degree search was silently collapsing the reference AUC on smaller datasets; we found the cause and replaced it with a fixed degree.
- A classical benchmark, the Random Forest, evaluated with the same metrics and the same data split.
- An exploratory analysis documenting the dataset layout, the $\Delta R$ artifact, and the choice of 10 constituents and 22 inputs.

## 13. Limitations and future work

- **Few replicas**: the statistical tests use five seeds, and the no-mass-cut regime is a single run with no estimated uncertainty.
- **Extreme compression**: pruning leaves ~2 input variables; relaxing it pushes the qubit count beyond what's simulable in reasonable time.
- **Only the 10 most energetic constituents out of up to 200 per jet**, an additional restriction that likely discards relevant low-energy substructure.
- **Calibration**: quantum accuracy and precision are weak even where AUC is good; a properly tuned decision threshold still needs to be studied.
- **Sine basis**: needs a constant term or more frequencies before it can be fairly judged against Chebyshev.
- **Simulated noise only**: `FakeManilaV2` approximates a real device, but is not one.
- **Depth-2 networks only**: deeper KANs would require intermediate measurement and re-encoding.
- **Other datasets**: a preprocessing pipeline exists for quark-gluon tagging, and Higgs detection is only mentioned as a future reference, with no development.

In sum, the project shows that a classical KAN can decide the shape of a quantum circuit, that knowledge transferred through a good basis matters, and can be measured, not just assumed; and that the resulting compact model preserves a useful part of the classification signal. It does not yet match the best classical baseline, and that limit is reported alongside the results.

## Acknowledgements

I want to thank my friend Eduardo Villamil for providing computational resources for this project, ML4SCI for the opportunity to take part in Google Summer of Code 2026, and all the admins, mentors, and members of the program for their useful feedback throughout the project.

---

## References

- <a id="ref-brehmer-2025"></a>Brehmer, J., Bresó, V., de Haan, P., Plehn, T., Qu, H., Spinner, J., & Thaler, J. (2025). *A Lorentz-equivariant transformer for all of the LHC*. SciPost Physics. https://arxiv.org/abs/2411.00446
- <a id="ref-esmail-2026"></a>Esmail, W., Hammad, A., & Nojiri, M. (2026). IAFormer: Interaction-aware transformer network for collider data analysis. *SciPost Physics, 20*, Article 108. https://arxiv.org/abs/2505.03258
- <a id="ref-gleyzer-2025"></a>Gleyzer, S., Nguyen, H., Ramakrishnan, D. P., & Reinhardt, E. A. F. (2025). Sinusoidal approximation theorem for Kolmogorov–Arnold networks. *Mathematics, 13*(19), Article 3157. https://doi.org/10.3390/math13193157
- <a id="ref-kasieczka-2019"></a>Kasieczka, G., Plehn, T., Thompson, J., & Russell, M. (2019). *Top quark tagging reference dataset* (Version v0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2603256
- <a id="ref-khatoniar-2025a"></a>Khatoniar, R. (2025a). *GSoC 2025 \| Quantum Kolmogorov-Arnold networks for high energy physics analysis at the LHC (Part I)* [Blog post]. Medium. https://medium.com/@riakhatoniar1234/gsoc-2025-quantum-kolmogorov-arnold-networks-for-high-energy-physics-analysis-at-the-lhc-a98207bf6d4c
- <a id="ref-khatoniar-2025b"></a>Khatoniar, R. (2025b). *GSoC 2025 \| Quantum Kolmogorov-Arnold networks for high energy physics analysis at the LHC (Part II)* [Blog post]. Medium. https://medium.com/@riakhatoniar1234/gsoc-2025-quantum-kolmogorov-arnold-networks-for-high-energy-physics-analysis-at-the-lhc-part-8b44f5616e6f
- <a id="ref-liu-2024-kan"></a>Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y., & Tegmark, M. (2024). *KAN: Kolmogorov-Arnold networks*. arXiv. https://arxiv.org/abs/2404.19756
- <a id="ref-liu-2024-kan2"></a>Liu, Z., Ma, P., Wang, Y., Matusik, W., & Tegmark, M. (2024). *KAN 2.0: Kolmogorov-Arnold networks meet science*. arXiv. https://arxiv.org/abs/2408.10205
- <a id="ref-macaluso"></a>Macaluso, S. (n.d.). *TopTagComparison* [Code repository]. GitHub. Retrieved September 21, 2026, from https://github.com/SebastianMacaluso/TopTagComparison
- <a id="ref-reinhardt-2024"></a>Reinhardt, E. A. F., Dinesh, P. R., & Gleyzer, S. (2024). SineKAN: Kolmogorov-Arnold networks using sinusoidal activation functions. *Frontiers in Artificial Intelligence, 7*. https://doi.org/10.3389/frai.2024.1462952
- <a id="ref-sidharth-2024"></a>Sidharth, S. S., Gokul, R., Anas, K. P., & Keerthana, A. R. (2024). *Chebyshev polynomial-based Kolmogorov-Arnold networks: An efficient architecture for nonlinear function approximation*. arXiv. https://arxiv.org/abs/2405.07200
- <a id="ref-werner-2025"></a>Werner, Y., Malemath, A., Liu, M., Fortes Rey, V., Palaiodimopoulos, N., Lukowicz, P., & Kiefer-Emmanouilidis, M. (2025). QuKAN: A quantum circuit Born machine approach to quantum Kolmogorov Arnold networks. *Scientific Reports, 15*. https://doi.org/10.1038/s41598-025-22705-9
