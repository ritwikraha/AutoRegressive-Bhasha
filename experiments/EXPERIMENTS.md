# Experiments

A parking lot for wild ideas, half-baked runs, and "dayumn bruh.." moments.

## Logs

### 1. [Gemma-3-1B-IT CoT Steering (CAA)](https://github.com/ritwikraha/AutoRegressive-Bhasha/blob/main/experiments/steering_gemma3_cot.ipynb)

- **Date:** 2026-02-23  
- **Status:** Completed  

- **Premise:**  
Can we induce step-by-step Chain-of-Thought (CoT) reasoning at inference time without modifying model weights by perturbing the latent trajectory?

- **Math:**  
Extracted a dense steering vector via contrastive activation addition (CAA) at the midpoint layer:

$$
\vec{v}_{\mathrm{caa}} = \frac{1}{N} \sum_{i=1}^{N} \left( \phi_{13}\left(x_i^{+}\right) - \phi_{13}\left(x_i^{-}\right) \right)
$$

Injected this vector during the autoregressive forward pass using an open-loop controller:

$$
\tilde{h}_{13} = h_{13} + \alpha \vec{v}_{\mathrm{caa}}
$$

- **Tested:**  
Swept intervention strength $\alpha \in \{0, 5, 10, 20, 35\}$ on algebra word problems.

- **Verified:**  
Successfully shifted the behavioral policy from blunt answers to verbose logical breakdowns. Verified that the optimal alignment zone is around $\alpha = 15$, and proved that a static $\alpha > 35$ pushes activations off the natural manifold, resulting in catastrophic collapse (gibberish).

---

### 2. [Gemma-3-1B-IT Dynamic SAE Steering (V2)](https://github.com/ritwikraha/AutoRegressive-Bhasha/blob/main/experiments/gemma3_sae_dynamic_steering_v2.ipynb)

- **Date:** 2026-02-24  
- **Status:** Completed  

- **Premise:**  
CAA vectors are noisy and polysemantic. Can we isolate a pure, monosemantic reasoning feature using a Sparse Autoencoder (SAE) and implement a dynamic, closed-loop controller to prevent the manifold collapse observed in V1?

- **Math:**  

Discovered the active feature by projecting the empirical CAA vector onto the SAE encoder:

$$
S = \vec{v}_{\mathrm{caa}} \cdot W_{enc}
$$

Extracted the pure decoder feature:

$$
\vec{f}_{\mathrm{reason}} = W_{dec}[:, \arg\max(S)]
$$

Implemented closed-loop dynamic scaling using an L2 norm threshold ($\tau$):

$$
\alpha_t = \alpha_{base} \cdot \min\left(1, \frac{\tau}{\lVert h_{13,t} \rVert}\right)
$$

Injected via:

$$
\tilde{h}_{13,t} = h_{13,t} + \alpha_t \vec{f}_{\mathrm{reason}}
$$

- **Tested:**  
Compared unsteered vs. steered trajectories using dynamic $\alpha_{base} = 60.0$ and mapped the high-dimensional latent states to a 2D plane via PCA.

- **Verified:**  
Verified that SAE Feature 9994 strictly maps to reasoning. Proved mathematically and visually that dynamic $\tau$-scaling preserves the structural integrity of the latent manifold while achieving a permanent, deterministic behavioral bifurcation at the logit threshold.

---

**Note to self:**  
Drop notebooks here. Update the log. If there is no log, there is no experiment.
