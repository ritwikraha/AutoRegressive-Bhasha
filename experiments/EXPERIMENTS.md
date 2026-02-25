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

### 4. [Verifier-RL: GRPO with Verifiable Math Rewards](https://github.com/ritwikraha/AutoRegressive-Bhasha/blob/main/experiments/verifier_rl_grpo_math.ipynb)

- **Date:** 2026-02-25
- **Status:** Ready to run

- **Premise:**
Can Qwen3-0.6B learn to reason about math through RL using only a symbolic verifier (regex + exact match) as the reward signal — no learned reward model, no human preferences? Trained with `/no_think` to isolate the RL signal from Qwen3's built-in thinking scaffolding.

- **Math:**
Group Relative Policy Optimization (GRPO) computes advantages within a group of $G$ sampled completions per prompt:

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^{G})}{\text{std}(\{r_j\}_{j=1}^{G})}
$$

Reward is binary correctness from a symbolic verifier plus a soft format bonus:

$$
r(y) = \mathbb{1}[\text{extract}(y) = \text{truth}] + 0.1 \cdot \mathbb{1}[\texttt{\backslash boxed\{\}} \in y]
$$

- **Setup:**
Base: Qwen3-0.6B (`/no_think` mode). Task: GSM8K. Algorithm: GRPO via TRL. 250 steps, group size 4, KL $\beta = 0.04$.

- **Research Questions:**
Does verifier-RL outperform SFT? Is GRPO stable without a critic? Does reward hacking emerge with unhackable rewards?

---

### 5. [Reasoning Depth Metric: Token-Level Measurement of LLM Thought Quality](https://github.com/ritwikraha/AutoRegressive-Bhasha/blob/main/experiments/reasoning_depth_metric.ipynb)

- **Date:** 2026-02-25
- **Status:** Ready to run

- **Premise:**
Accuracy conflates "the model knew the answer" with "the model figured it out". Can we build a decomposable metric that measures reasoning quality along five dimensions: logical chaining, branching, self-correction, decomposition, and verification?

- **Math:**
The Reasoning Depth Score (RDS) decomposes reasoning into weighted, length-normalized, log-scaled dimension counts:

$$
\text{RDS}(y) = \sum_{d \in \mathcal{D}} w_d \cdot \frac{\text{count}_d(y)}{\text{len}(y)} \cdot \log(1 + \text{count}_d(y))
$$

- **Setup:**
Models: Qwen3-0.6B, Qwen3-1.7B, Gemma-3-1B-IT (+ optional Qwen3.5-35B-A3B MoE). All Qwen3 models tested with `/no_think`. Dataset: GSM8K test (150 problems). Pure inference — no training.

- **Research Questions:**
Does depth correlate with accuracy? Do larger models reason deeper or just wider? Which reasoning dimensions predict correctness most strongly? Can RDS serve as a process reward for RL?

---

**Note to self:**
Drop notebooks here. Update the log. If there is no log, there is no experiment.
