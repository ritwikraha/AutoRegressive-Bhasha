# Experiments

A parking lot for wild ideas, half-baked runs, and "what if we tried..." moments.

## Log


| Date | Experiment | Status | Notes |
| :--- | :--- | :--- | :--- |
| 2026-02-23 | Gemma-3-1B-IT CoT Steering (CAA) | Completed | Extracted reasoning vector from 5 contrastive pairs; injected at Layer 13; ran baseline vs. steered comparison ($\alpha=15$) and swept $\alpha$ values (0, 5, 10, 20, 35). |
| 2026-02-24 | Gemma-3-1B-IT Dynamic SAE Steering (V2) | Completed | Projected CAA vector onto SAE encoder to isolate monosemantic reasoning feature; implemented closed-loop dynamic injection at Layer 13 using L2 norm thresholding ($\tau$) to scale $\alpha$; mapped latent trajectory bifurcation via PCA. |

Note to self:Drop Notebooks here, Update the log. Keep it weird.
