# NAG Latent

A ComfyUI implementation of Normalized Attention Guidance (NAG) based on the [research paper](https://doi.org/10.48550/arXiv.2505.21179).

## Overview

This implementation adapts NAG concepts for ComfyUI workflows, operating in latent space rather than at the attention level as described in the original paper.
I am still learning to figure out how to implement it at the attention level.

## Installation

1. Download the `nag_comfyui.py` file
2. Place it in your ComfyUI `custom_nodes` directory:
   ```
   ComfyUI/custom_nodes/nag_comfyui.py
   ```
3. Restart ComfyUI

## Usage

After installation, the NAG node (with the name `NAG Latent`) will be available in your ComfyUI node browser. Connect it between the checkpoint loader and the sampler's `model` slots.

## Reference

> Chen, Dar-Yen, Hmrishav Bandyopadhyay, Kai Zou, and Yi-Zhe Song. “Normalized Attention Guidance: Universal Negative Guidance for Diffusion Models.” arXiv, June 3, 2025. [https://doi.org/10.48550/arXiv.2505.21179](https://doi.org/10.48550/arXiv.2505.21179)
