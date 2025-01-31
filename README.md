# GPT2-JAX

[![Project Status: Working](https://img.shields.io/badge/status-working-brightgreen.svg)](https://github.com/yourusername/Audio-VQVAE-for-JAX)

A JAX-based implementation of GPT2, heavily based on HF's GPT2. This model is part of the JAXTTS (eXtended Text-To-Speech) series, where I rewrite XTTS in JAX to understand how it works from A to Z, and learn JAX along the way.

## Overview

This project leverages **JAX** and **Equinox** to develop a GPT2 model.

## 🚎 Roadmap

- [x] Have a working version of GPT2
- [ ] Add validation loss
- [ ] Add type annotations with jaxtyping
- [ ] Add docstrings
- [ ] Speed comparison with torch HF version
- [x] KV caching
- [ ] Inference temperature, beam, topk and topp


## Getting Started

```bash
git clone git@github.com:TugdualKerjan/gpt2-jax.git
cd gpt2-jax
uv sync
uv add jax["cuda"] # JAX has various versions optimized for the underlying architecture
uv run train_gpt2.py
```