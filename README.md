# Stable Diffusion Unboxed 

A comprehensive, hands-on deep dive into the inner workings of Stable Diffusion. Rather than using high-level, one-click pipelines, this repository breaks down the model into its fundamental components to explain exactly how AI image generation works under the hood.


---

##  Features & Concepts Covered

This notebook walks through the pipeline step-by-step, including:

1. **Loading the Sub-Models:** Manually assembling the Autoencoder (VAE), Text Encoder (CLIP), UNet, and Scheduler.
2. **The Manual Diffusion Loop:** Rebuilding the text-to-image pipeline from scratch to understand iterative denoising.
3. **Translating Latents:** Building custom functions to seamlessly encode pixels into latent space and decode them back to images.
4. **Image-to-Image (Img2Img):** Injecting noise into existing images to guide structural generation.
5. **Hacking Text Embeddings:** Intercepting the CLIP text encoder to perform **Concept Mixing** (e.g., mathematically averaging a puppy and a skunk) and **Prompt Averaging**.
6. **Visualizing Denoising:** Extracting the model's intermediate $x_0$ predictions to watch the AI "think" in real-time.
7. **Loss-Based Control:** Writing custom gradient math to hijack the diffusion loop and force the UNet to obey strict rules (e.g., forcing a campfire to be blue).
8. **Latent Space Geometry:** Implementing **Spherical Linear Interpolation (SLERP)** vs. **Linear Interpolation (LERP)** to understand the "soap bubble" effect of high-dimensional Gaussian distribution and create perfectly smooth image morphing.

---

##  Prerequisites & Installation

To run this notebook, you will need a machine with a CUDA-capable GPU (or access to Google Colab/Kaggle) and the following core libraries:

* `torch` (PyTorch)
* `diffusers`
* `transformers`
* `Pillow` (PIL)
* `matplotlib`
* `tqdm`

**Quick Setup:**
```bash
pip install torch torchvision torchaudio diffusers transformers accelerate matplotlib tqdm
```

---

##  Why This Repository?

Most Stable Diffusion tutorials focus on how to *use* the model (prompt engineering, UI tools). This repository focuses on how the model actually *works*. By stripping away the wrapper classes and exposing the raw tensors, gradients, and diffusion math, this guide is built for developers, students, and AI enthusiasts who want to truly understand Latent Diffusion Models (LDMs).

---
