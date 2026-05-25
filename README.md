# 📸 Image Captioning Mini-Project

An end-to-end Deep Learning project that automatically generates smart captions for images using the **Flickr8k dataset**. Built from scratch with PyTorch and deployed with a clean Flask web app!

---

## 🚀 Overview

Developed at **ENSIASD Taroudant**, this project explores the power of **Encoder-Decoder** architectures to bridge the gap between computer vision and natural language processing. 

### Key Highlights:
* **Custom Pipeline:** Built a custom text cleaning, vocabulary indexing, and padded data loading pipeline.
* **Architecture Battles:** Tested and compared three different decoders: LSTM, GRU, and Transformers.
* **Deep Hyperparameter Analysis:** Experimented with embedding sizes, layer depths, dropouts, and optimizers.
* **Web UI:** Created a local interactive web app to upload images and see predictions live.

---

## 🛠️ Architecture & Tech Stack

* **Encoder (Vision):** Pre-trained **ResNet-50** with its final classification layer removed to extract rich visual features.
* **Decoder (Language):** * **RNN-based:** LSTM / GRU options with the image features fed directly into the text embedding sequence.
  * **Attention-based:** A PyTorch Transformer Decoder utilizing causal masking for parallelized text generation.

---

## 📊 Quick Performance Matrix

After testing multiple hyperparameter combinations, here is what we found:

* **GRU vs. LSTM:** **GRU won** with a lower validation loss (~2.79 vs ~2.93) and cleaner confidence scores.
* **Embedding Size:** Bumping it up from **128 to 1024** dramatically increased caption accuracy.
* **Layer Depth:** Multi-layer networks suffered from vanishing gradients; **1 hidden layer** yielded the best results.
* **Optimizer:** **Adam smashed it** thanks to its adaptive learning rate, while standard SGD completely stalled out.
* **The Transformer Bonus:** The Transformer Decoder generated the most fluid and context-aware captions thanks to multi-head self-attention mechanisms.

> 💡 **The Dream Team Setup:** To get the best captions, use **ResNet-50** + **1-Layer GRU (1024 dim)** or **Transformer**, optimized with **Adam ($lr=3\times10^{-4}$)** and a **0.3 Dropout** rate.

---

## 💻 Interactive App

The interface is backed by **Flask**. It lets you interactively test your models on unseen images. Just pick an image, hit generate, and watch the model instantly spit out descriptions like: *"two dogs are playing in the grass"*!
