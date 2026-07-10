# ⌨️ Predictive Text Input System Using N-gram Markov Models
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Tkinter](https://img.shields.io/badge/Tkinter-blue?style=for-the-badge) ![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Project Overview](#🎯-project-overview)
- [What This Project Does](#🚀-what-this-project-does)
- [Key Innovation](#🔬-key-innovation)
- [Performance Highlights](#📊-performance-highlights)
- [Architecture](#🏗️-architecture)
- [Tech Stack](#🧱-tech-stack)
- [Quick Start](#💻-quick-start)

---

## 🎯 Project Overview
An AI-powered text prediction engine trained on Project Gutenberg literary works and custom WhatsApp chat exports. Builds conditional probability tables for bigram, trigram, and 4-gram Markov chains with backoff algorithms, wrapped in a real-time tkinter typing dashboard.

---

## 🚀 What This Project Does
* **The Challenge:** Standard predictive keyboards require massive neural models, which are too heavy to deploy on edge platforms or integrate into simple desktop scripts.
* **Our Solution:** A modular N-gram Markov chain text predictor with mathematical backoff smoothing and an interactive typing interface.

---

## 🔬 Key Innovation
| Feature | Traditional Deep Learning ❌ | N-gram Markov System ✅ | Benefit |
|---------|-----------------------------|-------------------------|---------|
| **Model Size** | Gigabytes of transformer parameters | **Conditional probability tables** | Runs instantly on CPU with low RAM |
| **Smoothing** | Softmax temperatures | **Kneser-Ney / Witten-Bell backoffs** | Handles out-of-vocabulary inputs cleanly |
| **UI** | CLI inputs or heavy web app | **Interactive Tkinter dashboard** | Real-time prediction suggestions |

---

## 📊 Performance Highlights
- ✅ **Multiple smoothing algorithms** (Laplace, Kneser-Ney, Witten-Bell).
- ✅ **Trained on Gutenberg and WhatsApp** corpuses (42,000+ sentences).
- ✅ **Tkinter GUI** showing predictions in under 1ms.

---

## 🏗️ Architecture
```\n[Core Architectural Components & Datastore Framework]\n```

---

## 🧱 Tech Stack
- Python with N-gram conditional probability tables
- Kneser-Ney, Witten-Bell, and Laplace smoothing methods
- Tkinter for real-time dashboard UI

---

## 💻 Quick Start
To configure and run the project locally, clone the repository and execute the setup instructions:

```bash
git clone https://github.com/Raghuram-sekar/Predictive-Text-Input-System.git
cd Predictive-Text-Input-System

# Execute local setup commands:
# MLE Probability Formula:
# P_MLE(w_n | h) = count(h, w_n) / count(h)
# Laplace Smoothing:
# P_Laplace(w_n | h) = (count(h, w_n) + 1) / (count(h) + V)

pip install -r requirements.txt
python simple_live_demo.py
```
