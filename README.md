# Predictive Text Input System Using N-gram Based Markov Models
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Tkinter](https://img.shields.io/badge/Tkinter-blue?style=for-the-badge)

## Overview
An AI-powered text prediction engine trained on Project Gutenberg literary works and custom WhatsApp chat exports. Builds conditional probability tables for bigram, trigram, and 4-gram Markov chains with backoff algorithms, wrapped in a real-time tkinter typing dashboard.

## System Architecture
```\n[Relational Database / Core API Architecture]\n```

## Features
- Markov probability tables for bigram, trigram, and 4-gram sequences.
- Backoff smoothing algorithms to handle unseen words.
- Interactive tkinter dashboard for real-time word recommendations.

## Tech Stack
- Python with N-gram conditional probability tables
- Kneser-Ney, Witten-Bell, and Laplace smoothing methods
- Tkinter for real-time dashboard UI

## Getting Started
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
