# smart-home-ids
A machine learning intrusion detection system for smart homes, showing how IoT or network data can be processed, trained, and evaluated to detect malicious activity.

# Smart Home Intrusion Detection System (SHIDD)

This project implements a machine learning–based Intrusion Detection System (IDS) for smart home and IoT environments using the  Smart Home Intrusion Detection
Dataset (SHIDD) by Jacob (2023) via Kaggle. The system analyses network traffic patterns to classify activity as either benign or malicious.

## Problem Statement
Smart homes rely on interconnected IoT devices that are vulnerable to cyber-attacks such as DDoS, probing, and unauthorised access. Traditional intrusion detection systems struggle with scalability, adaptability, and real-time detection. This project explores how machine learning can enhance intrusion detection performance in smart home environments.

## Dataset
The project uses the SHIDD dataset, published by Jacob (2023).  
The dataset contains labelled network traffic flows representing benign behaviour and multiple IoT malware scenarios.

Due to size and licensing constraints, the dataset is not included in this repository.  
Download instructions are provided in data/README.md.

## Approach Overview
The IDS follows a standard machine learning pipeline:
1. Data collection and cleaning
2. Data preprocessing and encoding
3. Feature engineering and selection
4. Model training using supervised learning
5. Model evaluation using multiple performance metrics
6. Prediction on unseen network traffic

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train.py

# Run predictions
python src/predict.py data/sample_input.csv

Results Summary

The trained models were evaluated using accuracy, precision, recall, F1-score, and confusion matrices.
Decision Tree and K-Nearest Neighbour classifiers achieved the strongest performance.

Detailed results are available in docs/RESULTS.md.

Repository Structure
 • notebooks/ – Jupyter notebook implementation
 • src/ – Python scripts for preprocessing, training, and prediction
 • docs/ – Methodology and results documentation
 • assets/ – Figures and visual outputs
 • data/ – Dataset instructions (no raw data uploaded)

Intended Audience

This repository is designed for students, researchers, and beginners interested in IoT security, intrusion detection systems, and applied machine learning.
