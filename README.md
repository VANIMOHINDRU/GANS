## Project Overview

This project learns the probability density function (PDF) of a transformed NO₂ concentration variable using a Generative Adversarial Network (GAN).

The model:
-Does not assume any parametric distribution
-Learns the distribution purely from data samples
-Uses generator samples to approximate the PDF via KDE

## Dataset

Source: India Air Quality Dataset (Kaggle)

Feature Used: NO₂ concentration

Missing values removed before processing

# Step 1: Variable Transformation

Each NO₂ value 𝑥 is transformed using:


Where:



For Roll Number 102303064:

a_r = 1.5
b_r = 1.5

# Step 2: GAN Architecture

The GAN learns the unknown distribution of the transformed variable 𝑧.

Generator
Input: 5D Gaussian noise

Layers:
Linear(5 → 32) + ReLU
Linear(32 → 32) + ReLU
Linear(32 → 1)

Discriminator
Input: 1D sample
Layers:
Linear(1 → 32) + ReLU
Linear(32 → 32) + ReLU
Linear(32 → 1) + Sigmoid

Training Details
Loss: Binary Cross Entropy
Optimizer: Adam
Learning Rate: 0.0002
Epochs: 2000
Batch Size: 128
Data normalized for stability

The discriminator distinguishes:
Real samples: 
𝑧
Fake samples: 
𝐺(𝜖), where 𝜖∼𝑁(0,1)

# Step 3: PDF Estimation

After training:
10,000 samples generated from the trained generator
Kernel Density Estimation (KDE) applied
Estimated PDF plotted and compared with real distribution

# Results
![GAN PDF](pdf.png)

✔ Mode Coverage
Dominant mode captured successfully
No visible mode collapse

✔ Training Stability
Balanced generator–discriminator losses
Stable adversarial training

✔ Distribution Quality
Right-skew structure learned
Good alignment with real distribution in dense regions
Minor deviation in extreme tails


🚀 How to Run
pip install torch pandas numpy matplotlib scikit-learn
python gan_training.py

📝 Key Takeaway

This project demonstrates that GANs can learn complex probability distributions purely from samples, without assuming Gaussian or any analytical form.