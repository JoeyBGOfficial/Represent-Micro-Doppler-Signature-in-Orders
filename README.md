# Represent Micro-Doppler Signature in Orders

## I. INTRODUCTION

### Write Sth. Upfront

This work presents a novel, physics-aware micro-Doppler representation method for through-the-wall radar (TWR) human activity recognition (HAR): **Chebyshev-Time Map (ChTM)**.

Unlike traditional methods that rely on raw Doppler-time map (DTM) or range-time map (RTM), which are often high-dimensional and computationally expensive, this work proposes characterizing micro-Doppler signature using approximate orthogonal Chebyshev polynomial orders. By projecting the kinematic envelopes and texture details into a coefficient space, we achieve a robust, interpretable, and compressed representation.

![Overall](https://github.com/user-attachments/assets/e616239a-1e5a-4a47-9f2a-8617165be90e)
Fig. 1. Overall concept of representing micro-Doppler in orders.

This approach is developed for the challenge of distinguishing subtle differences in activities, such as "Armed" vs. "Unarmed" walking and different indoor human identities under limited data conditions. **This work is not as that much effective as we expected. But we wish this could provide a great inspiration to our collegues**!

### Basic Information:

**My Email:** JoeyBG@126.com;

**Abstract:** Non-line-of-sight sensing of human activities is enabled by TWR, but the distinctiveness of micro-Doppler signatures between similar activities is minimal. Furthermore, the large scale of input images required for time-frequency spectrograms creates challenges for efficiency. To address this, the **Chebyshev-time map** is proposed. This method characterizes micro-Doppler signatures using polynomial orders. We first extract kinematic envelopes of the torso and limbs, then map the spectrum slices into a robust Chebyshev-time coefficient space. This preserves multi-order morphological details while compressing data. Numerical simulations and experiments demonstrate its capability to characterize armed and unarmed activities while achieving a balance between accuracy and input data dimensions.

**Corresponding Papers:**

[1] W. Gao, “Represent Micro-Doppler Signature in Orders,” arXiv preprint, Feb. 2026. Link: 

## II. CHTM GENERATION & VISUALIZATION

### A. Theory in Simple

The core innovation is transforming the DTM into the order-time domain. The process involves:
1.  **Envelope Extraction:** Adaptive thresholding to identify the macro-Doppler (Torso) and micro-Doppler (Limb) boundaries.
2.  **Orthogonal Projection:** Slicing the DTM at each time step and projecting the spectral vector onto Chebyshev polynomials of the first kind.
3.  **Coefficient Mapping:** The resulting coefficients represent physical features, where $c_0$ is energy, $c_1$ is centroid offset indicating acceleration, $c_2$ is spectral width, and high orders capture micro-motion texture.

![ChTM_Generation](https://github.com/user-attachments/assets/02ff9636-4ee4-4e5c-9675-0266a7389076)
Fig. 2. Schematic diagram of the DTM envelope extraction and ChTM generation method.

### B. Main Files Explanation

**1. DTM_to_ChTM.m**

This is the core function of the project. It accepts a raw DTM matrix and outputs two feature maps:
* **ChTM_Macro:** Features derived from the torso envelope, capturing bulk motion and RCS scintillation.
* **ChTM_Micro:** Features derived from the global limb envelope, capturing the detailed micro-Doppler texture of arms and legs.

**2. Synthetic_Demo.m**

A comprehensive demo script. It generates synthetic DTM data, adds noise, and runs the "DTM_to_ChTM" extraction.

### C. Supporting Codes Explanation

None.

## III. SUPERVISED CONTRASTIVE LEARNING (Not Included in Our Papaer)

### A. Theory in Simple

To fully utilize the ChTM features, we offer a possibly effective network structure: **Supervised Contrastive Learning (SupCon)** framework. The model pulls representations of the same class together in the latent space while pushing apart different classes, regardless of the specific view. Then, a linear classifier is trained on top of the frozen or fine-tuned backbone using standard Cross-Entropy loss.

The backbone is a lightweight convolutional neural network integrated with coordinate attention modules to capture the long-range dependencies in the ChTM's time and order dimensions.

### B. Main Files Explanation

**1. ChTM_SupCon.py**

This is the main training and validation script of the proposed SupCon network model. The code includes the following properties:
* **Dual-View Data Loading:** Custom "RadarDataset" that generates two augmented views of every sample for contrastive learning.
* **SupConLoss:** Implementation of the Supervised Contrastive Loss function.
* **Coordinate Attention Backbone:** A specialized CNN designed for radar feature maps.
* **Two-Stage Training:** Automatically handles the transition from SupCon pretraining to Classifier finetuning.

### C. Supporting Codes Explanation

None.

## IV. SOME THINGS TO NOTE

**(1) Environment Dependencies:** MATLAB R2024b+ is recommended for the signal processing scripts. The deep learning script is built on Paddlepaddle 3.3.0 in Python 3.10. Please ensure you have the correct version installed.

**(2) Data Format:** The "ChTM_SupCon.py" script expects the dataset root directory to be organized by class folders. Ensure your generated ChTM images are saved in this structure.

**(3) Parameter Tuning:** In "DTM_to_ChTM.m" function, the "torso_th" and "mD_th" thresholds are critical. If your DTMs have different noise floors than the synthetic demo, you may need to adjust these values to ensure accurate envelope extraction.

**(4) Rights:** ⭐ This repository open-sources the core algorithms. The specific simulated and measured datasets mentioned in the paper are not included. This code is for academic learning and research purposes only. Commercial use or redistribution without consent is prohibited. ⭐
