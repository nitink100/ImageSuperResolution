````markdown
# High-Fidelity Image Super-Resolution using ESRGAN

**Author:** Nitin Kanna (UCF ID: 5566997)

---

## Table of Contents
1.  Project Overview
2.  The Technology: Why ESRGAN?
3.  Real-World Applications
4.  Technology Stack
5.  Setup and Installation
6.  How to Run the Project
7.  Performance and Results
8.  Dataset Information
9.  Acknowledgements

---

## Project Overview

This project explores the domain of Single Image Super-Resolution (SISR), a fundamental task in computer vision aimed at enhancing image quality and recovering finer details from a single low-resolution input. The primary objective is to harness the powerful capabilities of Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN) to generate high-fidelity images.

The workflow is centered around a pre-trained ESRGAN model, which is applied to low-resolution images to upscale them. The quality and efficacy of the generated images are quantitatively measured using the **Structural Similarity Index (SSIM)**, a metric that evaluates perceptual similarity between the output and the original high-resolution image.

---

## The Technology: Why ESRGAN?

ESRGAN stands out from its predecessors by introducing several key architectural improvements that allow it to recover more realistic textures and sharper edges. It achieved first place in the PIRM-SR Challenge, demonstrating its state-of-the-art performance.

Key innovations include:
* **Residual-in-Residual Dense Blocks (RRDB):** Employs a deep network structure with dense blocks to capture more sophisticated and detailed image features.
* **Removal of Batch Normalization (BN) Layers:** Unlike previous models, ESRGAN removes BN layers, which were found to introduce artifacts and limit the model's ability to generalize across different image styles.
* **Relativistic Average GAN (RaGAN):** The discriminator was enhanced to predict the relative "realness" of an image rather than making a simple real/fake judgment, helping the generator create more convincing textures.
* **Enhanced Perceptual Loss:** The model uses a perceptual loss function based on VGG features *before* the activation layers, which has proven crucial for generating sharper edges and improving visual quality.

---

## Real-World Applications

High-quality super-resolution has significant practical importance in various fields, including:
* **Medical Imaging:** Enhancing scans for more accurate diagnosis.
* **Surveillance:** Improving the clarity of security footage for analysis.
* **Satellite Imagery:** Refining satellite photos for better mapping and environmental monitoring.

---

## Technology Stack

This project relies on a set of robust, industry-standard libraries for machine learning and image processing:
* **PyTorch:** A powerful deep learning framework used for implementing and running the ESRGAN model.
* **OpenCV (cv2):** An essential library for real-time computer vision and image manipulation tasks.
* **NumPy:** The fundamental package for numerical computation in Python, used for handling image data as arrays.
* **Matplotlib:** A comprehensive library for creating static and interactive visualizations for displaying images.
* **Scikit-learn:** A versatile machine learning library whose ecosystem provides the tools for calculating the SSIM metric.

---

## Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/Super-Resolution-ESRGAN.git](https://github.com/your-username/Super-Resolution-ESRGAN.git)
cd Super-Resolution-ESRGAN
````

**2. Create a virtual environment (recommended):**
This isolates the project dependencies and avoids conflicts.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install the required dependencies:**
The `requirements.txt` file contains all necessary libraries.

```bash
pip install -r requirements.txt
```

-----

## How to Run the Project

To generate your own high-resolution images, follow these instructions carefully.

**Step 1: Place Your Low-Resolution Images**

  * Navigate to the `ESRGAN/data/` folder inside the main project directory.
  * Place all the low-resolution images you wish to upscale into this folder.

**Step 2: Navigate to the ESRGAN Directory**

  * From the root of the project (`Super-Resolution-ESRGAN/`), change your directory into the `ESRGAN` folder.

<!-- end list -->

```bash
cd ESRGAN
```

**Step 3: Execute the Test Script**

  * Run the main testing script using Python.

<!-- end list -->

```bash
python test.py
```

**Step 4: Find Your Results**

  * The script will process each image from the `data/` folder.
  * The newly generated high-resolution images will be saved in the `ESRGAN/results/` folder.
  * You can also monitor the progress and see results printed to the terminal.

-----

## Performance and Results

The model's performance was quantitatively evaluated using the **Structural Similarity Index (SSIM)**, which measures the perceptual similarity between two images on a scale from -1 to 1 (where 1 is a perfect match).

The model achieved the following impressive scores on the test cases:

  * **Case 1 SSIM Score:** 0.9942
  * **Case 2 SSIM Score:** 0.9177

These high scores confirm that the ESRGAN model produces outputs that are not only high-resolution but also structurally and perceptually very close to the original ground-truth images.

-----

## Dataset Information

The model was tested using the **Urban100 dataset**, a widely recognized benchmark in the field of super-resolution. This dataset is composed of 100 high-resolution images of diverse urban environments, making it an excellent choice for evaluating model performance on complex scenes.

  * **Dataset Link:** You can access and download the dataset from Kaggle: [Urban100 Dataset](https://www.google.com/search?q=https://www.kaggle.com/datasets/harshraone/urban100/)

-----

## Acknowledgements

This project is an implementation and exploration of the work presented in the original ESRGAN paper.

  * **Original Paper:** "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" by Xintao Wang, Ke Yu, et al.

<!-- end list -->

```
```
