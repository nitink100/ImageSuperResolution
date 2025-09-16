# Image Super-Resolution using ESRGAN 🖼️

A PyTorch project that enhances low-resolution images into high-fidelity versions using a state-of-the-art ESRGAN model.

---

## ✨ Key Features

* **State-of-the-Art Model:** Utilizes a pre-trained ESRGAN, renowned for generating realistic textures and sharp edges.
* **High-Fidelity Upscaling:** Transforms low-resolution inputs into high-quality, detailed images, recovering fine details.
* **Performance-Driven:** Quantitatively measures output quality using the **Structural Similarity Index (SSIM)** to ensure perceptual accuracy.
* **Impressive Results:** Achieved an SSIM score of up to **0.9942** on the Urban100 test dataset, demonstrating near-perfect structural similarity.

---

## 🛠️ Tech Stack

* **Deep Learning:** PyTorch
* **Image Processing:** OpenCV
* **Numerical Computation:** NumPy
* **Data Visualization:** Matplotlib
* **Performance Metrics:** Scikit-learn

---

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

Make sure you have the following installed on your system:
* Python (3.8 or newer recommended)
* Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nitink100/Super-Resolution-ESRGAN.git
    cd ImageSuperResolution
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 How to Run the Application

1.  **Add your images:**
    Place the low-resolution images you wish to upscale into the `ESRGAN/data/` directory.

2.  **Navigate to the script directory:**
    From the project root, move into the ESRGAN folder.
    ```bash
    cd ESRGAN
    ```

3.  **Run the enhancement script:**
    ```bash
    python test.py
    ```

The upscaled images will be saved in the `ESRGAN/results/` folder.

---

## 🙏 Acknowledgements

This project is an implementation of the work from the original paper: *"ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"* by Xintao Wang, et al.
