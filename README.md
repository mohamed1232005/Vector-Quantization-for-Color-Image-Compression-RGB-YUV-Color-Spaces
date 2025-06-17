# Vector Quantization for Color Image Compression (RGB & YUV) Color-Spaces

This project implements a complete **Vector Quantization (VQ)** compression system for color images using both **RGB** and **YUV** color spaces. It was developed as the **Final Project** for the **Information Theory course 

The system allows switching between color spaces using a configurable boolean flag:

```java
boolean USE_YUV = true; // Enables YUV compression mode
// Set to false for RGB mode
```

---

## 📑 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Compression Methodology](#compression-methodology)
- [Color Space Modes](#color-space-modes)
- [Switching Between RGB and YUV](#switching-between-rgb-and-yuv)
- [Results and Evaluation](#results-and-evaluation)
- [Execution Instructions](#execution-instructions)
- [Technologies Used](#technologies-used)
- [Mathematical Formulas](#mathematical-formulas)
- [Conclusion](#conclusion)
- [References](#references)
- [Author](#author)
- [License](#license)

---

## 🧠 Overview

**Vector Quantization (VQ)** is a **lossy image compression technique** where the image is partitioned into blocks (e.g., 2×2 pixels), and each block is approximated using a codeword from a **codebook** trained via clustering algorithms such as the **Linde-Buzo-Gray (LBG)** algorithm.

This implementation:
- Compresses and decompresses color images.
- Trains a codebook of **256 codewords** for each channel (R, G, B or Y, U, V).
- Supports **YUV chroma subsampling (4:2:0)** for better compression performance.
- Outputs both compression ratio and **Mean Squared Error (MSE)**.

---

## 🖼️ Dataset

| Category | Training Images | Testing Images |
|----------|------------------|----------------|
| Nature   | 10               | 5              |
| Faces    | 10               | 5              |
| Animals  | 10               | 5              |
| **Total**| 30               | 15             |

- Training images are used to generate codebooks.
- Testing images are compressed and reconstructed using these codebooks.

---

## 🧪 Compression Methodology

### 🔷 Codebook Training (LBG Algorithm)

1. Convert each image into **non-overlapping 2×2 blocks**.
2. Flatten each block into a vector (length = 4).
3. Use **K-means/LBG algorithm** to cluster the vectors.
4. Select **K = 256** clusters (codewords).
5. Store the final centroids as the codebook for that channel.

Separate codebooks are created for each component:
- **RGB Mode**: `codebook_R`, `codebook_G`, `codebook_B`
- **YUV Mode**: `codebook_Y`, `codebook_U`, `codebook_V`

### 🔷 Compression

- Replace each block with the **index of its nearest codeword** in the corresponding codebook.
- In **YUV Mode**, `U` and `V` components are **subsampled by 2x** (half width and height).

### 🔷 Decompression

- Retrieve each block by looking up the corresponding codeword.
- Reconstruct the image by reversing subsampling (if needed).
- In **YUV Mode**, the image is converted back to RGB for display.

---

## 🎨 Color Space Modes

### ✅ RGB Mode

- Works directly on the **R**, **G**, and **B** channels.
- Each channel is processed at **full resolution**.
- Simpler and faster but with **higher MSE** and **lower compression ratio**.

### ✅ YUV Mode (Bonus)

- Convert RGB to **YUV** using standard transformation:
  - Y = 0.299R + 0.587G + 0.114B
  - U = -0.14713R - 0.28886G + 0.436B
  - V = 0.615R - 0.51499G - 0.10001B
- Apply **4:2:0 subsampling**:
  - `U` and `V` are reduced to **half width and height**
- Compress and decompress using VQ
- Upsample `U` and `V`, convert back to RGB

---

## 🔁 Switching Between RGB and YUV

To change compression mode, set the following flag in your Java code:

```java
boolean USE_YUV = true; // YUV mode with chroma subsampling
// false enables RGB mode
```

The logic inside the main class will automatically:
- Convert to/from YUV
- Subsample and upsample U/V channels
- Assign correct codebooks
- Control output and logging

---

## 📊 Results and Evaluation

### 🔬 Key Metrics

| Metric               | Description                                      |
|----------------------|--------------------------------------------------|
| **MSE**              | Measures distortion after reconstruction         |
| **Compression Ratio**| Ratio of original size to compressed size        |
| **Runtime**          | Time required for compression and reconstruction|

### 📈 Comparative Results

| Mode   | Avg MSE | Compression Ratio | Visual Quality | Processing Speed |
|--------|---------|-------------------|----------------|------------------|
| RGB    | Higher  | Lower              | Good           | Faster           |
| YUV    | Lower   | Higher             | Very Good      | Slightly Slower  |

- **YUV mode** provides superior results due to subsampling.
- **RGB mode** is faster and simpler, but less efficient.

---

## ⚙️ Execution Instructions

### 🧷 Setup

1. Ensure you have **Java JDK 8 or later** installed.
2. Place all training and testing images in respective folders:
   ```
   /train/
   /test/
   ```

### ▶️ Running the Program

```bash
javac *.java
java VQMain
```

### 🧪 Output

- **Compressed image files** and **reconstructed images** saved in `/output/`
- Console output includes:
  - MSE for each test image
  - Compression ratios
  - Timing information

---

## 💻 Technologies Used

| Component        | Role                                 |
|------------------|--------------------------------------|
| Java SE 8+       | Core programming and image handling  |
| LBG Algorithm    | Codebook generation (unsupervised)   |
| BufferedImage    | Image I/O, RGB pixel manipulation    |
| Custom Matrix Ops| Subsampling, Upsampling, Color space |
| Math Utilities   | Distance metrics, vector ops         |

---

## 📐 Mathematical Formulas

### 🎯 Mean Squared Error (MSE)

$$
MSE = \\frac{1}{N} \\sum_{i=1}^{N} (P_i - Q_i)^2
$$

Where:
- \( P_i \) = Original pixel value
- \( Q_i \) = Reconstructed pixel value
- \( N \) = Total number of pixels

---

### 🎯 Compression Ratio (CR)

$$
CR = \\frac{\\text{Original Size (bits)}}{\\text{Compressed Size (bits)}}
$$

> Codebook size is **excluded** from compression ratio calculation.

---

## 🏁 Conclusion

- The **Vector Quantization** system efficiently compresses RGB and YUV images.
- **YUV mode with chroma subsampling (4:2:0)** is more efficient:
  - Lower MSE
  - Higher compression ratio
  - Excellent visual quality
- The toggle-based design allows flexible switching between RGB and YUV.

---

## 📚 References

- Gonzalez & Woods, *Digital Image Processing*
- Sayood, *Introduction to Data Compression*
- [LBG Algorithm](https://en.wikipedia.org/wiki/Linde%E2%80%93Buzo%E2%80%93Gray_algorithm)
- [YUV Subsampling Explained](https://stackoverflow.com/questions/36949149/what-is-chroma-subsampling-420-422-444)

---
