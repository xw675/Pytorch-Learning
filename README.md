# Pytorch Learning

 Simple Convolutional Neural Network (CNN) for CIFAR-10 image classification.

* **Dataset**: CIFAR-10 (10 classes of 32×32 RGB images)
* **Model**: Custom CNN with convolutional blocks, batch normalization, and dropout

---

## 🚀 Model Architecture

The network consists of:

1. **Block 1**

   * Conv2d(3→32, kernel=3, padding=1)
   * BatchNorm2d(32)
   * ReLU
   * MaxPool2d(2)

2. **Block 2**

   * Conv2d(32→32, kernel=3, padding=1)
   * BatchNorm2d(32)
   * ReLU
   * MaxPool2d(2)

3. **Global Downsampling**

   * Conv2d(32→64, kernel=3, padding=1)
   * BatchNorm2d(64)
   * ReLU
   * AdaptiveAvgPool2d(output\_size=(2,2))

4. **Deep Feature Block**

   * Conv2d(64→128, kernel=3, padding=1)
   * Conv2d(128→256, kernel=3, padding=1)
   * BatchNorm2d(256)
   * ReLU

5. **Classification Head**

   * Flatten
   * Dropout(p=0.2)
   * Linear(256×2×2 → 256)
   * ReLU
   * Dropout(p=0.3)
   * Linear(256 → 64)
   * Linear(64 → 10)
   * Output logits for 10 CIFAR-10 classes

---

## 📈 Results

**Accuracy: 86.41%**
