# Retinal Disease Classification from Fundus Images
Multi-label classification of 7 ocular diseases from fundus photographs using EfficientNet-B3, trained on the ODIR-5K dataset.

---

## Results

| Class | Condition | AUC |
|-------|-----------|-----|
| N | Normal | 0.777 |
| D | Diabetic Retinopathy | 0.798 |
| G | Glaucoma | 0.885 |
| C | Cataract | 0.931 |
| A | Age-related Macular Degeneration | 0.833 |
| H | Hypertension | 0.828 |
| M | Myopia | 0.985 |
| — | **Macro AUC** | **0.8623** |

---

## Problem Statement

Ocular diseases are a leading cause of preventable blindness worldwide. Early detection through automated screening of fundus photographs can significantly improve patient outcomes, especially in regions with limited access to ophthalmologists.

This project builds a multi-label classifier that takes a single fundus image and predicts the presence of 7 retinal conditions simultaneously — reflecting the real-world clinical scenario where a patient can have multiple co-occurring conditions.

---

## Dataset

**ODIR-5K** (Ocular Disease Intelligent Recognition) — a structured ophthalmic database of 5,000 patients collected by Shanggong Medical Technology from hospitals across China.

- 3,500 training patients, each with left and right eye fundus images
- 8 diagnostic labels: Normal, Diabetes, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other
- Labels provided per patient, not per eye

**After cleaning:**
- Dropped the `Other` class — noisy catch-all with no clinical specificity
- Removed 551 patients whose only label was `Other` (would become unlabeled after dropping)
- Identified and removed 14 corrupted/near-black images (kept the healthy eye for those patients)
- Final dataset: **2,949 patients, 5,884 images, 7 classes**

---

## Approach

### Preprocessing
Raw fundus images have two main issues — inconsistent contrast across hospitals and large black borders surrounding the circular retinal region. Both are addressed before training:

**Fundus Circle Crop:** Detects the bright circular fundus region via thresholding and contour detection, then crops to its bounding box. Removes ~30-40% of wasted black pixels, forcing the model to focus entirely on retinal tissue.

**CLAHE (Contrast Limited Adaptive Histogram Equalization):** Applied to the green channel only, which has the highest contrast for retinal structures. Divides the image into small tiles and equalizes each independently, enhancing subtle features like hemorrhages, microaneurysms, and optic disc boundaries without over-amplifying noise. All images are preprocessed and saved to disk before training to avoid per-epoch overhead.

### Data Pipeline
Each patient contributes two independent samples (left eye, right eye), effectively doubling the dataset to ~5,884 images. The train/val split is performed at the **patient level** (80/20) to prevent data leakage — both eyes of the same patient always land in the same split.

### Model
EfficientNet-B3 pretrained on ImageNet, with the classification head replaced by:
```
Dropout(p=0.5) → Linear(1536 → 7)
```
Sigmoid is not applied in the model — it is handled internally by the loss function for numerical stability. At inference, sigmoid is applied to convert logits to probabilities.

### Loss Function
**Focal Loss** (α=1, γ=2) — chosen over standard Binary Cross Entropy to address class imbalance. Focal Loss down-weights easy examples (majority classes like Normal and Diabetes) and focuses training on hard, underrepresented examples (Hypertension: 103 samples, AMD: 164 samples). This is the same loss used in the RetinaNet object detection paper.

### Training
- **Optimizer:** Adam (lr=5e-5, weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau — halves LR when val loss stagnates for 3 epochs
- **Early stopping:** Stops training when macro AUC hasn't improved by >0.001 for 5 consecutive epochs
- **Augmentation:** Random horizontal/vertical flip, rotation (±15°), mild color jitter (brightness=0.2, contrast=0.2). Color jitter kept mild to preserve clinically relevant features like hemorrhage color

Training stopped at epoch 12 via early stopping. Best checkpoint saved at epoch 7.

### Explainability — Grad-CAM
Gradient-weighted Class Activation Mapping (Grad-CAM) hooks into the last convolutional block of EfficientNet-B3 to generate heatmaps showing which regions of the retina most influenced the prediction. This produces clinician-interpretable visualizations — for example, highlighting the optic disc region for Glaucoma predictions.

---

## Key Design Decisions

**Why patient-level split?** Each patient has two eyes with the same label. An image-level split risks placing `1209_left.jpg` in train and `1209_right.jpg` in val — the model would have effectively seen the patient during training, inflating validation metrics.

**Why treat each eye independently (not concatenate)?** Simplest approach that scales well. The model sees each eye as an independent sample. A natural v2 improvement is dual-eye inference — run both eyes separately and average predictions — which typically improves AUC by 1-2%.

**Why Focal Loss over weighted BCE?** Weighted BCE requires manually tuning per-class weights. Focal Loss adapts dynamically during training based on prediction confidence, making it more robust to the multi-label imbalance scenario here.

**Why dropout=0.5?** Initial experiments with dropout=0.3 showed clear overfitting by epoch 6 (train loss 0.036, val loss 0.085). Increasing to 0.5 combined with weight_decay=1e-4 brought train and val loss within 0.01 of each other through epoch 7.

**Why CLAHE on green channel only?** The green channel captures the highest contrast between retinal vessels, lesions, and background tissue. Applying CLAHE to all three channels can introduce color artifacts that destroy clinically relevant color information (e.g. redness of hemorrhages).

---

## Repository Structure

```
odir5k-retinal-classifier/
├── odir5k_kaggle.ipynb   # Full training pipeline
├── app.py                # Streamlit inference app with Grad-CAM
├── requirements.txt      # Dependencies
└── README.md
```

---

## How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Training:**
Open `odir5k_kaggle.ipynb` in Kaggle with the ODIR-5K dataset added. The notebook runs end to end — preprocessing, training, evaluation, and Grad-CAM visualization.

**Inference:**
```bash
streamlit run app.py
```
Place `best_model.pth` in the same directory before running.

---

## Future Improvements

- **Dual-eye inference:** Run left and right eye independently, average predictions at patient level — expected +1-2% macro AUC
- **Native resolution:** EfficientNet-B3's optimal input is 300×300. Training at 224×224 was a compute constraint; upgrading expected to improve subtle feature detection
- **Medically pretrained backbone:** Replace ImageNet pretraining with a backbone pretrained on fundus data — several are publicly available and typically outperform ImageNet initialization on retinal tasks
- **Test-time augmentation (TTA):** Average predictions across multiple augmented versions of the same image at inference
- **Ensembling:** Combine EfficientNet-B3 with a Vision Transformer (ViT) — CNNs and transformers capture complementary features

---

## References

- [ODIR-5K Dataset — Kaggle](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [Focal Loss for Dense Object Detection (RetinaNet)](https://arxiv.org/abs/1708.02002)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
