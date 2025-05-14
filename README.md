
# Left Atrium Segmentation with U-Net

This project implements a deep learning pipeline for segmenting the **left atrium** from cardiac MRI slices using a **U-Net** architecture. It was developed as a personal deep learning project focused on medical image segmentation using PyTorch and PyTorch Lightning.

The repository includes preprocessing, model training, evaluation, and full visualization of the segmentation results.

---

## 🖼️ Results

**Prediction Example:**

![Prediction overlay](images/train_result.png)

**Segmentation Animation:**

https://user-images.githubusercontent.com/luissm01/video_result.mp4  
*(or embed locally if preferred: `images/video_result.mp4`)*

---

## 📁 Project Structure

```
├── data/              # (ignored) original dataset goes here
├── images/            # Results and animations
├── logs/              # TensorBoard logs
├── notebooks/         # Jupyter notebooks
├── scripts/           # Model and dataset code
├── weights/           # Trained model checkpoints
├── .gitignore
└── README.md
```

---

## 📥 Download Dataset

Download the original dataset from the Medical Segmentation Decathlon:

- [Task02_Heart Dataset](https://drive.google.com/file/d/1wEB2I6S6tQBVEPxir8cA5kFB8gTQadYY/view)
- Place the extracted files inside `data/Task02_Heart/`

---

## 🧪 How to Run

```bash
git clone https://github.com/luissm01/left-atrium-segmentation.git
cd left-atrium-segmentation
pip install -r requirements.txt
```

---

## 🏋️ Training

> Training was done from a **Jupyter notebook** instead of a `.py` script due to a known issue:
>
> `num_workers > 0` caused a **"fail to allocate bitmap"** error when visualizing images.
>
> You can reuse the included weights if you prefer not to retrain.

Run the training notebook:

```python
# Inside notebooks/04-Train.ipynb
trainer.fit(model, train_loader, val_loader)
```

---

## 📈 TensorBoard

```bash
tensorboard --logdir=logs
```

Use this to monitor training and view segmentation overlays during validation.

---

## 📊 Evaluation

The evaluation notebook allows you to:

- Load a pretrained model
- Compute Dice score over the validation set
- Visualize predicted masks slice-by-slice
- Animate segmentation across a test volume

---

## 📄 License

This project is released under the MIT License.


---

## 🛠️ Environment Setup (Optional)

If you want to reproduce the exact environment used in this project, you can use one of the following options:

### Option 1: Using pip
```bash
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
conda env create -f environment.yml
conda activate your-env-name  # replace with the name in the .yml file
```

Make sure to have CUDA configured if you intend to use GPU for training or inference.

## 👤 Author

This project was developed by **Luis Sánchez Moreno** as a personal deep learning initiative to explore medical image segmentation using PyTorch and PyTorch Lightning.

The goal was to build a complete and reproducible segmentation pipeline applied to cardiac MRI data, with emphasis on understanding model training, loss functions, evaluation, and visualization techniques.

You can find more of my work at: [github.com/luissm01](https://github.com/luissm01)
