# Credit Card Fraud Detection using Autoencoders

This project utilizes Autoencoders, a type of neural network, to detect fraudulent transactions in credit card datasets. The Autoencoder model is trained to reconstruct data patterns for normal transactions, and discrepancies in reconstruction errors are used to identify fraud. The data used for this project is sourced from Kaggle's credit card fraud detection dataset, which includes anonymized transaction data.

---

## 1. Libraries and Modules Import

This section imports the essential libraries required for data preprocessing, model development, and evaluation:
- **`pandas`, `numpy`**: Data manipulation and numerical computations, including array handling and scaling.
- **`matplotlib`, `seaborn`**: For data visualization, helping to analyze and interpret trends, class distributions, and model outputs.
- **`sklearn`**: Provides tools for data preprocessing, scaling, and evaluating the model's performance.
- **`torch`, `torch.nn`**: For building and training the Autoencoder model using PyTorch.
- **`tqdm`**: For displaying progress bars during long computations.

---

## 2. Data Loading

The dataset is loaded from Google Drive and processed:
- **Input:** The dataset contains multiple features, including time and transaction amount, along with the target class (fraud or normal).
- **Output:** A DataFrame with columns such as `Time`, `V1-V28` (anonymized features), `Amount`, and `Class` (0 for normal, 1 for fraud).

---

## 3. Exploratory Data Analysis (EDA)

- **Class Distribution:** Explored the imbalance between normal and fraudulent transactions.
- **Visualization:** Used a pie chart to visualize the distribution of normal vs. fraudulent transactions.
- **t-SNE:** Applied t-SNE to reduce dimensionality and visualize the dataset in 2D, highlighting the separation between normal and fraudulent transactions.

---

## 4. Data Sampling

- **Normal Transactions:** 1000 random samples are taken from normal transactions.
- **Fraud Transactions:** All fraud transactions are included in the dataset.
- **Shuffling:** The data is shuffled to ensure a randomized and unbiased split for training and testing.

---

## 5. t-SNE Visualization (Before vs After Autoencoding)

- **Initial t-SNE Plot (Before Autoencoding):** The dataset is visualized in 2D before training the Autoencoder. This shows the distribution of normal and fraudulent transactions.
  
- **Post-Autoencoding t-SNE Plot (After Autoencoding):** After training the Autoencoder, t-SNE is applied again to visualize how the encoding process affects the separation of normal and fraudulent transactions. Ideally, the Autoencoder helps in improving the clustering of these two classes.

---

## 6. Data Preprocessing

- **Feature Scaling:** The `Amount` and `Time` columns are standardized using `StandardScaler` to normalize the data.
- **Train-Test Split:** The dataset is split into a training set (for normal transactions) and a test set (which includes both normal and fraudulent transactions).
- **Feature Selection:** The `Class` column is removed from the feature set, as it is the target variable.

---

## 7. Hyperparameters Definition

Defined key hyperparameters for training:
- **Learning Rate:** 0.01
- **Epochs:** 150
- **Batch Size:** 32
- **Threshold for Fraud Detection:** 0.75 (used to classify transactions as fraud or normal based on reconstruction error)

---

## 8. Autoencoder Model Development

Built an Autoencoder model to detect anomalies in the transaction data:
- **Encoder Architecture:** A series of layers that reduce the dimensionality of the input data.
- **Decoder Architecture:** A mirror of the encoder, reconstructing the original data from the encoded features.
- **Activation Functions:** Tanh activations are used in the hidden layers.
- **Optimizer:** Adam optimizer is used for efficient training with a learning rate of 0.01.

---

## 9. Model Training

- **Training Loop:** The model is trained using the reconstruction loss (Mean Squared Error) between the input and the reconstructed data.
- **Backpropagation:** The gradients are calculated and the model's weights are updated during training using backpropagation.
- **Validation:** Loss is monitored on the test set after each epoch, and results are stored for later evaluation.

---

## 10. Evaluation

### 10.1. Precision-Recall Curves

- **Plotting:** The precision-recall curves are plotted to visualize the trade-off between precision and recall for different threshold values.
- **Purpose:** Helps determine the optimal threshold for classifying transactions as fraudulent.

### 10.2. Reconstruction Error Plot

- **Thresholding:** A reconstruction error threshold is applied to detect anomalies (fraud).
- **Visualization:** A scatter plot is created to show the reconstruction error for both normal and fraudulent transactions, with a threshold line indicating the cutoff for fraud detection.

### 10.3. Classification Report

- **Metrics:** Precision, recall, F1-score, and accuracy are calculated for evaluating the model's performance in detecting fraudulent transactions.
- **Purpose:** Provides a comprehensive overview of the model's classification performance.
