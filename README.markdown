# Nucleus Patch Classification: Feature Extraction and SVM Classification

## Overview
This project implements a pipeline for classifying nucleus patches (benign vs. malignant) from microscopy images using handcrafted and deep features. The process involves:
1. Extracting handcrafted features (texture, shape, color) from nucleus patches.
2. Extracting deep features using VGG19 and reducing their dimensionality with PCA.
3. Fusing handcrafted and deep features.
4. Training an SVM classifier with k-fold cross-validation and hyperparameter tuning.
5. Evaluating performance and inspecting true/predicted labels for specific patches.

The dataset consists of RGB nucleus patch images split into training (2023 patches), validation (506 patches), and test (735 patches) sets.

## Pipeline Steps

### 1. Handcrafted Feature Extraction
- **Purpose**: Extract texture, shape, and color features from each nucleus patch.
- **Method**:
  - **Texture Features**: Haralick features from Gray-Level Co-occurrence Matrix (GLCM) with 4 angles (0, π/4, π/2, 3π/4), yielding 16 features (contrast, correlation, energy, homogeneity × 4 angles).
  - **Shape Features**: Area, perimeter, and eccentricity from the largest region in a binarized patch, yielding 3 features.
  - **Color Features**: Histograms of RGB channels (first 50 bins per channel), yielding 150 features.
  - **Total Features per Patch**: \(16 + 3 + 150 = 169\).
- **Code**: Uses `skimage` for GLCM (`graycomatrix`, `graycoprops`), region properties (`regionprops`, `label`), and histogram computation.
- **Output**:
  - Training set: `(2023, 169)` (2023 patches, 169 features each).
  - Validation set: `(506, 169)`.
  - Test set: `(735, 169)`.

### 2. Deep Feature Extraction with VGG19
- **Purpose**: Extract deep features from nucleus patches using a pre-trained VGG19 model.
- **Method**:
  - Load VGG19 (without top layers) with weights from a local file, input shape `(224, 224, 3)`.
  - Extract features from the `block5_pool` layer (shape `(7, 7, 512)`), flattened to 25,088 features per patch.
  - Preprocess images (resize to 224x224, normalize with `preprocess_input`).
- **Code**: Uses `tensorflow.keras.applications.VGG19` and `skimage.transform.resize`.
- **Output**:
  - Training set: `(2023, 25088)`.
  - Validation set: `(506, 25088)`.
  - Test set: `(735, 25088)`.

### 3. Dimensionality Reduction with PCA
- **Purpose**: Reduce the 25,088 deep features to 512 to manage computational complexity.
- **Method**:
  - Apply PCA (`sklearn.decomposition.PCA`) with `n_components=512`.
  - Fit PCA on training features and transform training, validation, and test features.
- **Code**: Uses `sklearn.decomposition.PCA`.
- **Output**:
  - Training set: `(2023, 512)`.
  - Validation set: `(506, 512)`.
  - Test set: `(735, 512)`.

### 4. Feature Fusion
- **Purpose**: Combine handcrafted and PCA-reduced deep features for a comprehensive representation.
- **Method**:
  - Concatenate handcrafted (169 features) and PCA-reduced deep features (512 features) along the feature axis.
- **Code**: Uses `np.concatenate`.
- **Output**:
  - Training set: `(2023, 681)` (169 + 512 features).
  - Validation set: `(506, 681)`.
  - Test set: `(735, 681)`.

### 5. SVM Classification with K-Fold Cross-Validation
- **Purpose**: Train a Support Vector Machine (SVM) classifier to predict benign (0) or malignant (1) labels for nucleus patches.
- **Method**:
  - Standardize fused features using `StandardScaler`.
  - Convert one-hot encoded labels to single labels (`np.argmax`).
  - Use `GridSearchCV` with 5-fold cross-validation to tune SVM hyperparameters:
    - `C`: [0.1, 1.0, 10.0] (regularization).
    - `gamma`: ['scale', 'auto', 0.001, 0.01] (RBF kernel scale).
    - `class_weight`: [None, 'balanced'] (handle imbalance).
  - Total candidates: \(3 \times 4 \times 2 = 24\), total fits: \(24 \times 5 = 120\).
  - Best parameters: `C=1.0`, `gamma=0.001`, `class_weight='balanced'`.
- **Code**: Uses `sklearn.svm.SVC`, `sklearn.model_selection.GridSearchCV`, `sklearn.preprocessing.StandardScaler`.
- **Evaluation**:
  - Compute accuracy, precision, recall, and F1-score (weighted average) on validation and test sets.
  - Generate a classification report for the test set with per-class metrics (benign, malignant).

### 6. Inspecting True and Predicted Labels
- **Purpose**: Check the true and predicted labels for a specific nucleus patch.
- **Method**:
  - Select a patch index (e.g., 0 to 734 for test set).
  - Retrieve true label (`y_test_labels[patch_index]`) and predicted label (`y_test_pred[patch_index]`).
  - Optionally visualize the patch image from `X_test`.
- **Code**: Uses `numpy` for label access and `matplotlib.pyplot` for visualization.

### 7. Inspecting Handcrafted Features
- **Purpose**: Display the 169 handcrafted features extracted per nucleus patch.
- **Method**:
  - Print breakdown: 16 texture (Haralick), 3 shape, 150 color features.
  - Confirm total features: \(16 + 3 + 150 = 169\).
- **Output**:
  ```
  Feature Breakdown Per Nucleus Patch:
  - Texture Features (Haralick, GLCM): 16 features (4 features × 4 angles: contrast, correlation, energy, homogeneity)
  - Shape Features: 3 features (area, perimeter, eccentricity)
  - Color Features (RGB Histogram): 150 features (50 bins per channel × 3 channels)
  - Total Features Per Nucleus Patch: 16 + 3 + 150 = 169

  Handcrafted Feature Dimensions:
  Training set (2023 patches): (2023, 169)
  Validation set (506 patches): (506, 169)
  Test set (735 patches): (735, 169)
  ```

## Requirements
- Python libraries: `numpy`, `scikit-learn`, `scikit-image`, `tensorflow`, `matplotlib`.
- Dataset: RGB nucleus patch images (`x_train`, `x_val`, `X_test`) and one-hot encoded labels (`y_train`, `y_val`, `Y_test`).
- VGG19 weights: Local file (`vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5`).

## Notes
- **Per Nucleus Patch**: All features (169 handcrafted, 512 PCA-reduced deep, 681 fused) and classifications are per nucleus patch, as each image in `x_train`, `x_val`, and `X_test` is a pre-cropped patch centered on a single nucleus.
- **Class Imbalance**: The SVM uses `class_weight='balanced'`, suggesting an imbalance (likely more benign than malignant patches).
- **Dataset Size**: The current dataset has 2023 training, 506 validation, and 735 test patches, differing from an earlier dataset (2462, 616, 885), indicating a possible subset or different preprocessing.

## Future Improvements
- **Hyperparameter Tuning**: Expand `param_grid` for finer SVM tuning or use `RandomizedSearchCV` for efficiency.
- **Feature Selection**: Apply techniques like Recursive Feature Elimination (RFE) to reduce the 681 fused features.
- **Alternative Models**: Test other classifiers (e.g., Random Forest, XGBoost) on fused features.
- **Misclassification Analysis**: Inspect misclassified patches to identify patterns (e.g., ambiguous nuclei).

## Contact
For questions or contributions, please open an issue or pull request in this repository.