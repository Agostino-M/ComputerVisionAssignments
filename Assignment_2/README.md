# Face Classification - 2° Assignment  
> **Disclaimer:** Only the text content from the original PDF has been automatically extracted with PyMuPDF. Images have been omitted. This Markdown file is for demonstration purposes only and is **not intended to replace the original PDF**.

## Face Description
- Faces are one of many forms of biometrics used to identify individuals and to verify their identity.  
- Feature extraction is a very important step in face verification.  
- Different feature extraction techniques like Principal Component Analysis (PCA), Fisher Linear Discriminant analysis (FLD) and Local Binary Patterns are generally used.  

---

## Face Detection
- Algorithm for Feature Extraction  

---

## Local Binary Patterns
- The local binary pattern (LBP) is a popular texture descriptor based on appearance features.  
- It describes local structures of an image and is invariant to changes in illumination.  
- LBP was first introduced in 1994 and has been used in a wide range of applications, especially face detection and recognition.  

### LBP Characteristics
- **Advantages**:
  - Robust to illumination variations (captures texture info under different lighting conditions).  
  - Computationally efficient.  
  - Invariant to image rotation and scale (in uniform modality).  
  - Highly discriminative for texture analysis.  

- **Limitations**:
  - Sensitive to noise.  
  - Captures only local texture information in the immediate pixel neighborhood.  
  - Cannot capture rotational information.  
  - Works on grayscale images only, no color features.  

*Reference: T. Ojala, M. Pietikäinen, and D. Harwood (1994), “Performance evaluation of texture measures with classification based on Kullback discrimination of distributions”, Proceedings of the 12th IAPR International Conference on Pattern Recognition (ICPR 1994).*

---

## Local Binary Patterns – Default Modality
- LBP re-encodes the local neighborhood by comparing each pixel with its neighbors.  
- Each surrounding pixel is assigned a binary value (greater/less than center pixel).  
- These binary values are concatenated into a binary number and transformed to decimal.  
- A histogram of values is used to describe the texture distribution.  
- The process is applied pixel by pixel, similar to spatial filtering.  

---

## Local Binary Patterns – Uniform Modality
- Traditional LBP is not rotation invariant and is grid-dependent.  
- Improvement: sample **P points** evenly on a circle of radius **R** around the central pixel.  
- Interpolation of pixel values is required for points not aligned with the pixel grid.  
- Compute the number of transitions between 0 and 1 in the circular neighborhood.  

*Reference: T. Ojala, M. Pietikainen, T. Maenpaa, “Multiresolution gray-scale and rotation invariant texture classification with local binary patterns”, IEEE Transactions on PAMI, vol. 24, no. 7, pp. 971-987, July 2002.*

---

## Face Forensics and DeepFake
- Synthetic image generation and manipulation are rapidly advancing.  
- Raises concerns about security, trust in digital content, and the spread of false information.  
- **Face Forensics** studies techniques to detect synthetic faces and validate their authenticity (related to face verification).  

---

## Assignment 2: Fake Face Detection
- Goal: train, validate, and test a classifier to detect if a face is fake or real.  

### Sketch of the Pipeline
1. Prepare a dataset of real and fake faces.  
2. Detect the face region in each image (discard if multiple faces detected).  
   - Use OpenCV’s `cv2.CascadeClassifier`.  
3. Compute LBP images and histograms using `skimage.feature.local_binary_pattern()`.  
4. Save extracted features with **pickle** to avoid recomputation.  
5. Partition dataset by subjects (not images!):  
   - 60% train, 20% validation, 20% test.  
6. Train multiple classifiers from scikit-learn on the training set.  
7. Select the best model using the validation set (model selection).  
8. Test the chosen model on the test set and report results.  
9. Document experiments and prepare slides for the exam.  

---

## Dataset
- **Fake faces**: subset of FaceForensics dataset (cropped fake faces).  
  [Kaggle link](https://www.kaggle.com/datasets/greatgamedota/faceforensics?resource=download)  
- **Real faces**: Faces in the Wild dataset.  
  [Kaggle link](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/data)  
  - Apply OpenCV face detector to crop faces consistently.  
- Labeling:  
  - Fake = 1  
  - Real = 0  

---

## Model Selection
- Use the validation set to select the best feature configuration and classifier.  

### Feature Extraction Parameters
- `method = default, P = 256, R = 1` (original implementation).  
- `method = uniform, P = 8, R = 1`.  

### Preprocessing
- Validate the effect of data standardization (`StandardScaler`).  
- Compute mean/covariance only on training, then apply to validation/test.  

### Classifiers to Compare
- RandomForestClassifier  
- LogisticRegression  
- LinearSVC  

- Number of experiments = 12 (different combinations of features, preprocessing, classifiers).  
- Parameterize code for easier experimentation.  

---

## Model Selection Results
- Prepare a results table reporting accuracy values for each experiment:  

| Features      | Standardization | acc. RF | acc. LR | acc. LinearSVC |
|---------------|-----------------|---------|---------|----------------|
| LBP - default | NO              |         |         |                |
| LBP - uniform | NO              |         |         |                |
| LBP - default | YES             |         |         |                |
| LBP - uniform | YES             |         |         |                |

- Comment the results and select the best configuration.  

---

## Test
- Apply the selected model to the test set.  
- Report accuracy and confusion matrix.  
- Provide interpretation of the results.  

---

## Discussion
- Highlight **pros and cons** of the approach.  
- Suggest possible improvements.  
- Reflect on what could be changed with more time.  
- Include references/materials studied.  
- Be ready to explain **every line of code** during the exam.  
