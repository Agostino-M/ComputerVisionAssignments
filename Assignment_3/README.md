# Classify Real vs AI-Generated Faces with Deep Learning - 3° Assignment
> **Disclaimer:** Only the text content from the original PDF has been automatically extracted with PyMuPDF. Images have been omitted. This Markdown file is for demonstration purposes only and is **not intended to replace the original PDF**.


## Task
Develop a binary image classification model to distinguish between:  
- **Real faces**  
- **AI-generated faces**  

---

## Dataset Construction
You need to build your own dataset from Internet resources.  
- Use multiple datasets and decide how to best organize your data.  
- Resize the images according to your computational resources.  
- You can use any number of images for training and testing.  

### Real Faces
1. **FFHQ (Flickr-Faces-HQ)**  
   - ~70,000 high-quality images (1024×1024) of real human faces.  
   - Includes diverse age, ethnicity, lighting, and background.  
   - [Official GitHub](https://github.com/NVlabs/ffhq-dataset)  
   - [Preprocessed on Kaggle (512×512, 21 GB)](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq)  

2. **LFW (Labeled Faces in the Wild)**  
   - Already used in the second assignment.  

### AI-Generated Faces
1. **This Person Does Not Exist (TPDNE)**  
   - >10,000 synthetic face images generated with StyleGAN2.  
   - [Kaggle link](https://www.kaggle.com/datasets/almightyj/person-face-dataset-thispersondoesnotexist)  

2. **All These People Don’t Exist**  
   - More StyleGAN2-generated synthetic faces with varying quality levels.  
   - *Note: May overlap with TPDNE dataset.*  
   - [Kaggle link](https://www.kaggle.com/datasets/bwandowando/all-these-people-dont-exist)  

3. **AI-Generated Faces – Kaggle**  
   - A collection of GAN-generated human faces.  
   - [Kaggle link](https://www.kaggle.com/datasets/chelove4draste/ai-generated-faces)  

---

## Dataset Organization and Experimental Protocol
- Decide strategies for:  
  - Balancing the dataset.  
  - Data augmentation.  
  - Preprocessing.  
- **Splits:**  
  - 70% training  
  - 15% validation  
  - 15% test  
- Ensure the **test set is balanced**.  

---

## Model Architecture
- Combine **convolutional layers** and **attention mechanism**.  
- Run experiments with **and without attention** to evaluate its contribution.  
- Keep the architecture simple (avoid unnecessary complexity).  

---

## Training
- Monitor **validation loss** for early stopping when performance plateaus.  
- Handle dataset imbalance by:  
  - Ensuring both classes are evenly represented, or  
  - Applying **class weighting**.  
- Optionally, experiment with **focal loss**.  

---

## Evaluation
Evaluate your model using the following metrics:  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1-Score**  
- **Confusion Matrix**  