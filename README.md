# Classification of Cancer Cells in Lymph Nodes Using Convolutional Neural Networks
## _MSc Mathematical Modeling in Modern Technologies – Diploma Thesis_
My MSc Thesis can be found [here](https://dspace.lib.ntua.gr/xmlui/handle/123456789/60646?show=full)

The aim of this project is to classify the degree of cancer cell spread from breast tissue to lymph nodes.  

The training of the Convolutional Neural Networks (CNNs) was based on the Breast MRI dataset provided by Duke University.  
This dataset was originally introduced in the paper [_A machine learning approach to radiogenomics of breast cancer: a study of 922 subjects and 529 DCE-MRI features_](https://pmc.ncbi.nlm.nih.gov/articles/PMC6134102/).  

---

## Preprocessing

- MRIs were provided in **NIFTI** format  
- MRIs were converted from **NIFTI to DICOM**  
- Based on annotation boxes, only the relevant slices of each MRI were retained  
- Each slice was converted into **PNG format** with a resolution of **512 × 512** pixels  

---

## Training

- Training was performed using an **NVIDIA Tesla A100 Ampere 40GB** GPU  
- The **Adam** and **AdamW** optimization algorithms were applied  
- The initial learning rate was **0.001**, reduced by a factor of 10 whenever the validation error did not decrease for 5 consecutive epochs  
- Training was terminated when the validation error failed to improve for 15 consecutive epochs  

---

## Evaluation

- Model evaluation was carried out on a **validation set**  
- Metrics considered: **Accuracy, Precision, Recall, and F1 Score**  
- Since the images were classified into **four classes**, **Macro, Micro, and Weighted averaging** methods were applied  

![Metrics Formulas](./Training%20and%20Evaluation/Models/Figures/Metrics_formulas.png)  

---

## Results  

### Adam  

![Results using Adam optimization algorithm](./Training%20and%20Evaluation/Models/Figures/Results_Adam.png)  

### AdamW  

![Results using AdamW optimization algorithm](./Training%20and%20Evaluation/Models/Figures/Results_AdamW.png)  

---

## Technologies

- PyTorch  
- Pandas  
- Pydicom  
- Matplotlib  

---

## License

MIT  
