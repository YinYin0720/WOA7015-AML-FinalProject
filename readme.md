# 1.Folder Description

## result

- **Model Folders (VGG16, AlexNet, ResNet18, Transformer, Transformer_Gray)** 

  Each folder contains the following files:

  - **events.out.tfevents.xxxx**: TensorBoard file recording the training process logs.
  - **f_t.csv**: Training phase F1 scores for each epoch.
  - **f_v.csv**: Validation phase F1 scores for each epoch.
  - **l_t.csv**: Training phase Loss values for each epoch.
  - **l_v.csv**: Validation phase Loss values for each epoch.

  > Note: F1 scores are used as a substitute for Accuracy to plot the training and validation performance curves.


## metrics_plot

- **all_result.xlsx** 

  Contains evaluation metrics for the training and validation phases of each model at the last epoch, including F1 score, precision, recall, sensitivity, and confusion matrix details.

- **result.ipynb**

  analyze and visualize the performance metrics and confusion matrices of different machine learning models, including both baseline and transformer-based architectures.



## img
- **gradcam_visualization_s1.png**  
  
  Image displays the Grad-CAM visualization used as an example in Scenario 1 of the report. 
  
- **gradcam_visualization_s2.png** 
  
  Image displays the Grad-CAM visualization used as an example in Scenario 2 of the report.



## pth

- **TransformerGRY_68_f1_0.9786.pth**  
  
  Contains the saved weights of the Transformer GRY model, with an F1 Score of 0.9786.
  
- **Transformer_49_f1_0.9847.pth** 
  
  Contains the saved weights of the Transformer model, with an F1 Score of 0.9847.
  
- **best_VGG16_51_f1_0.9977.pth** 
  
  Contains the saved weights of the VGG16 model, with an F1 Score of 0.9977.
  
- **best_ResNet18_50_f1_0.9970.pth** 
  
  Contains the saved weights of the ResNet18 model, with an F1 Score of 0.9970.
  
- **best_AlexNet_53_f1_0.9931.pth** 
  
  Contains the saved weights of the AlexNet model, with an F1 Score of 0.9931.



# 2.Code Description

## Model Inference and Metrics Plotting Code

- **Link**: [Model_Inference_and_Metrics_Plotting](https://www.kaggle.com/code/liewliew/7015-aa-a13469-1aadfd)
- **File**: `Model_Inference_and_Metrics_Plotting.ipynb`
- **Content**: 
  - Best models selected from all training epochs based on validation F1 scores.
  - Further analysis on model inference speed and performance on the validation dataset.
  



## All Model Training Code

- **Link**: [All_Model_Training_Code](https://www.kaggle.com/code/liewliew/fork-of-last-version-final)
- **File**: `All_Model_Training_Code.ipynb`
- **Content**:
  - Contains training code for all models.
  - Includes the complete workflow for training models.



## Evaluation Metrics Along Epochs Plot Code

- **File**: `Evaluation_Metrics_Along_Epochs_PLot.ipynb`
- **Content**:
  - Visualizes evaluation metrics across all training epochs.
  - Analyzes trends in metrics during the training process.



## Preoptimized Baseline Model Training Code

- **File**: `Preoptimized_Baseline_Model_Training_Code.ipynb`
- **Content**:
  - Includes training code for unoptimized baseline models.
  - Displays results and visualizations



# 3.Notes

1. Use the **Result Folder** to evaluate model performance and trends during training and validation. 
2. Refer to the [Google Drive Link](https://drive.google.com/drive/folders/1T_-Qh7l4RObrnOfmj7lFhlpJBnowP3M6?usp=drive_link) to download the **Pth Folder** for pre-trained weights of the best-performing models. 
3. Check **Project Files** for training workflows, metric visualizations, and baseline performance analysis.





