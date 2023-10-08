# Gender Classification of Adult Mosquitoes with EfficientNet: A Comprehensive Analysis Using Explainable AI

This work introduces a novel and impactful application of deep learning technology, specifically the EfficientNet architecture, for the automated classification of mosquito genders. Mosquitoes, as vectors of devastating diseases, require careful monitoring and control. Accurate identification of mosquito genders is vital for studying their behavior, and population dynamics, and designing targeted vector control strategies.

Our research showcases a sophisticated approach that leverages the power of EfficientNet, a cutting-edge deep-learning model, to classify adult mosquitoes by gender. We employ a diverse dataset containing images of male and female mosquitoes across various species and environments, ensuring the model's robustness and adaptability.

What distinguishes our study is the comprehensive exploration of explainable AI techniques, including Class Activation Mapping (CAM), Gradient-weighted Class Activation Mapping (grad-CAM), and t-Distributed Stochastic Neighbor Embedding (t-SNE). These techniques offer valuable insights into the decision-making processes of our deep learning model, guiding us in selecting the most interpretable and effective approach.

This research not only advances the field of entomology but also underscores the significance of explainable AI techniques in model selection and evaluation. By harnessing the capabilities of state-of-the-art deep learning models and making them interpretable, we contribute to more effective mosquito population monitoring and disease prevention efforts, ultimately enhancing global health outcomes.

## Dataset Details

![dataset_details.png](https://github.com/FarhatBuet14/mosquitoAI/blob/main/adultNET/Gender%20Classification/images/dataset_details.png)

## Requirements
* Python 3.10.12
* Tensorflow 2.12
* Keras 2.12

## Test Results

### Model Architecture - *EfficientNET-B0*

### Confusion Matrix - Validation Set

| Actual / Predicted | Male | Female | Accuracy (%) |
|--------------------|------|--------|--------------|
| Male               | 17   | 1      | 94.44        |
| Female             | 2    | 16     | 88.88        |

Total Accuracy: 91.62%

### Confusion Matrix - Test Set

| Actual / Predicted | Male | Female | Accuracy (%) |
|--------------------|------|--------|--------------|
| Male               | 14   | 4      | 77.77        |
| Female             | 1    | 17     | 94.44        |

Total Accuracy: 86.11%

### Test Grad-CAMs

![test_results.png](https://github.com/FarhatBuet14/mosquitoAI/blob/main/adultNET/Gender%20Classification/images/test_results.png)
