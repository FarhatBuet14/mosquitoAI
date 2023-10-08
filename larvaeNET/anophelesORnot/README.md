## Deep Learning based Detection of Anopheles stephensi: Unraveling the Optimal Approach through Explainable AI

The detection of Anopheles stephensi mosquitoes is vital for malaria control and elimination efforts, public health protection, and research into combating this deadly disease. Timely and accurate detection informs targeted interventions, reduces disease transmission, and ultimately saves lives. This paper delves into these fields by harnessing the capabilities of deep learning AI models to detect stephensi. Furthermore, our research goes beyond mere detection, comparing two distinct classification methods: full body and wing-based classification, and integrates explainable AI techniques to enhance transparency and model interpretability.

What sets our study apart is the incorporation of explainable AI techniques, including Class Activation Mapping (CAM), Gradient-weighted Class Activation Mapping (grad-CAM), and t-Distributed Stochastic Neighbor Embedding (t-SNE). These techniques facilitate a deeper understanding of our models' decision-making processes and guide us in selecting the most interpretable and robust model for Anopheles stephensi detection.

## Dataset Details

![dataset.png](https://github.com/FarhatBuet14/mosquitoAI/blob/main/larvaeNET/anophelesORnot/images/dataset_details.png)

## Requirements
* Python 3.10.12
* Tensorflow 2.12
* Keras 2.12

## Test Result

### Model Architecture - *EfficientNET-B0*

### Confusion Matrix - Validation Set

| Actual / Predicted | not_ano | ano |
|--------------------|---------|-----|
| not_ano            | 48      | 0   |
| ano                | 0       | 48  |

Total Accuracy: 100%

### Confusion Matrix - Test Set

| Actual / Predicted | not_ano | ano |
|--------------------|---------|-----|
| not_ano            | 64      | 0   |
| ano                | 0       | 64  |

Total Accuracy: 100%
