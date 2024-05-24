# Precise Localization of Mosquito Larvae Body Parts with Faster R-CNN Deep Learning
It presents an application of the Faster Region-based Convolutional Neural Network (Faster R-CNN) in the field of entomology and vector-borne disease control. we introduce an innovative approach for the automated and precise localization of the distinct body parts of mosquito larvae, namely the head, thorax, abdomen, and tail.

## Dataset Details

![dataset.png](https://github.com/FarhatBuet14/mosquitoAI/blob/main/larvaeNET/Larvae Localization/larvae_full_body_localization/images/dataset_details.png)

## Requirements
* Python 3.10.12
* Tensorflow 2.12
* Keras 2.12
* detectron2

### Model Architecture - *Faster R-CNN* (base model - resnet50)

| Metric           | Value             |
|------------------|-------------------|
| AP               | 66.99857674606875 |
| AP-abdomen       | 67.54893542900362 |
| AP-head          | 83.1999215672602  |
| AP-lower         | 49.63484664050527 |
| AP-thorax        | 67.61060334750589 |
| AP50             | 96.1516422254115  |
| AP75             | 79.25430946651576 |
| APl              | 66.99857674606875 |

![test_results.png](https://github.com/FarhatBuet14/mosquitoAI/blob/main/larvaeNET/Larvae%20Localization/larvae_anatomy_localization/images/test_results.png)

We are also hosting this localizaion test process in the [website](https://mosquito-localization.web.app/) where you can upload an image and check the results with the bounding boxes within less than a minute.