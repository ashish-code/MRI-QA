## Project Under Development 

# MRI-QA
The assessment of quality of MRI is in important precursor to ameliorating biases in subsequent data analysis and precluding erroneous acquisitions. Since manual visual inspection is impractical for large volumes of data, as well as being subjective in nature, it is very useful to automate the task of image quality assessment (IQA). There is existing research on IQA, that typically utilize hand-crafted visual features in conjunction with machine learning algorithms. However, these techniques have been made obsolete in many research fields by much more powerful and expressive deep learning based models. We introduce a model that uses a Convolutional Neural Network for  feature extraction from 3D volumes of MRI data of a subject and a Fully Connected Network for classification of the quality of the MRI. This model is trained on a multi-site freely available dataset, called ABIDE 1, used in study of Autism. By utilizing two of the seventeen sites as hold-out data, we demonstrate that our model achieves state-of-art performance on unseen data from novel sites. Furthermore, we evaluate our trained model on a MRI dataset from TCIA, used in study of Glioblastoma, to demonstrate the ability of our model to effectively adapt to different types of neuro-imaging data.

## Introduction
Image analysis on data with artifacts can lead to misleading diagnosis. It is therefore very important that data be processed for quality control prior to analysis. MRI data is rarely completely devoid of artifacts. The assessment of their quality has been a challenging research topic for a long time. The traditional approach is to inspect MRI images visually by one or more experts; the images of unacceptably poor quality are pruned out. Manual assessment is costly in terms of time and subject to a degree of ambiguity due to subjective differences between raters. There is also intra-rater variation due to fatigue. Although MRI acquisition devices are regularly inspected, they do nevertheless tend to drift from their calibrated settings. All of these conditions underscore the importance of reliable quality control at the preliminary stages of the processing pipeline for diagnosis.

The principal challenges facing quality assessment are: absence of a universally accepted quantitative definition of quality metrics; variation in expert rating for the same MRI; inter-site variation in acquisition creating uncharacteristic artifacts. Quality assessment methods in literature are typically grouped into 3 types: Full-Reference (FR), where the original and degraded image pair are available for training a model; Partial-Reference (PR), where some information from the original image is available along with the degraded image; No-Reference (NR), where the original and degraded pair are never available. Our focus is NR quality assessment, also called Blind quality assessment, since there is no known dataset of the same MRI with and without artifacts.

## Contributions

-- We present the first attempt at employing a 3D CNN based deep learning approach to image quality assessment in MRI data. Our approach leverages the state-of-art ability of deep learning at both accurate representation of artifacts in MRI and transferability of the trained model to other sites and even different types of neuro-imaging data. The only meaningfully similar work to our knowledge utilized hand-crafted features called Image Quality Metrics and basic machine learning classifier like Random Forest \cite{Esteban2017}.

-- While other methods look to assess quality in 2D image slices, our model is trained with 3D volumes, which makes is inherently better suited to model information between slices and adapt to different types and locations of artifacts.

-- The backbone of our model is trained using medical images sources from numerous sites and multiple modalities. We then fine-tune our model on ABIDE 1 dataset. This approach allows our pre-trained model to be flexible; it can be directly used for quality assessment on data from a different site; and it can also be alternatively optimized for a specific data acquisition modality.

--------------------------------------------------------------------------------------------------------------------------

## IQMs
We have utilized Image Quality Metrics (IQMs) as features and build a FCN for quality assessment on ABIDE-1 and DS030 MRI datasets.
Overview of the IQMs we used is tabulated below
![Image Quality metrics](https://www.dropbox.com/s/y77ergfdclwh3lh/iqms.png?dl=0)

The FCN network architecture is
![FCN model architecture](https://www.dropbox.com/s/sh6vbu8r0bcmde6/network_arch.PNG?dl=0)

The current training, validation and testing performance is
![Training and Validation Performance](https://www.dropbox.com/s/3k23k4quj3lo3bj/performance.png?dl=0)

The aggregate performance is:
![Quality Assessment performance](https://www.dropbox.com/s/sfzv1hcwdhg8p76/results_table.png?dl=0)
