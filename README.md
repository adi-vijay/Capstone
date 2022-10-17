![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png)
# Honey Bee Image Classification: Subspecies and Health Prediction

#### Author: Aditya Vijay
--------------------------

## Problem Statement
Honey bees are an often overlooked, but essential component in global agriculture. In the United States alone, honey bees contribute upwards of [$15 billion](https://www.usda.gov/media/blog/2017/06/20/being-serious-about-saving-bees) to agricultural production through pollination. Today, commercial beekeepers earn the majority of their revenue through leasing their colonies for pollination of large-scale agricultural operations. Both commercial and non-commercial bee hives are in greater danger today than ever before, with beekeepers losing an average of [approximately 45.5%](https://ocm.auburn.edu/newsroom/news_articles/2021/06/241121-honey-bee-annual-loss-survey-results.php) of their managed hives in between 2020 and 2021. Some of these colonies succumb to [Colony Collapse Disorder](https://www.epa.gov/pollinator-protection/colony-collapse-disorder), whereas others become [Africanized](https://extension.uga.edu/publications/detail.html?number=B1290&title=africanized-honey-bees) and are no longer managable or safe. As the number of beehives in the country has decreased from over 6 million in the 1940's to less than 3 million today, ensuring cohesiveness and a high quality of health in the hive is paramount.

This project aims to create a solution for beekeepers who are unsure of the quality of their hives by using Convolutional Neural Networks (CNN) to classify the health and subspecies of a honey bee based on its image. Separate CNNs are trained on the [Honey Bee Annotated Image Dataset](https://www.kaggle.com/jenny18/honey-bee-annotated-images) from Kaggle to identify the subspecies and potential health issues of a honey bee, and are deployed in a convenient web application for any user to predict using their own images.
______________________________________


## Executive Summary

Image data along with corresponding annotations on fields of health, subspecies, pollen carrying status, time of day of image capture, bee caste, and location were obtained from the [Honey Bee Annotated Image Dataset](https://www.kaggle.com/jenny18/honey-bee-annotated-images) from Kaggle. The dataset contains 5,100+ images extracted from still time-lapse videos of bees. Background images were averaged and subtracted from each frame to accentuate individual bees. A portion of the bees do not have a known species.

All training, testing, and implementations of models and web apps were done through Google Colab, a collaborative cloud-based python IDE akin to Jupyter Notebook with access to GPUs. The entire `Capstone` repo should be cloned and uploaded to the default main folder of Google Drive for pathnames to reference files properly.

**Training images are not provided in this repo due to size. Download from [Kaggle](https://www.kaggle.com/jenny18/honey-bee-annotated-images), unzip, rename folder to `bee_imgs` if not already named so, and move the entire folder inside the `data` folder in cloned repo**


Multiple CNNs were trained separately, using subspecies and health as targets. Models were initially made using a CNN with 2 convolutional layers and 1 MaxPooling layer for reducing dimensionality, and improved upon using dropout layers, real-time image augmentation, and custom learning rate modifications. Dropout layers helped to avoid overfitting while image augmentation provided multiple transformations of each image for further training, creating a more robust network. Learning rate modifications were adapted from a thread on StackOverflow to reduce the learning rate as epochs progress. Images in their provided state were not of uniform shape, so image pre-processing was done before fitting in order to scale all images to 100x100 px, with 3 color channels. The Streamlit implementation uses the `PIL.Image` libarary to convert images from RGBA to RGB.

The image annotations showed imbalanced classes for both targets, subspecies and health, with the lowest ratio of majority class to non-majority class at about 6:1. Though there are multiple ways to balance classes, the `class_weight` parameter was used during fitting of the CNNs to account for imbalances. Class weights were computed using the `compute_class_weights()` function from `sklearn.utils`. Models were trained with and without balancing classes, and the final implementation of the models in Streamlit use the balanced class models.

Image pre-processing steps were taken by the authors of the dataset to reduce background and accentuate individual bees, however exact details of these steps are not available and as such are not replicable. Further elucidation of these methods can help improve the models, as background brightness and contrast with bees is a significant factor in accurate prediction. Images with similar background and bee colors are likely to be misclassified, or inconsistent in classification.


The web app is the full implementation of image pre-processing and use the balanced health and subspecies CNN models found in this repo. The web app can be run by opening the `streamlit.ipynb` notebook and selecting 'Run All' after uploading this repo to your Google Drive.



## Conclusion

Separate convolutional neural networks were successfully trained to predict health status and honey bee subspecies, as evidence by their improvement upon the baseline accuracies. The best performing health prediction model scored **86.09%, an improvement from the baseline majority class prediction accuracy of 65.43%**. The best performing subspecies prediction model scored **89.27%, an improvement from the baseline majority class prediction accuracy of 58.16%**. Misclassifications are often on images with similar background and bee colors. To ensure accuracy, users should upload images with high contrast between the bee and the background.

Additionally, this model is limited, as it was trained on a relatively small dataset concentrated in specific regions of the United States. It is unlikely to correctly predict the subspecies and health statuses of images of bees from outside of the country, or in areas with unique health conditions/disorders.


These models and their web app implementation provide US beekeepers with a quick and easy method for identification of bees and their health status in their apiary, without having to open or disturb the hive. This is helpful for on-the-fly predictions of probable subspecies and health statuses of bees, but should not be considered as a replacement for established beekeeping and biological methods for identification of subspecies and impaired health.

## Recommendations

Further development of the CNNs might include adding convolutional layers and pooling layers. Additionally, transfer learning using pre-existing image identification models can be far more robust than ones built from scratch.

Training the CNNs on larger datasets will likely result in more accurate predictions. Having images be of similar sizes can also improve model performance, as fewer images will have to be scaled.

As it stands, this model is useful for beekeepers and those who wish to predict qualities of individual bee images. However, due to the nature of bees, it's fairly tedious to capture an image of solitary bees, one by one. Further developments of this solution could be to use a computer vision library to identify and extract images of bees from short videos, or to use an additional library/CNN to extract individual bees from a crowded image.



## Sources

- [Honey Bee Annotated Image Dataset](https://www.kaggle.com/datasets/jenny18/honey-bee-annotated-images)
- [Being Serious about Saving Bees](https://www.usda.gov/media/blog/2017/06/20/being-serious-about-saving-bees)
- ['Like sending bees to war': the deadly truth behind your almond milk obsession](https://www.theguardian.com/environment/2020/jan/07/honeybees-deaths-almonds-hives-aoe)
- [US beekeepers continue to report high colony loss rates, no clear progression toward improvement ](https://ocm.auburn.edu/newsroom/news_articles/2021/06/241121-honey-bee-annual-loss-survey-results.php)
- [Africanized Honey Bees](https://extension.uga.edu/publications/detail.html?number=B1290&title=africanized-honey-bees)
- [Colony Collapse Disorder | US EPA](https://www.epa.gov/pollinator-protection/colony-collapse-disorder)



## Libraries and Packages

 - `numpy`
 - `pandas`
 - `matplotlib`
 - `seaborn`
 - `sklearn`
 - `skimage`
 - `imageio`
 - `PIL` Python Imaging Library [github](https://github.com/python-pillow/Pillow)
 - `tensorflow`
 - `keras`
 - `ann-visualizer` [github](https://github.com/RedaOps/ann-visualizer)
 - `streamlit`
 - `localtunnel`