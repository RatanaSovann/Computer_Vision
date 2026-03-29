# Transfer Learning on Food 101 Dataset
<img width="2334" height="1077" alt="image" src="https://github.com/user-attachments/assets/f3f7e6ee-1455-4edd-a802-cfbba4fece56" />
Full article: https://www.sciencedirect.com/science/article/pii/S2666285X22000334

## Objectives:
The goal of this project is to:
- Analyse architectural differences between GoogLeNet, MobileNetV3, and ResNet and evaluate their performance on the Food-101 Dataset.
- Apply transfer learning using pre-trained ImageNet weights to predict 101 food classes.
- Replace each model’s classification head with a custom 3-layer fully connected network for final prediction.
- Freeze all pre-trained layers and train only the new head on the Food-101 dataset.

## Data Understanding:
The Dataset used is Food101 dataset. This dataset consists of 101 food categories, with 101,000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. All images were rescaled to have a maximum side length of 512 pixels.

<img width="876" height="604" alt="image" src="https://github.com/user-attachments/assets/7e5cbe15-76ae-4260-becc-0931e164e1ad" />

The biggest challenges in classifying this dataset are:
- Intra-Class Variability: the same food can look very different depending on lighting, ingredients, presentation style, angle and background.
- Background Clutter: Images often include non-food objects such as plates, utensils, table settings, and hands.
- Unbalanced Visual Complexity: Some classes are easier (e.g., a hot dog is visually distinct), while others are extremely hard (e.g., chocolate cake vs brownie).
