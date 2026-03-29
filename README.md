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

## Vision Model Architecture: 
### 1. GoogleLeNet

The main idea behind GoogleLeNet is a multi-scale feature extraction by having 4 blocks operate on the same input, but they process inputs at different scales and types of operations:
- Some look for fine details
- Some look for larger patterns
- Some smooth out and compress the signal.
This creates a richer feature representation when all branches are concatenated together.

<p align ='center'><img width="568" height="145" alt="image" src="https://github.com/user-attachments/assets/195a78f8-0782-40f0-93c5-9703a12853ff" /><p align ='center'>

This architecture starts with convolution and pooling layers that quickly reduce the image size while extracting basic features like edges and textures. The model then uses Inception modules, which process the input in parallel using different filter sizes (1×1, 3×3, 5×5, and pooling) to capture patterns at multiple scales at the same time. Between these modules, additional max-pooling layers further reduce spatial dimensions, making the network more efficient while keeping important features. Overall, the design allows the model to learn both fine and coarse details efficiently without becoming too computationally heavy.

### 2. MobileNet V3
MobileV3 is designed for mobile and edge devices that demands more efficiency and lower computational cost.

This architecture combines efficient convolution blocks with smart feature extraction. The Conv2dNormActivation layer applies convolution, batch normalization, and a non-linear activation (ReLU or Hardswish) to learn stable and meaningful features.

Some layers also include Squeeze-and-Excitation (SE) blocks, which help the model focus on the most important features by weighting channels. Instead of traditional pooling, the model uses strided convolutions to reduce spatial size, keeping the architecture lightweight while still learning rich representations.

### 3. Resnet50
Instead of learning the full feature mapping, Resnet is designed to learn from residuals (the change needed from input and not the entire transformation). ResNet 50 has 50 layers, which provide a balance between training time and accuracy compared to deeper model like Resnet101 (101 layers).
  
The main idea behind ResNet is skip connections—learning from residuals—which allows the gradient to flow through while bypassing the ReLU during backpropagation. This enables deeper layers to be constructed, creating a richer model.

More indepth detail can be found: [transfer_learning.ipynb]()
