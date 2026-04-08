---
sidebar_position: 1
slug: cnns
title: "Convolutional Neural Networks"
---


# Convolutional Neural Networks

:::info[What You'll Learn]
- How convolutions detect features in images
- Pooling layers and why they help
- Building CNN architectures (LeNet to ResNet)
- Training a CNN on real image data
:::

:::note[Prerequisites]
[Neural Networks Introduction](/curriculum/month-1/neural-networks-intro) and [Your First Model](/curriculum/month-1/your-first-model) from Month 1.
:::

**Estimated time:** Reading: ~30 min | Exercises: ~3 hours

Images are high-dimensional — a modest 224×224 RGB photo has over 150,000 input features. A fully connected layer mapping that to just 1,000 hidden units would need **150 million** parameters. Convolutional Neural Networks (CNNs) solve this by exploiting the spatial structure of images: nearby pixels are more related than distant ones, and the same pattern (an edge, a texture) can appear anywhere in the frame.

## The Convolution Operation

A convolution slides a small **kernel** (also called a **filter**) across the input and computes element-wise products followed by a sum at every position. The result is a **feature map** that highlights where the kernel's pattern appears.

:::note[Discrete 2D Convolution]

:::info[Plain English: What Does This Formula Mean?]
Imagine you have a tiny magnifying glass (the kernel) that you slide across a photograph. At every spot, the magnifying glass checks: "Does the pattern I'm looking for exist here?" It does this by multiplying each tiny square of pixels under the magnifying glass by a corresponding number in the kernel, and then adding all those products together. A big result means "yes, the pattern is here!"; a small result means "no match."
:::

For an input image \(I\) and kernel \(K\) of size \(k \times k\):

$$
(I * K)[i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I[i+m,\; j+n] \cdot K[m, n]
$$

**Reading the formula:** *\(I\)* is the input image (a grid of numbers). *\(K\)* is the kernel (a small grid of learnable weights). *\(i, j\)* is the position on the output feature map we're computing. *\(m, n\)* loop over every row and column inside the kernel. At each position, we multiply the image value \(I[i+m, j+n]\) by the kernel value \(K[m,n]\), then sum everything up. The *\(\sum\)* (sigma) just means "add up all these products."

In practice, deep learning frameworks compute **cross-correlation** (no kernel flip), but the literature calls it "convolution" by convention.
:::

### Why convolution works for vision

Three properties make convolution powerful for spatial data:

1. **Parameter sharing** — the same kernel is reused across every spatial location, so a single 3×3 kernel has only 9 learnable weights regardless of image size.
2. **Local connectivity** — each output neuron depends only on a small receptive field, not the entire image.
3. **Translation equivariance** — if the input shifts, the feature map shifts by the same amount, so the network detects features regardless of position.

## Filters, Stride, and Padding

### Filters (Kernels)

Each convolutional layer learns multiple filters. The first layer often learns simple edge detectors; deeper layers compose those into textures, parts, and objects.

:::info[Filter Dimensions]
A filter in layer \(\ell\) has shape \((C_{\ell-1},\; k_h,\; k_w)\), where \(C_{\ell-1}\) is the number of input channels. If the layer has \(C_\ell\) filters, the weight tensor is \((C_\ell,\; C_{\ell-1},\; k_h,\; k_w)\).
:::

### Stride

Stride controls how far the kernel moves between positions. A stride of 1 produces a feature map nearly the same spatial size as the input; a stride of 2 halves each dimension, acting as built-in downsampling.

### Padding

Without padding, each convolution shrinks the spatial dimensions by \(k - 1\). **"Same" padding** adds zeros around the border so the output size equals the input size (when stride = 1). **"Valid" padding** means no padding at all.

The output spatial dimension is:

:::info[Plain English: What Does This Formula Mean?]
Think of it like tiling a floor. You have a room (the input), tiles of a certain size (the kernel), and you can choose how far apart to place each tile (the stride). Padding is like extending the edges of the room with extra blank floor. This formula tells you how many tiles fit across the room given those choices.
:::

$$
\text{out} = \left\lfloor \frac{\text{in} + 2p - k}{s} \right\rfloor + 1
$$

**Reading the formula:** *in* is the input size (width or height in pixels). *p* is the padding (how many extra zeros we add on each side). *k* is the kernel size. *s* is the stride (how many pixels the kernel jumps each step). The *⌊ ⌋* (floor brackets) mean "round down to the nearest whole number." The formula calculates: take the input, add padding on both sides, subtract the kernel footprint, divide by the stride, round down, then add 1.

where \(p\) is padding and \(s\) is stride.

```python title="Convolution output size calculator"
import torch
import torch.nn as nn

# Example: 32x32 input, 5x5 kernel, stride 1, padding 2 ("same")
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
x = torch.randn(1, 3, 32, 32)
print(conv(x).shape)  # torch.Size([1, 16, 32, 32])

# Stride 2 halves spatial dimensions
conv_s2 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
print(conv_s2(x).shape)  # torch.Size([1, 16, 16, 16])
```

:::tip[Line-by-Line Walkthrough]
- **`import torch` / `import torch.nn as nn`** — Load PyTorch and its neural network module.
- **`nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)`** — Create a 2D convolution layer: 3 input channels (RGB), 16 output filters, each filter is 5×5, it slides 1 pixel at a time, and 2 pixels of zero-padding are added on each side (so output size = input size).
- **`torch.randn(1, 3, 32, 32)`** — Make a fake image: 1 image in the batch, 3 color channels, 32×32 pixels. Values are random.
- **`conv(x).shape`** — Pass the image through the convolution and print the output shape. With "same" padding, the spatial size stays 32×32 but we now have 16 channels (one per filter).
- **`nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)`** — A second conv layer with stride 2, meaning the kernel jumps 2 pixels at a time, halving the spatial dimensions to 16×16.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to a file, e.g. `conv_output_size.py`
2. Open a terminal and run: `python conv_output_size.py`

**Expected output:**
```
torch.Size([1, 16, 32, 32])
torch.Size([1, 16, 16, 16])
```

</details>

## Pooling Layers

Pooling reduces spatial dimensions and introduces a degree of translation invariance. The two most common variants:

- **Max pooling** — takes the maximum value in each window. Preserves the strongest activations.
- **Average pooling** — takes the mean. Smoother but can dilute strong signals.

Modern architectures often use **global average pooling** before the classifier: reduce each channel's feature map to a single scalar by averaging over all spatial positions.

```python title="Pooling examples"
pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(1, 16, 32, 32)
print(pool(x).shape)  # torch.Size([1, 16, 16, 16])

gap = nn.AdaptiveAvgPool2d(1)
print(gap(x).shape)  # torch.Size([1, 16, 1, 1])
```

:::tip[Line-by-Line Walkthrough]
- **`nn.MaxPool2d(kernel_size=2, stride=2)`** — Create a max-pooling layer with a 2×2 window that moves 2 pixels at a time. In each 2×2 block, it keeps only the largest value, cutting the spatial size in half.
- **`torch.randn(1, 16, 32, 32)`** — A fake feature map: 1 image, 16 channels, 32×32 spatial size.
- **`pool(x).shape`** — After max pooling, the spatial dimensions drop from 32×32 to 16×16.
- **`nn.AdaptiveAvgPool2d(1)`** — Global average pooling: no matter the input spatial size, this squishes it down to 1×1 by averaging all values in each channel.
- **`gap(x).shape`** — Output is 1×1 spatially — one number per channel.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to a file, e.g. `pooling_examples.py` (add `import torch; import torch.nn as nn` at the top)
2. Run: `python pooling_examples.py`

**Expected output:**
```
torch.Size([1, 16, 16, 16])
torch.Size([1, 16, 1, 1])
```

</details>

## Classic CNN Architectures

### LeNet-5 (1998)

Yann LeCun's pioneering architecture for handwritten digit recognition. Two convolutional layers followed by three fully connected layers. Small by today's standards (~60K parameters) but established the conv → pool → fc pattern.

### AlexNet (2012)

The architecture that reignited deep learning. Key innovations: ReLU activations instead of sigmoid/tanh, dropout for regularization, GPU training, and data augmentation. Won ImageNet with a 10+ percentage-point margin.

### VGGNet (2014)

Demonstrated that depth matters: stack many 3×3 conv layers instead of fewer large kernels. Two stacked 3×3 layers have the same receptive field as one 5×5 layer but with fewer parameters and more non-linearity.

### ResNet (2015)

:::info[Skip Connections]
ResNet introduced **residual connections**: the output of a block is \(F(x) + x\), where \(F\) is the learned transformation. This lets gradients flow directly through the identity path, enabling training of networks with 100+ layers.
:::

ResNet-50 remains one of the most widely used backbones in computer vision. Its residual blocks come in two flavors: the basic block (two 3×3 convs) and the bottleneck block (1×1 → 3×3 → 1×1).

## Building a CNN in PyTorch

Let's build a CNN for CIFAR-10 classification (10 classes, 32×32 RGB images).

```python title="CNN for CIFAR-10"
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 8x8 -> 1x1
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


model = CIFAR10CNN()
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# ~175K parameters — small and trainable on a laptop
```

:::tip[Line-by-Line Walkthrough]
- **`class CIFAR10CNN(nn.Module)`** — Define a new neural network class that inherits from PyTorch's base `Module`.
- **`self.features = nn.Sequential(...)`** — Stack all the feature-extraction layers in order. The image flows through each layer top-to-bottom.
- **`nn.Conv2d(3, 32, 3, padding=1)`** — First conv layer: takes 3-channel (RGB) input, produces 32 feature maps using 3×3 kernels, with padding to keep the size the same.
- **`nn.BatchNorm2d(32)`** — Normalize the 32 feature maps so training is more stable and faster.
- **`nn.ReLU()`** — Activation function: replace all negative values with zero (keeps only the "excited" neurons).
- **`nn.MaxPool2d(2, 2)`** — Shrink the spatial size by half (32×32 → 16×16) by keeping only the max in each 2×2 block.
- **`nn.AdaptiveAvgPool2d(1)`** — Collapse the final feature maps to a single number per channel by averaging.
- **`self.classifier = nn.Linear(128, num_classes)`** — A fully connected layer that maps the 128 features to 10 class scores.
- **`x.view(x.size(0), -1)`** — Flatten the 3D feature map into a 1D vector so it can go into the linear layer.
- **`sum(p.numel() for p in model.parameters())`** — Count every learnable number in the entire model.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch
```

**Steps:**
1. Save the code to `cifar10_cnn.py`
2. Run: `python cifar10_cnn.py`

**Expected output:**
```
Parameters: 175,178
```
(The exact number may vary slightly depending on PyTorch version.)

</details>

### Training Loop

```python title="Training the CIFAR-10 CNN"
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CIFAR10CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Epoch {epoch+1:02d} | Loss: {running_loss/len(train_loader):.4f} | Acc: {correct/total:.4f}")
```

:::tip[Line-by-Line Walkthrough]
- **`T.Compose([...])`** — Chain multiple image transformations into a pipeline. Training images get random flips and crops (data augmentation); test images just get normalized.
- **`T.RandomHorizontalFlip()`** — Randomly mirror the image left-to-right (50% chance). This teaches the model that a cat facing left is still a cat.
- **`T.RandomCrop(32, padding=4)`** — Pad the image by 4 pixels on each side, then randomly crop back to 32×32. This simulates slight shifts in position.
- **`T.Normalize((0.4914, ...), (0.2470, ...))`** — Subtract the dataset mean and divide by the standard deviation for each RGB channel. This centers the data around zero.
- **`torchvision.datasets.CIFAR10(..., download=True)`** — Download (first time only) and load the CIFAR-10 dataset with our transforms applied.
- **`DataLoader(..., batch_size=128, shuffle=True)`** — Serve the data in batches of 128 images, shuffled each epoch so the model doesn't memorize the order.
- **`model.train()`** — Put the model in training mode (enables dropout, batch norm uses batch stats).
- **`optimizer.zero_grad()`** — Clear old gradients before computing new ones (PyTorch accumulates gradients by default).
- **`loss = criterion(model(images), labels)`** — Forward pass + compute how wrong the predictions are (cross-entropy loss).
- **`loss.backward()`** — Backpropagation: compute gradients of the loss with respect to every parameter.
- **`optimizer.step()`** — Update the parameters using the computed gradients.
- **`model.eval()`** — Switch to evaluation mode (disables dropout, batch norm uses running stats).
- **`torch.no_grad()`** — Don't track gradients during evaluation (saves memory and speed).
- **`model(images).argmax(dim=1)`** — Get the predicted class by picking the index with the highest score.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision
```

**Steps:**
1. Save both the `CIFAR10CNN` class and this training code into one file, e.g. `train_cifar10.py`
2. Run: `python train_cifar10.py`
3. The CIFAR-10 dataset (~170 MB) will be downloaded automatically on first run.

**Expected output:**
```
Epoch 01 | Loss: 1.3245 | Acc: 0.6823
Epoch 02 | Loss: 0.9187 | Acc: 0.7654
...
Epoch 20 | Loss: 0.2541 | Acc: 0.9012
```
Training takes ~5–15 minutes on a modern laptop CPU, faster with a GPU.

</details>

:::tip[Expected Results]
This architecture should reach **~90% test accuracy** on CIFAR-10 within 20 epochs. Adding more aggressive data augmentation, a cosine learning rate schedule, or more depth can push it past 93%.
:::

## Exercises

<ExerciseBlock title="Visualize Learned Filters" difficulty="beginner" hints={["Access the first conv layer weights with model.features[0].weight.data", "Normalize each filter to [0, 1] for display", "Use matplotlib's imshow with a grid layout"]}>

After training the CIFAR-10 CNN, extract and visualize the 32 learned filters from the first convolutional layer. What patterns do you see? Do any resemble edge or color detectors?

</ExerciseBlock>

<ExerciseBlock title="Implement a Residual Block" difficulty="intermediate" hints={["The residual connection adds the input to the block output: out = F(x) + x", "If channel dimensions change, use a 1x1 conv as a projection shortcut", "Place BatchNorm after each conv, and ReLU after each BatchNorm"]}>

Implement a `ResidualBlock` module in PyTorch. It should support:
- Two 3×3 convolutions with batch normalization and ReLU
- A skip connection that adds the input to the output
- A projection shortcut (1×1 conv) when the number of channels changes or stride > 1

Then replace the plain conv blocks in the CIFAR-10 CNN with your residual blocks and compare accuracy.

</ExerciseBlock>

<ExerciseBlock title="Receptive Field Analysis" difficulty="advanced" hints={["The receptive field grows by (kernel_size - 1) × product_of_previous_strides at each layer", "Pooling with kernel 2 and stride 2 doubles the effective stride", "For our 3-block CNN, compute layer by layer"]}>

Calculate the theoretical receptive field of the final feature map in our CIFAR-10 CNN. Walk through each convolutional and pooling layer, tracking how the receptive field grows. Is the final receptive field large enough to "see" the entire 32×32 input?

</ExerciseBlock>

## Resources

<ResourceCard title="CS231n: Convolutional Neural Networks for Visual Recognition" url="https://cs231n.stanford.edu/" type="course" author="Stanford / Andrej Karpathy" description="The gold-standard course on CNNs and computer vision. Lecture notes are freely available." />

<ResourceCard title="Deep Residual Learning for Image Recognition" url="https://arxiv.org/abs/1512.03385" type="paper" author="Kaiming He et al." description="The ResNet paper — introduced skip connections and enabled training of 100+ layer networks." />

<ResourceCard title="CNN Explainer" url="https://poloclub.github.io/cnn-explainer/" type="tool" description="Interactive visualization that lets you watch data flow through a CNN in real time." />

<ResourceCard title="PyTorch Vision Models" url="https://pytorch.org/vision/stable/models.html" type="tutorial" description="Official documentation for all pretrained models available in torchvision." />
