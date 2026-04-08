---
sidebar_position: 5
slug: transfer-learning
title: "Transfer Learning"
---


# Transfer Learning

Training a deep network from scratch requires large datasets and significant compute. **Transfer learning** lets you leverage a model that was already trained on a massive dataset (like ImageNet's 1.2 million images) and adapt it to your specific task — even if you only have a few hundred examples.

## What Is Transfer Learning and Why It Works

Neural networks learn hierarchical features. Early layers of a CNN trained on ImageNet learn universal patterns — edges, textures, colors, simple shapes — that are useful for virtually any visual task. Middle layers learn compositional features like object parts. Only the final layers learn task-specific concepts (e.g., "this is a golden retriever").

:::info[The Key Insight]
Features learned on one task often transfer to other tasks, especially when the data domains are related. A model trained to recognize 1,000 ImageNet categories has already learned excellent representations for textures, shapes, and spatial relationships. Those representations are a powerful starting point for medical imaging, satellite imagery, product classification, and more.
:::

Transfer learning works best when:
- Your target dataset is **small** (hundreds to low thousands of images).
- Your target domain is **related** to the source domain (both are natural images, for instance).
- You need to train **quickly** or with limited compute.

Even when the domains differ (natural images → medical X-rays), transfer learning usually outperforms training from scratch, because the low-level features still transfer.

## Feature Extraction vs. Fine-Tuning

There are two main strategies for transfer learning:

### Strategy 1: Feature Extraction

Freeze all layers of the pretrained model and only train a new classifier head on top. The pretrained model acts as a fixed feature extractor.

**When to use:** Small dataset, high risk of overfitting, source and target domains are similar.

```python title="Feature extraction with a frozen backbone"
import torch
import torch.nn as nn
import torchvision.models as models

# Load pretrained ResNet-50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier head
num_classes = 5
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Only model.fc parameters have requires_grad=True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
# Trainable: 10,245 / 25,567,173 (0.04%)
```

:::tip[Line-by-Line Walkthrough]
- **`models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)`** — Load ResNet-50 with pretrained weights from ImageNet (version 2, which has better accuracy). The model already knows how to recognize edges, textures, shapes, and objects.
- **`for param in model.parameters(): param.requires_grad = False`** — Freeze every layer: tell PyTorch "don't update these weights during training." This preserves the pretrained knowledge.
- **`model.fc = nn.Linear(model.fc.in_features, num_classes)`** — Replace the final classification layer (originally 1000 ImageNet classes) with a new one for your task (5 classes). New layers have `requires_grad=True` by default.
- **`sum(p.numel() for p in model.parameters() if p.requires_grad)`** — Count only the trainable parameters. With freezing, only the new head (~10K params) is trainable out of ~25M total — that's 0.04%!
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision
```

**Steps:**
1. Save to `feature_extraction.py`
2. Run: `python feature_extraction.py`
3. The pretrained weights (~100 MB) will be downloaded automatically on first run.

**Expected output:**
```
Trainable: 10,245 / 25,567,173 (0.0%)
```

</details>

### Strategy 2: Fine-Tuning

Start from the pretrained weights but allow some or all layers to update during training. Typically you:

1. Replace the classifier head.
2. Train only the head for a few epochs (warm-up).
3. Unfreeze deeper layers and train the entire network with a smaller learning rate.

**When to use:** Larger dataset, or source and target domains differ significantly.

```python title="Fine-tuning with differential learning rates"
import torch.optim as optim

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Different learning rates: low for pretrained layers, high for new head
optimizer = optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.layer3.parameters(), "lr": 5e-6},
    {"params": model.fc.parameters(), "lr": 1e-3},
], weight_decay=0.01)

# Freeze earlier layers entirely
for name, param in model.named_parameters():
    if "layer1" in name or "layer2" in name or "conv1" in name or "bn1" in name:
        param.requires_grad = False
```

:::tip[Line-by-Line Walkthrough]
- **`model.fc = nn.Linear(model.fc.in_features, num_classes)`** — Replace the head, same as feature extraction.
- **`optim.AdamW([{...}, {...}, {...}], weight_decay=0.01)`** — Create an optimizer with *different learning rates* for different parts of the model. This is called "discriminative learning rates."
- **`"params": model.layer4.parameters(), "lr": 1e-5`** — The deepest pretrained block (layer4) gets a tiny learning rate (0.00001) — it already has good features, so we adjust them gently.
- **`"params": model.layer3.parameters(), "lr": 5e-6`** — Even smaller LR for the middle block — these features are more general and need less change.
- **`"params": model.fc.parameters(), "lr": 1e-3`** — The new head gets a 100× larger learning rate (0.001) — it needs to learn from scratch.
- **`for name, param in model.named_parameters():`** — Loop through all parameters by name to selectively freeze the earliest layers (layer1, layer2, conv1, bn1). These learn very general features (edges, textures) that transfer well and don't need updating.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision
```

**Steps:**
1. Save to `fine_tuning.py` (define `num_classes = 5` at the top)
2. Run: `python fine_tuning.py`

**Expected output:** No printed output — this sets up the model and optimizer. Use in a training loop.

</details>

:::tip[Discriminative Learning Rates]
Use **lower learning rates** for earlier pretrained layers (they already encode good features) and **higher learning rates** for the new head (it needs to learn from scratch). A common ratio is 10x between the head and the backbone. This prevents catastrophic forgetting of useful pretrained features.
:::

### When to Use Which Strategy

| Scenario | Dataset Size | Domain Similarity | Strategy |
|----------|-------------|-------------------|----------|
| Few-shot classification | < 500 per class | High | Feature extraction |
| Medical imaging | 1,000–10,000 total | Medium | Fine-tune top layers |
| Large custom dataset | 50,000+ | Low | Fine-tune everything |

## Using Pretrained Models from torchvision

PyTorch's `torchvision.models` provides pretrained weights for dozens of architectures.

```python title="Available pretrained models"
import torchvision.models as models

# Modern API with named weight enums
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# Each weight enum includes the preprocessing transform
preprocess = models.ResNet50_Weights.IMAGENET1K_V2.transforms()
print(preprocess)
# ImageClassification(
#     crop_size=[224],
#     resize_size=[232],
#     mean=[0.485, 0.456, 0.406],
#     std=[0.229, 0.224, 0.225],
#     interpolation=InterpolationMode.BILINEAR,
# )
```

:::tip[Line-by-Line Walkthrough]
- **`models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)`** — Load ResNet-50 with the best available ImageNet weights. The modern API uses named weight enums instead of the old `pretrained=True`.
- **`models.efficientnet_b0(weights=...)`** — EfficientNet-B0: a more efficient architecture that achieves similar accuracy with fewer parameters.
- **`models.vit_b_16(weights=...)`** — Vision Transformer (ViT): a transformer-based architecture for images. Preview of Month 3 concepts!
- **`weights.transforms()`** — Get the exact preprocessing pipeline the model was trained with. This ensures your images are resized, cropped, and normalized identically to the training data.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision
```

**Steps:**
1. Save to `pretrained_models.py`
2. Run: `python pretrained_models.py`
3. Model weights will be downloaded automatically (~100–350 MB each on first run).

**Expected output:**
```
ImageClassification(
    crop_size=[224],
    resize_size=[232],
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    interpolation=InterpolationMode.BILINEAR,
)
```

</details>

:::warning[Preprocessing Must Match]
You **must** use the same normalization (mean, std) and image size that the pretrained model was trained with. Using different normalization will corrupt the feature representations. The `weights.transforms()` API gives you the exact preprocessing pipeline.
:::

## Practical: Fine-Tune ResNet on a Custom Dataset

Let's fine-tune ResNet-50 on a small flower classification dataset. We'll walk through the complete pipeline.

### Step 1: Data Preparation

```python title="Data loading and augmentation"
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# Use the pretrained model's expected transforms for validation
weights = models.ResNet50_Weights.IMAGENET1K_V2
val_transform = weights.transforms()

# Add augmentation for training
train_transform = T.Compose([
    T.RandomResizedCrop(224),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Assumes directory structure: data/flowers/{class_name}/*.jpg
full_dataset = ImageFolder("data/flowers", transform=train_transform)
num_classes = len(full_dataset.classes)
print(f"Classes: {full_dataset.classes}")
print(f"Total images: {len(full_dataset)}")

# Split into train/val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

# Override val transform (random_split shares the dataset object, so we handle this manually)
val_set.dataset = ImageFolder("data/flowers", transform=val_transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)
```

:::tip[Line-by-Line Walkthrough]
- **`weights.transforms()`** — Get the official preprocessing pipeline for ResNet-50 (resize to 232, center-crop to 224, normalize with ImageNet stats). Used for validation — no randomness.
- **`T.Compose([T.RandomResizedCrop(224), ...])`** — Training pipeline with data augmentation: random crops, flips, rotations, and color changes. This makes the model see different versions of each image.
- **`ImageFolder("data/flowers", transform=train_transform)`** — Load images from a folder structure where each subfolder is a class (e.g., `data/flowers/rose/`, `data/flowers/daisy/`). Each image automatically gets its class label from the folder name.
- **`random_split(full_dataset, [train_size, val_size])`** — Split the dataset into 80% training and 20% validation.
- **`val_set.dataset = ImageFolder("data/flowers", transform=val_transform)`** — Override the validation set's transform so it uses deterministic preprocessing (no random augmentation).
- **`DataLoader(..., num_workers=4)`** — Load data using 4 parallel workers for speed.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision
```

**Steps:**
1. Create a folder structure `data/flowers/` with subfolders for each flower class containing `.jpg` images.
2. Save the code to `data_prep.py` (add `import torchvision.models as models` at the top).
3. Run: `python data_prep.py`

**Expected output:**
```
Classes: ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
Total images: 3670
```
(Output depends on your actual dataset.)

</details>

### Step 2: Model Setup

```python title="Configure model for transfer learning"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Replace and unfreeze the classifier
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, num_classes),
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
```

:::tip[Line-by-Line Walkthrough]
- **`torch.device("cuda" if torch.cuda.is_available() else "cpu")`** — Use GPU if available, otherwise CPU.
- **`for param in model.parameters(): param.requires_grad = False`** — Freeze the entire backbone.
- **`model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(...))`** — Replace the classifier with a dropout layer (30% dropout to prevent overfitting) followed by a linear layer mapping to your number of classes.
- **`model.fc.parameters()`** — Only pass the new head's parameters to the optimizer — the frozen backbone won't be updated.
- **`CosineAnnealingLR(optimizer, T_max=10)`** — Smoothly decay the learning rate over 10 epochs.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision
```

**Steps:**
1. Combine with the data preparation code above in one file.
2. Run: `python model_setup.py`

**Expected output:** No errors. The model is loaded, frozen, and ready for training.

</details>

### Step 3: Training — Phase 1 (Head Only)

```python title="Train classifier head"
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

# Phase 1: train only the head
print("Phase 1: Training classifier head")
for epoch in range(5):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"Epoch {epoch+1} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")
```

:::tip[Line-by-Line Walkthrough]
- **`def train_epoch(...)`** — A reusable function that trains the model for one full pass over the data.
- **`model.train()`** — Enable training mode (dropout active, batch norm uses batch stats).
- **`optimizer.zero_grad()`** — Clear gradients from the previous step.
- **`loss.backward()`** — Compute gradients. Only the unfrozen head parameters will get gradients.
- **`optimizer.step()`** — Update only the head parameters (the optimizer only has those).
- **`total_loss += loss.item() * images.size(0)`** — Accumulate loss weighted by batch size (for correct averaging at the end).
- **`outputs.argmax(1) == labels`** — Compare predicted classes with true labels to count correct predictions.
- **`@torch.no_grad()`** — Decorator that disables gradient computation for the entire function — faster and uses less memory during evaluation.
- **`model.eval()`** — Disable dropout, use running statistics for batch norm.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision
```

**Steps:**
1. Combine with all previous steps in one file.
2. Run: `python train_phase1.py`

**Expected output:**
```
Phase 1: Training classifier head
Epoch 1 | Train: 0.782 | Val: 0.856
Epoch 2 | Train: 0.891 | Val: 0.901
...
Epoch 5 | Train: 0.934 | Val: 0.923
```
(Exact numbers depend on your dataset.)

</details>

### Step 4: Training — Phase 2 (Fine-Tune)

```python title="Unfreeze and fine-tune deeper layers"
# Unfreeze layer4 and layer3
for name, param in model.named_parameters():
    if "layer4" in name or "layer3" in name:
        param.requires_grad = True

# New optimizer with discriminative learning rates
optimizer = optim.AdamW([
    {"params": model.layer3.parameters(), "lr": 1e-5},
    {"params": model.layer4.parameters(), "lr": 5e-5},
    {"params": model.fc.parameters(), "lr": 1e-4},
], weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

print("\\nPhase 2: Fine-tuning backbone")
best_val_acc = 0
for epoch in range(10):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_flower_model.pt")
    print(f"Epoch {epoch+1} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")

print(f"\\nBest validation accuracy: {best_val_acc:.3f}")
```

:::tip[Line-by-Line Walkthrough]
- **`if "layer4" in name or "layer3" in name: param.requires_grad = True`** — Selectively unfreeze the top two ResNet blocks. Layer3 and layer4 learn higher-level features that may need adjustment for your task.
- **`optim.AdamW([{...}, {...}, {...}])`** — Create a new optimizer with three parameter groups, each at a different learning rate: layer3 at 1e-5 (very gentle), layer4 at 5e-5, and the head at 1e-4.
- **`if val_acc > best_val_acc:`** — Track the best validation accuracy and save the model whenever we beat the previous best.
- **`torch.save(model.state_dict(), "best_flower_model.pt")`** — Save only the model weights (not the architecture), which is more portable and smaller.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision
```

**Steps:**
1. Run after Phase 1 in the same script.
2. The best model will be saved to `best_flower_model.pt`.

**Expected output:**
```
Phase 2: Fine-tuning backbone
Epoch 1 | Train: 0.951 | Val: 0.938
...
Epoch 10 | Train: 0.987 | Val: 0.956

Best validation accuracy: 0.956
```

</details>

:::tip[Expected Results]
With only a few hundred images per class, fine-tuning ResNet-50 typically achieves **90–95%+** accuracy on flower classification. Training from scratch on the same data would likely cap around 50–70%.
:::

### Step 5: Inference

```python title="Running inference on new images"
from PIL import Image

def predict(model, image_path, transform, class_names, device, top_k=3):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_idxs = probs.topk(top_k, dim=1)

    results = []
    for prob, idx in zip(top_probs[0], top_idxs[0]):
        results.append((class_names[idx], prob.item()))
    return results

# predictions = predict(model, "test_rose.jpg", val_transform, full_dataset.classes, device)
# for name, prob in predictions:
#     print(f"  {name}: {prob:.1%}")
```

:::tip[Line-by-Line Walkthrough]
- **`Image.open(image_path).convert("RGB")`** — Open an image file and ensure it's in RGB format (some images may be grayscale or RGBA).
- **`transform(image).unsqueeze(0)`** — Apply the validation transform (resize, crop, normalize) and add a batch dimension (the model expects a batch, even for a single image).
- **`torch.softmax(logits, dim=1)`** — Convert raw scores into probabilities that sum to 1.
- **`probs.topk(top_k, dim=1)`** — Get the top-k highest probabilities and their indices (e.g., top 3 most likely classes).
- **`class_names[idx]`** — Map the class index back to a human-readable name (e.g., "rose", "daisy").
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:**
```bash
pip install torch torchvision pillow
```

**Steps:**
1. Include in the same script as the training code.
2. Uncomment the last 3 lines and provide a real image path.
3. Run: `python inference.py`

**Expected output:**
```
  rose: 89.3%
  tulip: 7.1%
  daisy: 2.4%
```

</details>

## Exercises

:::tip[Feature Extraction vs Fine-Tuning Comparison — beginner]

Compare feature extraction and fine-tuning on the same dataset. Train both for 15 total epochs and plot their validation accuracy curves. When does fine-tuning start to outperform feature extraction? How much does the advantage vary with dataset size?

<details>
<summary>Hints</summary>

1. Use the exact same dataset split for both experiments
2. For feature extraction: freeze all backbone params, train only fc
3. For fine-tuning: after 5 epochs of head-only, unfreeze the entire model with lr=1e-5
4. Track and plot validation accuracy over epochs for both

</details>

:::

:::tip[Cross-Domain Transfer — intermediate]

Test transfer learning on a domain that is **very different** from ImageNet — such as satellite imagery (EuroSAT), medical images, or abstract art. Does the pretrained ImageNet model still help? Compare against a randomly initialized model trained from scratch. Report accuracy differences and training time.

<details>
<summary>Hints</summary>

1. Use torchvision's EuroSAT dataset or download a medical imaging dataset
2. The domain gap is large — satellite/medical images look nothing like ImageNet photos
3. Try both feature extraction and full fine-tuning; compare which works better
4. Consider that ImageNet features may still help with edges and textures

</details>

:::

:::tip[Freeze Depth Analysis — advanced]

Systematically investigate **how many layers to unfreeze**. Using ResNet-50 on your custom dataset, try unfreezing:
1. Only the classifier head
2. `layer4` + head
3. `layer3` + `layer4` + head
4. Everything

Plot validation accuracy vs. number of trainable parameters. Is there a point of diminishing returns? Does the optimal depth depend on dataset size?

<details>
<summary>Hints</summary>

1. ResNet-50 has: conv1, bn1, layer1, layer2, layer3, layer4, fc
2. Try unfreezing from the top: first just fc, then layer4+fc, then layer3+4+fc, etc.
3. Plot val accuracy vs number of unfrozen layers
4. More unfreezing doesn't always help — it depends on dataset size

</details>

:::

## Resources

- **[Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)** _(tutorial)_ by PyTorch — Official PyTorch tutorial walking through feature extraction and fine-tuning with ResNet.

- **[How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792)** _(paper)_ by Yosinski et al. — Foundational paper quantifying which layers transfer well and which are task-specific.

- **[Practical Deep Learning for Coders (fastai)](https://course.fast.ai/)** _(course)_ by Jeremy Howard — Excellent course that teaches deep learning through transfer learning first — the fastai library makes fine-tuning extremely easy.

- **[timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)** _(tool)_ by Ross Wightman — Massive collection of pretrained vision models (700+). Often has better weights than torchvision.
