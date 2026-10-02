# Trash / Recycling Classifier — Project Plan

## 1. Project Overview

The goal of this project is to build an image-based waste classification system that helps users determine how an item should be disposed of according to the published waste-sorting rules of **one specific municipality**.

The system will take an image of a waste item, classify it into a disposal category, and provide the appropriate disposal guidance. Rather than simply identifying what an object is, the system will answer:

> **"Given this item and the rules of this city, where should it go?"**

The initial target municipality will be **San Diego**. The system will use publicly available datasets combined with personally collected/labeled images.

The initial system will classify one object at a time. Multi-object detection will be treated as a stretch feature.

---

# 2. Project Objectives

## Primary Objectives

1. Research and define disposal categories using official municipal rules.
2. Build a labeled image dataset appropriate for those categories.
3. Combine existing waste datasets with personally collected images.
4. Address class imbalance and differences between clean and real-world images.
5. Train an image classification model using transfer learning.
6. Evaluate the model using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
7. Add a confidence threshold and an **"I'm not sure"** prediction state.
8. Deploy the model to a Raspberry Pi 5.
9. Connect a camera to the Raspberry Pi.
10. Demonstrate real-time classification and disposal guidance.

## Stretch Objectives

- Detect multiple objects in a single image.
- Add object detection using YOLO.
- Publish the personally collected dataset.
- Add more difficult/ambiguous waste examples.
- Improve inference speed through quantization.
- Create a more polished user interface.

---

# 3. Recommended Initial Scope

The first major decision is the **label set**.

Do not begin by downloading thousands of images. First determine exactly what the model needs to predict.

## Proposed Initial Classes

| Class | Meaning |
|---|---|
| Recycling | Item belongs in the recycling stream |
| Compost | Item belongs in the organic/compost stream |
| Trash | Item belongs in the landfill/trash stream |
| Not Sure | Model confidence is below the required threshold |

The **Not Sure** category should not necessarily be a trained image class. Instead, it can be produced when:

```text
model confidence < threshold
```

Example:

```text
Prediction: Recycling
Confidence: 91%

→ Recycle
```

versus:

```text
Prediction: Compost
Confidence: 48%

→ I'm not sure. Please check the local disposal guide.
```

The exact categories should be finalized after researching the official San Diego rules. Some items may require special handling and could require an additional category or special-rules lookup.

---

# 4. Step 1 — Research San Diego Disposal Rules

## Goal

Create the authoritative rule set that the machine-learning system will follow.

The model should not invent disposal rules. The project should use published information from relevant San Diego municipal/waste-management sources.

## Research

Determine:

- What belongs in recycling?
- What belongs in trash?
- What belongs in organics/compost?
- What items have special disposal requirements?
- What items are commonly confused?
- Are there differences between residential and commercial collection?
- Are there items that look recyclable but are not accepted?

## Create a Rules Table

| Item | Category | Notes | Source |
|---|---|---|---|
| Clean cardboard | Recycling | Verify local acceptance | Official source |
| Food scraps | Compost | Organic material | Official source |
| Greasy pizza box | Compost/Other | Verify local rule | Official source |
| Plastic bottle | Recycling | Verify accepted material | Official source |
| Plastic bag | Special/Trash | Verify local rule | Official source |

The **Source** column is important because this project specifically uses real municipal rules.

## Deliverable

**Deliverable 1: San Diego Waste Classification Rules**

A Markdown/PDF document containing:

- Disposal categories
- Accepted items
- Non-accepted items
- Ambiguous items
- Special cases
- Source links
- Date accessed

---

# 5. Step 2 — Define the Dataset Classes

Once the rules are understood, determine which physical objects the model will recognize.

Do not try to recognize every possible piece of trash.

Start with a manageable subset.

## Example Classes

### Recycling

- Plastic bottles
- Aluminum cans
- Cardboard
- Paper
- Glass containers

### Compost

- Food scraps
- Fruit/vegetable waste
- Leaves
- Compostable food-related items where officially accepted

### Trash

- Certain contaminated materials
- Non-recyclable packaging
- Other landfill materials

The exact classes should be determined by the municipal research.

## Important Distinction

The model's **visual classes** and **disposal categories** do not necessarily have to be identical.

For example:

```text
Image
  ↓
Plastic bottle
  ↓
San Diego disposal rule
  ↓
Recycling
```

This architecture can be more useful than training the model to directly predict only:

```text
Recycling / Compost / Trash
```

However, for the first MVP, directly predicting disposal categories is simpler.

---

# 6. Step 3 — Find Existing Datasets

Start with existing datasets rather than collecting everything yourself.

## Dataset Sources

Investigate:

- TrashNet
- TACO
- Kaggle waste datasets
- Roboflow Universe datasets
- Other publicly available waste-image datasets

## Dataset Requirements

For every dataset, record:

| Dataset | Images | Classes | Image Type | License | Useful? |
|---|---:|---|---|---|---|
| TrashNet | ~2,500 | Multiple | Mostly clean | Verify | Yes |
| TACO | Smaller | Multiple | Real-world | Verify | Yes |
| Kaggle Dataset | TBD | TBD | TBD | Verify | TBD |

Do not automatically merge datasets.

First determine:

1. What classes exist?
2. Are the labels compatible?
3. Are there duplicates?
4. Are the images studio-quality?
5. Are there enough examples?
6. What licenses apply?

---

# 7. Step 4 — Build a Dataset Pipeline

Create a consistent dataset structure.

```text
trash-classifier/
│
├── data/
│   ├── raw/
│   │   ├── trashnet/
│   │   ├── taco/
│   │   └── kaggle/
│   │
│   ├── processed/
│   │
│   ├── train/
│   ├── validation/
│   └── test/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   └── inference/
│
├── models/
│
├── raspberry_pi/
│
├── results/
│
├── docs/
│
├── requirements.txt
└── README.md
```

---

# 8. Step 5 — Clean and Standardize the Dataset

Before training, inspect the data.

Look for:

- Corrupted images
- Duplicate images
- Incorrect labels
- Extremely low-resolution images
- Images that don't contain the labeled object
- Different naming conventions
- Severe class imbalance

Create a preprocessing script rather than manually doing everything.

## Example Pipeline

```text
Raw datasets
      ↓
Remove corrupted images
      ↓
Remove duplicates
      ↓
Standardize labels
      ↓
Resize images
      ↓
Train / Validation / Test split
      ↓
Data augmentation
      ↓
Training
```

---

# 9. Step 6 — Add Personally Collected Images

One of the most valuable parts of the project is personally collected data.

Existing datasets often contain clean, centered images. Real users will take pictures in much less controlled environments.

## Collect Examples With

- Different backgrounds
- Different lighting
- Shadows
- Different camera angles
- Objects partially obscured
- Crushed containers
- Dirty containers
- Wrinkled cardboard
- Different object sizes
- Multiple brands
- Different colors
- Indoor/outdoor environments

## Suggested Initial Target

Aim for approximately **50–100 personally collected images per important class** initially.

If time allows, increase this substantially for difficult classes.

## Deliverable

**Deliverable 2: Personally Collected Dataset**

Include:

- Images
- Labels
- Collection methodology
- Class counts
- Annotation guidelines
- Dataset license/usage notes if publishing

---

# 10. Step 7 — Analyze Class Imbalance

Create a graph showing the number of images per class.

Example:

```text
Recycling   ████████████████████ 1200
Trash       █████████████████     1000
Compost     ███████                400
```

If one class is significantly smaller, investigate:

- Collecting additional images
- Augmentation
- Class weights
- Oversampling

Do not simply duplicate the same image hundreds of times.

---

# 11. Step 8 — Establish a Baseline Model

Start simple.

The first model should be a transfer-learning image classifier.

## Recommended Models

Start with:

- **MobileNetV3**
- **EfficientNet-B0**

These are good candidates because the final goal includes edge deployment.

## Initial Experiment

```text
Input image
     ↓
Pretrained CNN
     ↓
Classification head
     ↓
Recycling / Compost / Trash
```

Do not begin by building a CNN from scratch.

Transfer learning allows the project to focus on waste classification rather than spending the entire project training a basic vision model.

---

# 12. Step 9 — Data Augmentation

Use augmentation to simulate real-world conditions.

Potential augmentations:

- Rotation
- Horizontal flip
- Random crop
- Brightness changes
- Contrast changes
- Blur
- Noise
- Perspective changes
- Background variation

Albumentations can be used for this.

The goal is to make the training distribution more representative of how users will actually photograph waste.

---

# 13. Step 10 — Train the Model

Create a reproducible training script.

The script should specify:

- Dataset location
- Image size
- Batch size
- Learning rate
- Number of epochs
- Optimizer
- Model architecture
- Augmentation
- Random seed
- Output model path

Example:

```text
train.py

Dataset
   ↓
Preprocessing
   ↓
Augmentation
   ↓
MobileNetV3
   ↓
Training
   ↓
Best model
   ↓
models/best_model.pth
```

Save training results.

---

# 14. Step 11 — Evaluate the Model

Do not rely only on accuracy.

## Metrics

Calculate:

- Accuracy
- Precision
- Recall
- F1-score

## Confusion Matrix

Example:

```text
                 Predicted
              R     C     T
Actual R     90     5     5
       C      7    85     8
       T      4     6    90
```

This helps identify which categories the model confuses.

If compost and trash are frequently confused, investigate why.

---

# 15. Step 12 — Test With Real-World Images

Create a separate real-world test set.

Do **not** simply use augmented training images as proof that the system works in the real world.

Include:

- Messy backgrounds
- Different lighting
- Dirty objects
- Unusual orientations
- Objects partially obscured
- Images taken with the Raspberry Pi camera

Compare:

```text
Clean Dataset Performance
             vs.
Real-World Dataset Performance
```

This comparison can become one of the strongest parts of the final project.

---

# 16. Step 13 — Add the "I'm Not Sure" System

A classifier should not be forced to make a confident prediction when it is uncertain.

Example:

```text
Model prediction:
Trash

Confidence:
0.47
```

If the threshold is:

```text
0.70
```

then:

```text
"I'm not sure. Please check the local disposal guide."
```

## Experiment With Different Thresholds

Test thresholds such as:

```text
0.50
0.60
0.70
0.80
0.90
```

Measure how the threshold changes:

- Correct predictions
- Incorrect predictions
- Abstentions

Choose a threshold based on evaluation rather than arbitrarily.

---

# 17. Step 14 — Connect Classification to Local Rules

The final application should not simply say:

> Recycling

It should provide useful guidance.

Example:

```text
Detected:
Plastic bottle

Disposal:
Recycling

Confidence:
93%

San Diego guidance:
[Local rule/guidance]
```

For an uncertain prediction:

```text
Detected:
Unknown food container

Confidence:
52%

Result:
I'm not sure.

Please check the San Diego disposal guide.
```

This separates:

### Computer Vision

> "What is this?"

from:

### Rule System

> "Where should this go?"

This separation makes the project easier to maintain if municipal rules change.

---

# 18. Step 15 — Raspberry Pi Deployment

Once the model works on the computer, move it to the Raspberry Pi.

## Recommended Hardware

- Raspberry Pi 5
- Camera
- Monitor/display
- Appropriate power supply
- Optional table-mounted camera

## Deployment Pipeline

```text
Camera
   ↓
Capture image
   ↓
Resize/preprocess
   ↓
Model inference
   ↓
Prediction
   ↓
Confidence threshold
   ↓
Local disposal rule
   ↓
Display result
```

---

# 19. Step 16 — Optimize the Model

Measure:

- Model size
- Inference time
- RAM usage
- CPU usage
- Frames/images per second

If necessary, investigate:

- Quantization
- Smaller model architecture
- ONNX
- TensorFlow Lite
- Other edge inference formats

The goal is a reasonable balance between:

```text
Accuracy
+
Speed
+
Model Size
```

---

# 20. Step 17 — Build the Final Demo

The final demo should be simple.

A user places an item under the camera.

```text
        CAMERA
           ↓
    ┌─────────────┐
    │    ITEM     │
    └─────────────┘
           ↓

    ┌──────────────────┐
    │     RECYCLING    │
    │                  │
    │ Confidence: 94%  │
    └──────────────────┘
```

The system should display:

1. Detected category
2. Confidence
3. Disposal recommendation
4. Local rule/guidance
5. "I'm not sure" when confidence is too low

---

# 21. Stretch Feature — Multi-Object Detection

Only begin this after the single-object classifier works.

Example:

```text
 ┌─────────────────────────┐
 │  🍎       🥤       📦   │
 │                         │
 └─────────────────────────┘

 Apple       → Compost
 Bottle      → Recycling
 Cardboard   → Recycling
```

This requires moving from pure classification toward object detection.

Possible model:

**YOLO11**

Pipeline:

```text
Image
  ↓
YOLO
  ↓
Object 1 ──→ classification/rule
Object 2 ──→ classification/rule
Object 3 ──→ classification/rule
```

This is explicitly a **stretch goal** and should not delay the core project.

---

# 22. Major Deliverables

## Deliverable 1 — Requirements & Rules Document

Contains:

- Problem statement
- Project objectives
- Scope
- San Diego disposal rules
- Initial classification categories
- Special cases
- Sources

## Deliverable 2 — Dataset

Contains:

- Existing datasets
- Personally collected images
- Cleaned/processed images
- Labels
- Train/validation/test splits
- Dataset statistics

## Deliverable 3 — Data Processing Pipeline

Code for:

- Cleaning
- Resizing
- Splitting
- Augmentation
- Class balancing

## Deliverable 4 — Trained Model

Contains:

- Training code
- Saved model
- Training configuration
- Training metrics
- Model version

## Deliverable 5 — Evaluation Report

Contains:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class results
- Real-world test results
- Error analysis

## Deliverable 6 — Uncertainty/Abstention System

Demonstrates:

```text
High confidence → prediction

Low confidence → "I'm not sure"
```

Include threshold experiments.

## Deliverable 7 — Raspberry Pi Prototype

Working system containing:

- Raspberry Pi 5
- Camera
- Model
- Inference pipeline
- Display/output
- Local disposal guidance

## Deliverable 8 — Final Demonstration

Demonstrate:

1. Easy clean item
2. Normal real-world item
3. Difficult item
4. Ambiguous item
5. Low-confidence example
6. Multiple objects if stretch feature is completed

## Deliverable 9 — Final Documentation

README should contain:

```text
Project Overview
Installation
Dataset
Dataset Preparation
Training
Evaluation
Model
Raspberry Pi Deployment
Usage
Local Waste Rules
Results
Limitations
Future Work
```

---

# 23. Planned Schedule

**Assumption: project begins October 2 and final deliverables are due at the beginning of December.**

There are approximately **9 weeks**, so the schedule should prioritize getting a functioning MVP early.

---

## Week 1 — October 2–8
### Research + Project Setup

### Goals

- Finalize project scope
- Research San Diego waste rules
- Determine initial categories
- Identify datasets
- Create GitHub repository
- Create project structure
- Set up Python environment

### Deliverables

- Project proposal
- Initial requirements
- San Diego rules document
- Dataset inventory
- Repository structure

### Milestone

> **Know exactly what the model is supposed to classify.**

---

## Week 2 — October 9–15
### Dataset Collection

### Goals

- Download selected datasets
- Examine class distributions
- Determine compatible labels
- Begin personally collecting images
- Establish annotation guidelines

### Deliverables

- Raw datasets
- Initial personal dataset
- Dataset statistics
- Label mapping

### Milestone

> **Have enough data to begin training.**

---

## Week 3 — October 16–22
### Dataset Processing + Baseline

### Goals

- Clean data
- Remove corrupted/duplicate images
- Split train/validation/test
- Implement augmentation
- Train first MobileNetV3/EfficientNet model

### Deliverables

- Data preprocessing pipeline
- Baseline model
- Initial training results

### Milestone

> **First working classifier.**

It does not need to be highly accurate yet.

---

## Week 4 — October 23–29
### Model Improvement

### Goals

- Analyze baseline errors
- Address class imbalance
- Increase personal dataset
- Experiment with augmentation
- Tune hyperparameters
- Compare models if time permits

Possible comparison:

```text
MobileNetV3
vs.
EfficientNet-B0
```

### Deliverables

- Improved model
- Training graphs
- Model comparison
- Updated dataset

### Milestone

> **Reliable desktop classifier.**

---

## Week 5 — October 30–November 5
### Evaluation + Real-World Testing

### Goals

- Create real-world test set
- Generate confusion matrix
- Calculate precision/recall/F1
- Identify difficult classes
- Analyze incorrect predictions

### Deliverables

- Confusion matrix
- Evaluation report
- Error analysis
- Real-world test results

### Milestone

> **Understand where the model succeeds and fails.**

---

## Week 6 — November 6–12
### Confidence + Local Rules

### Goals

- Implement confidence threshold
- Test multiple thresholds
- Add "I'm not sure"
- Connect predictions to San Diego rules
- Create user-facing output

### Deliverables

Example:

```text
Prediction: Recycling
Confidence: 91%

San Diego Guidance:
[rule]
```

and:

```text
Confidence: 43%

I'm not sure.
Please check the local disposal guide.
```

### Milestone

> **Complete functional software MVP.**

---

## Week 7 — November 13–19
### Raspberry Pi Deployment

### Goals

- Set up Raspberry Pi 5
- Connect camera
- Install dependencies
- Transfer model
- Run inference
- Measure inference speed
- Optimize model if necessary

### Deliverables

- Raspberry Pi prototype
- Camera pipeline
- Performance measurements

### Milestone

> **Model runs on Raspberry Pi.**

---

## Week 8 — November 20–26
### Integration + Testing

### Goals

- Integrate camera + model + rules
- Test different objects
- Test different lighting
- Test dirty/ambiguous objects
- Measure latency
- Fix bugs
- Improve UI

### Deliverables

- Complete prototype
- Test results
- Final model
- Final dataset

### Milestone

> **End-to-end system works reliably enough for demonstration.**

---

## Week 9 — November 27–December 3
### Finalization

### Goals

- Final testing
- Documentation
- README
- Final graphs
- Final screenshots
- Demo preparation
- Presentation
- Optional stretch feature only if core system is complete

### Deliverables

- Final GitHub repository
- Final report
- Final presentation
- Working Raspberry Pi demo
- Final evaluation
- Demo video if required

### Milestone

> **Project complete.**

---

# 24. Suggested Weekly Priority

## Tier 1 — Must Have

```text
San Diego rules
       ↓
Dataset
       ↓
Image classifier
       ↓
Evaluation
       ↓
Confidence threshold
       ↓
Raspberry Pi deployment
```

## Tier 2 — Strong Improvements

```text
Personal dataset
       ↓
Real-world testing
       ↓
Class imbalance handling
       ↓
Model optimization
```

## Tier 3 — Stretch

```text
Multi-object detection
       ↓
YOLO
       ↓
Dataset publication
       ↓
Advanced UI
```

**Do not start Tier 3 until Tier 1 is working.**

---

# 25. Final System Architecture

```text
                         ┌──────────────────┐
                         │   Camera/Image   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Image Processing │
                         │ Resize/Normalize │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ CNN Classifier   │
                         │ MobileNetV3      │
                         └────────┬─────────┘
                                  │
                         ┌────────▼────────┐
                         │   Confidence    │
                         │    Threshold    │
                         └───────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
             High confidence            Low confidence
                    │                         │
                    ▼                         ▼
             ┌─────────────┐          ┌──────────────┐
             │ Prediction  │          │  "Not Sure"  │
             └──────┬──────┘          └──────────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │ San Diego Rule Set  │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ Disposal Guidance   │
          └──────────┬──────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ User Display        │
          │ Category + Guidance │
          └─────────────────────┘
```

---

# 26. Final Success Criteria

## Machine Learning

- [ ] Dataset contains multiple waste categories
- [ ] Dataset includes real-world images
- [ ] Personally collected images are included
- [ ] Class imbalance has been analyzed
- [ ] Transfer-learning model has been trained
- [ ] Test set is separate from training data
- [ ] Confusion matrix is produced
- [ ] Precision/recall/F1 are reported
- [ ] Model errors are analyzed

## Uncertainty

- [ ] Confidence score is displayed
- [ ] Confidence threshold is implemented
- [ ] Low-confidence predictions produce "I'm not sure"
- [ ] Threshold is experimentally evaluated

## Local Rules

- [ ] San Diego rules are documented
- [ ] Sources are recorded
- [ ] Predictions are connected to disposal guidance
- [ ] Ambiguous/special cases are documented

## Deployment

- [ ] Model runs on Raspberry Pi 5
- [ ] Camera captures images
- [ ] Images are classified on-device
- [ ] Inference time is measured
- [ ] Final system provides user-facing guidance

## Documentation

- [ ] README completed
- [ ] Dataset documented
- [ ] Training process documented
- [ ] Evaluation documented
- [ ] Raspberry Pi setup documented
- [ ] Limitations documented
- [ ] Future work documented

---

# 27. Where to Start Today

**Do not start training a model yet.**

The first work session should focus on the San Diego rules and dataset inventory.

## Task 1 — Define the San Diego Rule Set

Create:

```text
docs/
└── san-diego-waste-rules.md
```

Start a table containing:

```text
Item
Visual Class
Disposal Category
Special Instructions
Source
```

Research approximately **15–25 common waste items** and determine how San Diego says they should be handled.

---

## Task 2 — Create the GitHub Repository

Set up:

```text
trash-classifier/
│
├── data/
├── docs/
├── notebooks/
├── src/
├── models/
├── results/
├── raspberry_pi/
└── README.md
```

---

## Task 3 — Make a Dataset Inventory

Create a spreadsheet with:

```text
Dataset
URL
License
Number of Images
Classes
Image Quality
Real-world?
Potentially Useful Classes
```

---

## Task 4 — Choose the First Dataset

Do not merge five datasets immediately.

Start with one dataset and determine:

> "Can I map its labels to my San Diego classification scheme?"

Then add additional datasets only where they fill gaps.

---

## Task 5 — Establish Your Baseline

Once the dataset is ready:

```text
Dataset
   ↓
Train/Validation/Test Split
   ↓
MobileNetV3
   ↓
Training
   ↓
Accuracy + Confusion Matrix
```

This gives you a measurable starting point.

---

# 28. Recommended Definition of the MVP

By approximately **November 12**, the project should ideally have:

> **A computer-based application that takes a single image of a waste item, classifies it into a San Diego disposal category, provides a confidence score, and returns "I'm not sure" when confidence is below a selected threshold.**

Then:

- **November 13–26:** Move the system to Raspberry Pi.
- **November 27–December 3:** Testing, documentation, presentation, and optional stretch work.

This prevents the Raspberry Pi or YOLO portion from becoming a dependency that causes the entire project to fail.

---

# 29. Potential Final Project Story

The project demonstrates a complete machine-learning pipeline:

```text
Real-world problem
       ↓
Municipal rule research
       ↓
Dataset curation
       ↓
Personal data collection
       ↓
Data preprocessing
       ↓
Transfer learning
       ↓
Model evaluation
       ↓
Error analysis
       ↓
Uncertainty handling
       ↓
Rule-based guidance
       ↓
Edge deployment
       ↓
Real-world testing
```

This makes the project more substantial than simply training a trash classifier because it addresses the full path from:

**real-world data → ML model → uncertainty → domain-specific rules → edge deployment**
