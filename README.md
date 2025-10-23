# Image-Text Retrieval Challenge

A challenge to translate text embeddings to image embeddings. This project provides a baseline model and evaluation scripts for a Kaggle-style competition focused on retrieving images that match given text descriptions.

## Table of Contents

- [About the Challenge](#about-the-challenge)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Setup](#data-setup)
- [Usage](#usage)
  - [Baseline Model](#baseline-model)
  - [Training a New Model](#training-a-new-model)
  - [Evaluation](#evaluation)
  - [Generating Submission File](#generating-submission-file)
- [Project Structure](#project-structure)
- [Models](#models)
- [Contributing](#contributing)

## About the Challenge

The core task of this challenge is to build a model that can translate text embeddings into image embeddings. Given a set of text captions, the goal is to find the corresponding images from a large collection. The performance is measured by retrieval metrics such as Recall@k and Mean Reciprocal Rank (MRR).

This repository contains everything you need to get started, including training data, a baseline model, and evaluation scripts.

## Dataset

The dataset is located in the `data/` directory and is structured as follows:

```
data/
├── train/
│   ├── train_data.pt      # Training image and caption embeddings
│   └── captions.txt       # Text captions for training data
└── test/
    ├── test_data_incomplete.pt # Test caption embeddings
    └── captions.txt            # Text captions for the test set
```

- **`data/train/`**: Contains the image and caption embeddings for training and validation.
- **`data/test/`**: Contains the caption embeddings for the test set. The corresponding image embeddings are kept secret for evaluation on the Kaggle platform.

## Getting Started

Follow these steps to set up your local environment.

### Prerequisites

- Python 3.11+
- Pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd image-text-retrieval-challenge
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Data Setup

The training and test data required for the challenge should be placed in the `data/` directory as described in the [Dataset](#dataset) section. If you need to download the data, please refer to the competition's data download instructions.

## Usage

This section explains how to use the provided notebooks and scripts to train, evaluate, and generate predictions.

### Baseline Model

The `baseline.ipynb` notebook provides a complete walkthrough of the process:
1.  Loading the training and test data.
2.  Defining and training a simple MLP baseline model.
3.  Evaluating the model's performance using retrieval metrics.
4.  Visualizing the results.
5.  Generating a submission file for Kaggle.

To get started quickly, open and run the cells in `baseline.ipynb`.

### Training a New Model

You can define your own model architecture in the `src/` directory and use the `pipeline.py` script (or adapt the `baseline.ipynb` notebook) to train it. The main components for training are:
- `src/preprocess_data.py`: For any data preprocessing needs.
- `src/common/config.py`: To manage configurations and hyperparameters.

### Evaluation

The evaluation script `src/eval/eval.py` is used to compute the retrieval metrics. You can use it to evaluate your model's performance on the validation set. The key metrics are:
- **Recall@k (R@k)**: The fraction of queries for which the correct image is ranked within the top-k results.
- **Mean Reciprocal Rank (MRR)**: The average of the reciprocal ranks of the correct images.

The `kaggle_evaluation.py` script is provided to replicate the evaluation logic used in the Kaggle competition.

### Generating Submission File

After training your model, you can generate a `submission.pt` file containing the predicted image embeddings for the test captions. The `baseline.ipynb` notebook includes a section that demonstrates how to create this file.

## Project Structure

```
├── data/                 # Train and test sets
├── models/               # Saved model checkpoints
├── src/                  # Source code
│   ├── common/           # Config, utils
│   ├── eval/             # Evaluation and metrics
│   └── preprocess_data.py # Data preprocessing scripts
├── baseline.ipynb        # Main notebook with baseline model
├── baseline.py           # Python script version of the baseline
├── kaggle_evaluation.py  # Kaggle evaluation script
├── requirements.txt      # Project dependencies
└── README.md             # This file
```
