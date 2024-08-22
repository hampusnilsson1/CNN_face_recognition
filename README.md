# AI 2 Course Project - Hampus Nilsson

## Overview

This repository contains code for a project that involves training Convolutional Neural Network (CNN) models for face analysis. Specifically, the project includes:

- **Expression Model**: Trained on one dataset.
- **Age, Ethnicity, and Gender Models**: Trained on a separate dataset.

## Data

The training data for the models (`gender_age.csv`) is not included in the repository due to its large size. You can download the dataset from Kaggle:

- [Download Dataset](https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv/data)

## Application

The project features a Streamlit application that allows users to:

- Detect age, gender, and/or expression from webcam input.
- Analyze model performance.
- Perform prediction tests.

**Note**: The age model is not used in the `Home.py` webcam prediction functionality.
