# AP

## Group Constitution

- João Coelho - PG55954 - [JoaoCoelho2003](https://github.com/JoaoCoelho2003)
- João Faria - PG55953 - [JoaoGDFaria](https://github.com/JoaoGDFaria)
- Jorge Teixeira - PG55965 - [JorgeTeixeira20](https://github.com/JorgeTeixeira20) 
- Rafael Alves - PG55999 - [19Rafa21](https://github.com/19Rafa21)


## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [Datasets](#datasets)
- [Implemented Models](#implemented-models)
- [Future Plans](#future-plans)

## Introduction
AP is a deep learning project for the course, aiming to develop AI models that can distinguish between AI-generated and human-written text. The project includes a frontend for interaction and a backend handling the model inference.

## Project Structure
The repository is organized inside `src` folder:
- **`datasets/`** - Datasets made available by the teachers to evaluate our models.
    - **`clean/`** - Processed datasets put together.
- **`models/`** - Base code for our models divided in two folders:
    - **`numpyModels/`** - Models written using the numpy library (Logistic Regression, DNN and RNN)
    - **`tensorflowModels/`** - Models written using the tensorflow library (Transformer, RNN [LSTM & GRU] and DNN)
    - **`tensorflowModels/`** - Models written in Jupyter Notebooks using the tensorflow library (CNN, RNN [LSTM & GRU], DNN and BERT)
- **`preprocessed/`** - Processed data, ready to be used for numpy models.
- **`preprocessed_tf/`** - Processed data, ready to be used for tensorflow models.
- **`stats/`** - Used to compare model results with test dataset, both for tensorflow and numpy models.
- **`Submissao1/`** - Documents (notebook and CSVs) for the first evaluation phase of this work.
- **`Submissao2/`** - Documents (notebook and CSVs) for the second evaluation phase of this work.
- **`Submissao3/`** - Documents (notebook and CSVs) for the third evaluation phase of this work.
- **`trained_models/`** - Pre-trained models, ready to be used (numpy and tensorflow).
- **`clean_dataset.py`** - Used to remove duplicated lines in a dataset.
- **`create_datasets.py`** - Used to create datasets, based on AI and human texts.
- **`predict_tf.py`** - Loads trained model and evaluates new text inputs to predict whether they are AI-generated or human-written. It supports LSTM, GRU, Transformer and Ensemble tensorflow models and provides the predict.
- **`predict.py`** - Loads a trained model and evaluates new text inputs to predict whether they are AI-generated or human-written. It supports Logistic Regression, DNN, and RNN models and provides a command-line interface for user input.
- **`preprocessing_tf.py`** - Preprocesses the dataset by cleaning and tokenizing text, balancing the dataset, extracting additional features, and training a Word2Vec model. It saves the processed data and models for later use in training and prediction.
- **`preprocessing.py`** - Preprocesses the dataset by cleaning and tokenizing text, balancing the dataset, vectorizing text using TF-IDF, and training a Word2Vec model. It saves the processed data and models for later use in training and prediction. 
- **`train_tf.py`** - Handles the training of different tensorflow models (LSTN, GRU, Transformer, DNN, Ensemble). It loads preprocessed data, initializes all the models, trains tehm, and saves the trained model weights.
- **`train.py`** - Handles the training of different models (Logistic Regression, DNN, RNN) based on the command-line argument provided. It loads preprocessed data, initializes the appropriate model, trains it, and saves the trained model weights.
- **`PresentationAndReport/`** - Contains the presentation used in the project defense and the project report.
    - **`presentation.pdf`** - Presentation used in the project defense.
    - **`report.pdf`** - Project report.

## How to use

Depending on the phase of the project, you can run the models in different ways.

### First Phase

1. First run the `preprocessing.py` script to preprocess the datasets:

```sh
$ python3 preprocessing.py
```

2. Train the models using the `train.py` script and specify the model you want to train.

```sh
$ python3 train.py <model_name>
```

Replace `<model_name>` with the desired model: `logistic`, `dnn`, or `rnn`.

3. After training, you can use the `predict.py` script to evaluate the models. You can specify the model you want to use for prediction:

```sh
$ python3 predict.py <model_name>
```

Replace `<model_name>` with the desired model: `logistic`, `dnn`, or `rnn`.

4. To test the models you can use the **Jupyter Notebook** provided in the `stats/` folder, more specifically the `evaluate_phase1.ipynb` file. This notebook contains code to evaluate the models using the test dataset and compare their performance.

### Second Phase

1. Follow the same steps as in the first phase to preprocess the datasets and train the models but this time use the `preprocessing_tf.py` and `train_tf.py` scripts for the TensorFlow models (in this phase the `train_tf` automatically trains all the models available).

2. After training, you can use the `predict_tf.py` script to evaluate the models. You can specify the model you want to use for prediction:

```sh
$ python3 predict_tf.py <model_name>
```

Replace `<model_name>` with the desired model: `lstm`, `gru`, `transformer`, `dnn`, or `ensemble`.

3. To test the models you can use the **Jupyter Notebook** provided in the `stats/` folder, more specifically the `evaluate_phase2.ipynb` file. This notebook contains code to evaluate the models using the test dataset and compare their performance.

### Third Phase

1. Finally, to run the models, you no longer need any python scripts, simply run the **Jupyter Notebooks** present at `models/notebooks/` folder. The notebooks are organized by model type and contain all the necessary code to train and evaluate the models.

2. If you want to evaluate even further the models, you can use the **Jupyter Notebook** provided in the `stats/` folder, more specifically the `evaluate_phase3.ipynb` file. This notebook contains code to evaluate the models using the test dataset and compare their performance.

## Datasets
The project uses two datasets:

1. **Hugging Face Dataset**: [`dmitva/human_ai_generated_text`](https://huggingface.co/datasets/dmitva/human_ai_generated_text), which contains human-written and AI-generated text samples. The dataset is loaded as a streaming dataset in the backend:

    ```python
    from datasets import load_dataset

    dataset = load_dataset("dmitva/human_ai_generated_text", split="train", streaming=True)
    ```

2. **Validation Dataset**: [`andythetechnerd03/AI-human-text`](https://huggingface.co/datasets/andythetechnerd03/AI-human-text), used for validation purposes. This dataset is also loaded as a streaming dataset:

    ```python
    from datasets import load_dataset

    validation_dataset = load_dataset("andythetechnerd03/AI-human-text", split="train", streaming=True)
    ```

3. **Custom Dataset**: The project also includes a custom dataset created by the team, which contains human-written texts that were scraped from wikipedia and AI-generated texts created using the `gemini` model. This dataset is used for training and testing the models. The dataset is located in the `datasets/` folder, more specifically `custom_dataset.csv`.