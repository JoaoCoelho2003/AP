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

## How to use

You can either run the frontend and backend together (full application) or run each model separately.

### Full Application

1. Ensure that the trained models are available in the `trained_models/` directory.
2. Navigate to the `frontend/` folder and run:
```sh
$ npm install  
$ npm run dev  
``` 
3. Navigate to the `backend/` folder and run:
```sh
$ python3 app.py
```
4. Access the frontend at `http://localhost:5173/` and interact with the application.

### Running Models Individually

If you prefer to run each model separately (with no frontend), go to the `backend/`directory and run the following commands:

1. Preprocess the datasets:
```sh
$ python3 preprocessing.py
```
2. Train a model:
```sh
$ python3 train.py <model_name>
```
3. Predict using a trained model:
```sh
$ python3 predict.py <model_name>
```

Replace `<model_name>` with the desired model: `logistic`, `dnn`, or `rnn`.

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

## Implemented Models
Currently, the backend supports the following models:
- **RNN** (Recurrent Neural Network)
- **DNN** (Deep Neural Network)
- **Logistic Regression**

## Future Plans
- Expanding the dataset for better performance.
- Adding more models to improve classification accuracy.
- Use libraries like tensorflow instead of building the models by hand using numpy.
