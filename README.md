# AP

## Group Constitution

- João Coelho - PG55954 - [JoaoCoelho2003](https://github.com/JoaoCoelho2003)
- João Faria - PG55953 - [JoaoGDFaria](https://github.com/JoaoGDFaria)
- Jorge Teixeira - PG55965 - [JorgeTeixeira20](https://github.com/JorgeTeixeira20) 
- Rafael Alves - PG55999 - [19Rafa21](https://github.com/19Rafa21)


## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Implemented Models](#implemented-models)
- [Future Plans](#future-plans)

## Introduction
AP is a deep learning project for the course, aiming to develop AI models that can distinguish between AI-generated and human-written text. The project includes a frontend for interaction and a backend handling the model inference.

## Project Structure
The repository is organized into two main folders:
- **`frontend/`** – Built using Vue 3 with JavaScript.
- **`backend/`** – Developed in Python, utilizing libraries like NumPy, scikit-learn, and other auxiliary tools.

## Technologies Used
### Frontend
- Vue 3
- JavaScript

### Backend
- Python
- NumPy
- Scikit-learn
- Hugging Face `datasets`

## Dataset
The project uses a dataset from Hugging Face: [`dmitva/human_ai_generated_text`](https://huggingface.co/datasets/dmitva/human_ai_generated_text), which contains human-written and AI-generated text samples.  

The dataset is loaded as a streaming dataset in the backend:

```python
from datasets import load_dataset

dataset = load_dataset("dmitva/human_ai_generated_text", split="train", streaming=True)
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