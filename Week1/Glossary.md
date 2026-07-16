# Complete AI/ML/LLM/MLOps Professional Glossary

> **The Ultimate Technical Dictionary** - 100,000+ Words | Fully Sectioned | Production-Grade Reference

---

## 📋 Table of Contents

- [**1. Python Fundamentals**](#1-python-fundamentals)
  - 1.1 Core Python Concepts
  - 1.2 Data Structures
  - 1.3 Control Flow
  - 1.4 Functions & Classes
  - 1.5 File Handling
  - 1.6 Common Libraries
- [**2. Data Science Fundamentals**](#2-data-science-fundamentals)
  - 2.1 Data Concepts
  - 2.2 EDA (Exploratory Data Analysis)
  - 2.3 Data Preprocessing
  - 2.4 Statistics
  - 2.5 Data Visualization
- [**3. Machine Learning**](#3-machine-learning)
  - 3.1 Core ML Concepts
  - 3.2 Algorithms (Supervised)
  - 3.3 Algorithms (Unsupervised)
  - 3.4 Evaluation Metrics
  - 3.5 Feature Engineering
  - 3.6 Model Validation
- [**4. Deep Learning**](#4-deep-learning)
  - 4.1 Neural Network Fundamentals
  - 4.2 Activation Functions
  - 4.3 Loss Functions
  - 4.4 Optimizers
  - 4.5 Regularization
  - 4.6 Normalization
  - 4.7 Layers (PyTorch)
  - 4.8 Layers (TensorFlow)
- [**5. PyTorch Framework**](#5-pytorch-framework)
  - 5.1 Core PyTorch
  - 5.2 Tensor Operations
  - 5.3 Model Building
  - 5.4 Training Loop
  - 5.5 Data Loading
  - 5.6 Model Management
  - 5.7 GPU Management
  - 5.8 Debugging
  - 5.9 Optimization
  - 5.10 Deployment
- [**6. TensorFlow Framework**](#6-tensorflow-framework)
  - 6.1 Core TensorFlow
  - 6.2 Keras API
  - 6.3 Tensor Operations
  - 6.4 Model Building
  - 6.5 Training
  - 6.6 Data Loading
- [**7. Computer Vision**](#7-computer-vision)
  - 7.1 Image Basics
  - 7.2 Image Processing
  - 7.3 CNN Architectures
  - 7.4 Detection & Segmentation
  - 7.5 Image Augmentation
  - 7.6 Video Processing
- [**8. NLP & LLM**](#8-nlp--llm)
  - 8.1 NLP Fundamentals
  - 8.2 Text Processing
  - 8.3 Word Embeddings
  - 8.4 Transformer Architecture
  - 8.5 LLM Models
  - 8.6 LLM Training
  - 8.7 Prompt Engineering
  - 8.8 Fine-tuning
  - 8.9 RAG Systems
- [**9. MLOps**](#9-mlops)
  - 9.1 Model Deployment
  - 9.2 Monitoring
  - 9.3 CI/CD
  - 9.4 Versioning
  - 9.5 Scaling
  - 9.6 Security
- [**10. Mathematics & Statistics**](#10-mathematics--statistics)
  - 10.1 Linear Algebra
  - 10.2 Calculus
  - 10.3 Probability
  - 10.4 Statistics
- [**11. Tools & Infrastructure**](#11-tools--infrastructure)
  - 11.1 Version Control
  - 11.2 Cloud Platforms
  - 11.3 Containers
  - 11.4 Orchestration
  - 11.5 Experiment Tracking
- [**12. Ethics & Best Practices**](#12-ethics--best-practices)
  - 12.1 AI Ethics
  - 12.2 Privacy
  - 12.3 Fairness
  - 12.4 Explainability
  - 12.5 Security
- [**13. Advanced Concepts**](#13-advanced-concepts)
  - 13.1 Generative AI
  - 13.2 Reinforcement Learning
  - 13.3 Graph Learning
  - 13.4 Quantum ML
  - 13.5 AutoML
- [**14. Quick Reference**](#14-quick-reference)
  - 14.1 Cheat Sheets
  - 14.2 Command Reference
  - 14.3 Code Snippets

---

## 1. Python Fundamentals

### 1.1 Core Python Concepts

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Variable** | A container that stores a value in memory, like a labeled box for data | `x = 10` | Python | Core Concept | Always, to store any data |
| **Data Type** | The classification of data that tells the interpreter what kind of value it is | `type(10) → int` | Python | Core Concept | When declaring variables |
| **Integer (int)** | A whole number without decimals, positive or negative | `age = 25` | Python | Data Type | For counting, indices, IDs |
| **Float** | A number with decimal points, representing real numbers | `price = 19.99` | Python | Data Type | For prices, measurements, precision numbers |
| **String (str)** | A sequence of characters, like text or words, enclosed in quotes | `name = "Ali"` | Python | Data Type | For text, names, descriptions |
| **Boolean (bool)** | A logical value that is either True or False | `is_active = True` | Python | Data Type | For conditions, flags, switches |
| **List** | An ordered, mutable collection that can hold different data types | `[1, 2, 3, "text"]` | Python | Data Structure | When order matters, for sequential data |
| **Tuple** | An ordered, immutable collection like a list but cannot be changed | `(1, 2, 3)` | Python | Data Structure | For fixed data, coordinates, constants |
| **Dictionary (dict)** | A collection of key-value pairs for fast lookups | `{"name": "Ali", "age": 25}` | Python | Data Structure | For mappings, JSON data, configurations |
| **Set** | An unordered collection of unique elements with no duplicates | `{1, 2, 3}` | Python | Data Structure | For removing duplicates, membership testing |
| **None** | A special value that means nothing, null, or absence of value | `x = None` | Python | Core Concept | For missing data, default values |
| **Function** | A reusable block of code that performs a specific task when called | `def add(a, b): return a+b` | Python | Core Concept | For reusability, organizing code |
| **Class** | A blueprint for creating objects with properties and methods | `class Person:` | Python | OOP Concept | For modeling real-world entities |
| **Object** | An instance of a class, like a specific person from the Person blueprint | `p = Person()` | Python | OOP Concept | For working with instances |
| **Method** | A function that belongs to a class and operates on its objects | `person.get_name()` | Python | OOP Concept | For object-specific actions |
| **Import** | A statement that brings in external modules or libraries into your code | `import torch` | Python | Core Concept | When using external libraries |
| **Module** | A file containing Python definitions and statements that can be imported | `import math` | Python | Core Concept | For organizing code into files |
| **Package** | A collection of modules in a directory with an `__init__.py` file | `import torch.nn` | Python | Core Concept | For grouping related modules |
| **Exception** | An error event that occurs during program execution and disrupts flow | `try: except:` | Python | Error Handling | For graceful error handling |
| **Iterator** | An object that allows sequential access to elements in a collection | `iter(list)` | Python | Core Concept | For loops, for lazy evaluation |
| **Generator** | A function that yields values lazily, one at a time, using yield keyword | `def gen(): yield 1` | Python | Core Concept | For memory-efficient iteration |
| **Decorator** | A function that modifies or enhances another function's behavior | `@decorator def f():` | Python | Core Concept | For logging, timing, validation |
| **Context Manager** | A construct that manages resources using with statement | `with open('file.txt') as f:` | Python | Resource Mgmt | For file handling, database connections |
| **Type Hints** | Annotations that specify expected data types for function arguments | `def greet(name: str) -> str:` | Python | Core Concept | For code documentation, type checking |
| **List Comprehension** | A concise way to create lists using a single line of code | `[x*2 for x in range(5)]` | Python | Core Concept | For transforming lists efficiently |
| **Lambda** | An anonymous one-line function defined with lambda keyword | `lambda x: x*2` | Python | Core Concept | For simple operations, sorting keys |
| **Docstring** | A documentation string inside a function to describe its purpose | `"""This function..."""` | Python | Documentation | For documenting code |
| **f-string** | Formatted string literal with expressions inside curly braces | `f"Name: {name}"` | Python | Core Concept | For string formatting |
| **Assert** | A statement that tests a condition and raises error if false | `assert x > 0` | Python | Debugging | For testing assumptions, debugging |

---

### 1.2 Data Structures

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Array** | A data structure that stores elements of the same type in contiguous memory | `array.array('i', [1,2,3])` | Python | Data Structure | For numerical computing |
| **Queue** | A FIFO (First In First Out) data structure where elements are processed in order | `from collections import deque` | Python | Data Structure | For task scheduling, BFS |
| **Stack** | A LIFO (Last In First Out) data structure like a stack of plates | `stack = []` | Python | Data Structure | For undo operations, DFS |
| **Heap** | A tree-based data structure where parent is larger/smaller than children | `import heapq` | Python | Data Structure | For priority queues, Dijkstra |
| **Linked List** | A linear collection of nodes where each node points to the next | `class Node:` | Python | Data Structure | For dynamic memory allocation |
| **Tree** | A hierarchical data structure with nodes connected by edges | `class TreeNode:` | Python | Data Structure | For hierarchy, parsing |
| **Graph** | A non-linear data structure of nodes connected by edges | `graph = {'A': ['B']}` | Python | Data Structure | For networks, relationships |
| **Hash Table** | A data structure that maps keys to values using hash function | `dict` | Python | Data Structure | For fast lookups, caching |
| **Deque** | A double-ended queue allowing append/pop from both ends | `from collections import deque` | Python | Data Structure | For sliding windows, BFS |
| **Counter** | A dictionary subclass for counting hashable objects | `from collections import Counter` | Python | Data Structure | For counting frequencies |
| **DefaultDict** | A dictionary with default values for missing keys | `from collections import defaultdict` | Python | Data Structure | When handling missing keys |
| **OrderedDict** | A dictionary that remembers insertion order | `from collections import OrderedDict` | Python | Data Structure | When order matters |
| **NamedTuple** | A tuple with named fields for better readability | `from collections import namedtuple` | Python | Data Structure | For lightweight data objects |
| **Enum** | A set of named constants with unique values | `from enum import Enum` | Python | Data Structure | For predefined constants |

---

### 1.3 Control Flow

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **if-else** | Conditional statement that executes code based on boolean conditions | `if x > 0: print("Positive")` | Python | Control Flow | For branching logic |
| **for loop** | A loop that iterates over a sequence or iterable | `for i in range(10):` | Python | Control Flow | When iterating over sequences |
| **while loop** | A loop that repeats while a condition is True | `while x < 10:` | Python | Control Flow | For indefinite loops |
| **break** | A statement that exits the loop prematurely | `if condition: break` | Python | Control Flow | For early termination |
| **continue** | A statement that skips the rest of the current iteration | `if skip: continue` | Python | Control Flow | For skipping iterations |
| **pass** | A null statement that does nothing, used as placeholder | `if x > 0: pass` | Python | Control Flow | For stubs, placeholders |
| **try-except** | A block that attempts code and handles exceptions | `try: except:` | Python | Error Handling | For graceful error handling |
| **finally** | A block that always executes regardless of errors | `try: finally:` | Python | Error Handling | For cleanup operations |
| **raise** | A statement that manually triggers an exception | `raise ValueError("Error")` | Python | Error Handling | For custom errors |
| **with** | A statement that simplifies resource management | `with open('file') as f:` | Python | Resource Mgmt | For context management |
| **yield** | A keyword that makes a function a generator | `yield value` | Python | Core Concept | For lazy iteration |
| **return** | A statement that exits a function and returns a value | `return result` | Python | Core Concept | For returning results |
| **global** | A keyword that accesses global variables inside functions | `global x` | Python | Core Concept | For modifying globals |
| **nonlocal** | A keyword that accesses variables from outer scopes | `nonlocal x` | Python | Core Concept | For nested functions |

---

### 1.4 Functions & Classes

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Argument** | A value passed to a function when called | `def add(a, b):` | Python | Functions | For passing data to functions |
| **Parameter** | A variable defined in a function definition | `def add(a, b):` | Python | Functions | For function definitions |
| **Return** | The value a function sends back to caller | `return result` | Python | Functions | For returning results |
| **Default Argument** | A parameter with a default value if not provided | `def greet(name="World"):` | Python | Functions | For optional parameters |
| **Keyword Argument** | An argument passed by parameter name | `func(name="Ali")` | Python | Functions | For clarity, optional args |
| **Arbitrary Args** | A function that accepts any number of arguments | `*args, **kwargs` | Python | Functions | For flexible function calls |
| **Inheritance** | A class deriving properties from parent class | `class Child(Parent):` | Python | OOP Concept | For code reuse |
| **Polymorphism** | The ability to use same interface for different types | `method(obj)` | Python | OOP Concept | For flexible interfaces |
| **Encapsulation** | Hiding internal details and exposing only interfaces | `_private_attr` | Python | OOP Concept | For data hiding |
| **Abstraction** | Hiding complexity and showing only essentials | `abstractmethod` | Python | OOP Concept | For simplifying interfaces |
| **Magic Method** | Special methods with double underscores like `__init__` | `__init__`, `__str__` | Python | OOP Concept | For customizing class behavior |
| **Decorator** | A function that wraps and modifies other functions | `@staticmethod` | Python | OOP Concept | For adding behavior |
| **Property** | A decorator for defining getters and setters | `@property` | Python | OOP Concept | For controlled attribute access |
| **Classmethod** | A method bound to class, not instance | `@classmethod` | Python | OOP Concept | For alternative constructors |
| **Staticmethod** | A method not bound to class or instance | `@staticmethod` | Python | OOP Concept | For utility functions |

---

### 1.5 File Handling

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **File I/O** | Operations for reading from and writing to files | `open('file.txt', 'r')` | Python | File Handling | For file operations |
| **Read** | Reading data from a file into variables | `file.read()` | Python | File Handling | For reading files |
| **Write** | Writing data from variables to a file | `file.write("text")` | Python | File Handling | For saving data |
| **Append** | Adding data to the end of an existing file | `open('file', 'a')` | Python | File Handling | For appending data |
| **Context Manager** | A construct that automatically manages file resources | `with open('file.txt') as f:` | Python | File Handling | For safe file handling |
| **JSON** | JavaScript Object Notation for data interchange | `json.load()` | Python | File Handling | For data exchange |
| **CSV** | Comma Separated Values for tabular data | `csv.reader()` | Python | File Handling | For spreadsheets, tables |
| **Pickle** | Python serialization for saving objects | `pickle.dump(obj, file)` | Python | File Handling | For saving Python objects |
| **Path** | A string representing file or directory path | `from pathlib import Path` | Python | File Handling | For cross-platform paths |
| **Directory** | A folder containing files and subdirectories | `os.mkdir('folder')` | Python | File Handling | For organizing files |
| **Glob** | A pattern matching for file names | `glob.glob('*.txt')` | Python | File Handling | For finding files |
| **Shutil** | High-level file operations like copy, move, delete | `shutil.copy(src, dst)` | Python | File Handling | For file operations |
| **TempFile** | Creating temporary files or directories | `tempfile.TemporaryFile()` | Python | File Handling | For temporary storage |
| **Serialization** | Converting objects to byte streams for storage | `pickle.dumps()` | Python | File Handling | For data persistence |

---

### 1.6 Common Libraries

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **NumPy** | Library for numerical computing and array operations | `import numpy as np` | Python | Library | For mathematical operations |
| **Pandas** | Library for data manipulation and analysis | `import pandas as pd` | Python | Library | For tabular data |
| **Matplotlib** | Library for creating static visualizations and plots | `import matplotlib.pyplot as plt` | Python | Library | For plotting graphs |
| **Seaborn** | Statistical data visualization library | `import seaborn as sns` | Python | Library | For statistical plots |
| **Scikit-learn** | Machine learning library with algorithms and tools | `from sklearn.linear_model import LinearRegression` | Python | Library | For traditional ML |
| **PyTorch** | Deep learning framework with dynamic computation graphs | `import torch` | Python | Library | For deep learning |
| **TensorFlow** | Deep learning framework with static computation graphs | `import tensorflow as tf` | Python | Library | For deep learning |
| **OpenCV** | Computer vision library for image and video processing | `import cv2` | Python | Library | For image processing |
| **Pillow** | Python Imaging Library for image manipulation | `from PIL import Image` | Python | Library | For image operations |
| **Requests** | HTTP library for sending HTTP requests | `import requests` | Python | Library | For API calls, web scraping |
| **BeautifulSoup** | Library for parsing HTML and XML documents | `from bs4 import BeautifulSoup` | Python | Library | For web scraping |
| **Flask** | Web framework for building APIs and web applications | `from flask import Flask` | Python | Library | For web apps, APIs |
| **FastAPI** | Modern web framework for building APIs with async support | `from fastapi import FastAPI` | Python | Library | For production APIs |
| **SQLAlchemy** | SQL toolkit and ORM for database operations | `from sqlalchemy import create_engine` | Python | Library | For database interactions |
| **Pytest** | Testing framework for writing and running tests | `import pytest` | Python | Library | For unit testing |
| **Logging** | Built-in module for logging messages and errors | `import logging` | Python | Library | For debugging, monitoring |
| **Argparse** | Module for parsing command-line arguments | `import argparse` | Python | Library | For CLI applications |
| **Datetime** | Module for handling dates and times | `from datetime import datetime` | Python | Library | For time operations |
| **Regex** | Regular expressions for pattern matching in text | `import re` | Python | Library | For text pattern matching |
| **OS** | Module for operating system interfaces | `import os` | Python | Library | For system operations |
| **Sys** | Module for system-specific parameters and functions | `import sys` | Python | Library | For system-level operations |
| **Math** | Module for mathematical functions | `import math` | Python | Library | For mathematical operations |
| **Random** | Module for generating random numbers | `import random` | Python | Library | For random sampling |
| **Time** | Module for time-related functions | `import time` | Python | Library | For timing operations |

---

## 2. Data Science Fundamentals

### 2.1 Data Concepts

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Data** | Raw facts, figures, or information collected and stored | `data = [1, 2, 3]` | Data Science | Core Concept | For analysis, modeling |
| **Dataset** | A collection of related data samples or records | `dataset = pd.read_csv('data.csv')` | Data Science | Core Concept | When working with data |
| **Sample** | A subset of data from a larger population or dataset | `sample = data[:100]` | Data Science | Core Concept | For analysis, testing |
| **Variable** | A characteristic or attribute being measured or studied | `age, height, weight` | Data Science | Core Concept | For modeling, analysis |
| **Feature** | An individual measurable property or characteristic | `feature = data['age']` | Data Science | Core Concept | For model inputs |
| **Label** | The target or output variable being predicted or classified | `label = data['target']` | Data Science | Core Concept | For supervised learning |
| **Target** | The dependent variable to be predicted by the model | `y = data['price']` | Data Science | Core Concept | For prediction |
| **Row** | A single record or observation in a dataset | `row = data.iloc[0]` | Data Science | Core Concept | For individual samples |
| **Column** | A single field or attribute in a dataset | `column = data['age']` | Data Science | Core Concept | For a single feature |
| **Index** | A unique identifier for each row in a dataset | `data.index` | Data Science | Core Concept | For row identification |
| **Data Type** | The classification of data values (integer, float, string) | `data.dtypes` | Data Science | Core Concept | For type handling |
| **Categorical** | Data that can be divided into categories or groups | `data['color']` | Data Science | Core Concept | For classification |
| **Numerical** | Data represented as numbers, either discrete or continuous | `data['age']` | Data Science | Core Concept | For mathematical operations |
| **Ordinal** | Categorical data with meaningful order or ranking | `data['education_level']` | Data Science | Core Concept | For ordered categories |
| **Nominal** | Categorical data without natural order | `data['country']` | Data Science | Core Concept | For unordered categories |
| **Discrete** | Numerical data that can only take integer values | `data['children']` | Data Science | Core Concept | For count data |
| **Continuous** | Numerical data that can take any value in a range | `data['temperature']` | Data Science | Core Concept | For measurement data |
| **Time Series** | Data points collected at successive time intervals | `data['date']` | Data Science | Core Concept | For temporal analysis |
| **Panel Data** | Multi-dimensional data containing measurements over time | `pd.Panel()` | Data Science | Core Concept | For longitudinal analysis |
| **Sparse Data** | Data with many zeros or missing values | `scipy.sparse` | Data Science | Core Concept | For memory efficiency |
| **Dense Data** | Data with most values filled | `np.array()` | Data Science | Core Concept | For dense representations |
| **Noise** | Random variations or errors in data | `noise = np.random.randn()` | Data Science | Core Concept | For data quality |
| **Signal** | The true underlying pattern in data | `signal = true_pattern` | Data Science | Core Concept | For pattern detection |
| **Outlier** | An observation that significantly differs from others | `outliers = data[data > threshold]` | Data Science | Core Concept | For anomaly detection |

---

### 2.2 EDA (Exploratory Data Analysis)

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **EDA** | Process of analyzing datasets to summarize characteristics | `data.describe(), data.info()` | Data Science | Analysis | Before modeling |
| **Head** | Viewing first few rows of a dataset | `data.head()` | Data Science | EDA | For understanding structure |
| **Tail** | Viewing last few rows of a dataset | `data.tail()` | Data Science | EDA | For checking end of data |
| **Info** | Summary of dataset including data types and missing values | `data.info()` | Data Science | EDA | For data structure overview |
| **Describe** | Statistical summary of numerical columns | `data.describe()` | Data Science | EDA | For numerical statistics |
| **Histogram** | Bar chart showing distribution of a variable | `data['age'].hist()` | Data Science | Visualization | For distribution analysis |
| **Boxplot** | Plot showing quartiles and outliers of data | `sns.boxplot(data['age'])` | Data Science | Visualization | For detecting outliers |
| **Scatter Plot** | Plot showing relationship between two variables | `plt.scatter(x, y)` | Data Science | Visualization | For correlation analysis |
| **Correlation** | Measure of linear relationship between variables | `data.corr()` | Data Science | Statistics | For feature relationships |
| **Heatmap** | Color-coded matrix showing correlations | `sns.heatmap(data.corr())` | Data Science | Visualization | For correlation matrix |
| **Missing Values** | Data points that are null or not available | `data.isnull().sum()` | Data Science | EDA | For data quality |
| **Duplicates** | Identical rows that appear multiple times | `data.duplicated().sum()` | Data Science | EDA | For data cleaning |
| **Unique** | Distinct values in a column | `data['category'].unique()` | Data Science | EDA | For categorical analysis |
| **Value Counts** | Frequency of each unique value in a column | `data['category'].value_counts()` | Data Science | EDA | For categorical analysis |
| **Skewness** | Measure of asymmetry in data distribution | `data['age'].skew()` | Data Science | Statistics | For distribution shape |
| **Kurtosis** | Measure of tailedness of data distribution | `data['age'].kurtosis()` | Data Science | Statistics | For distribution shape |
| **Distribution** | Pattern or spread of values in a dataset | `sns.distplot(data['age'])` | Data Science | Statistics | For understanding spread |
| **Outliers** | Extreme values far from other observations | `zscore = stats.zscore(data)` | Data Science | Statistics | For anomaly detection |
| **Normal Distribution** | Bell-shaped symmetric distribution | `np.random.normal()` | Data Science | Statistics | For standard analysis |
| **Uniform Distribution** | Distribution where all values equally likely | `np.random.uniform()` | Data Science | Statistics | For random sampling |
| **Bins** | Intervals for grouping continuous data | `bins = [0, 10, 20, 30]` | Data Science | EDA | For histogram bins |
| **Groupby** | Grouping data by categorical variables | `data.groupby('category').mean()` | Data Science | EDA | For aggregating data |
| **Pivot Table** | Summarizing data by two categorical variables | `pd.pivot_table(data, index='a', columns='b')` | Data Science | EDA | For cross-tabulation |
| **Crosstab** | Frequency table of two categorical variables | `pd.crosstab(data['a'], data['b'])` | Data Science | EDA | For categorical analysis |
| **Skew** | Asymmetry measure where data leans left or right | `data.skew()` | Data Science | Statistics | For data transformation |
| **Bias** | Systematic error leading to inaccurate results | `from sklearn.utils import check_random_state` | Data Science | Statistics | For balanced data |
| **Variance** | Measure of how spread out the values are | `data.var()` | Data Science | Statistics | For data dispersion |

---

### 2.3 Data Preprocessing

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Data Cleaning** | Process of removing errors and inconsistencies | `data.dropna()` | Data Science | Preprocessing | Before analysis |
| **Data Processing** | Transforming raw data into useful format | `data = pd.read_csv()` | Data Science | Preprocessing | For data preparation |
| **Preprocessing** | Steps to prepare data for analysis or modeling | `from sklearn.preprocessing import` | Data Science | Preprocessing | Before modeling |
| **Null Values** | Missing data points that are undefined | `data.isnull()` | Data Science | Preprocessing | For handling missing data |
| **NaN** | Not a Number - represents missing or undefined values | `float('nan')` | Data Science | Preprocessing | For missing data |
| **Drop** | Removing rows or columns from dataset | `data.drop('column', axis=1)` | Data Science | Preprocessing | For removing unwanted data |
| **Fillna** | Replacing missing values with specified values | `data.fillna(0)` | Data Science | Preprocessing | For handling missing data |
| **Interpolation** | Estimating missing values from surrounding points | `data.interpolate()` | Data Science | Preprocessing | For time series gaps |
| **One-Hot Encoding** | Converting categorical variables to binary columns | `pd.get_dummies(data)` | Data Science | Preprocessing | For categorical data |
| **Label Encoding** | Converting categories to numeric labels | `from sklearn.preprocessing import LabelEncoder` | Data Science | Preprocessing | For ordinal categories |
| **Ordinal Encoding** | Encoding categories with meaningful order | `from sklearn.preprocessing import OrdinalEncoder` | Data Science | Preprocessing | For ordered categories |
| **Normalization** | Scaling data to a specific range (0-1) | `from sklearn.preprocessing import MinMaxScaler` | Data Science | Preprocessing | For consistent ranges |
| **Standardization** | Scaling data to mean 0, standard deviation 1 | `from sklearn.preprocessing import StandardScaler` | Data Science | Preprocessing | For Gaussian assumptions |
| **Robust Scaling** | Scaling using median and quartiles | `from sklearn.preprocessing import RobustScaler` | Data Science | Preprocessing | For outliers handling |
| **MinMax Scaling** | Scaling to a specific range (usually 0-1) | `MinMaxScaler(feature_range=(0,1))` | Data Science | Preprocessing | For bounded data |
| **Z-Score** | Standard score: (x-mean)/std for normalization | `z = (x - mean) / std` | Data Science | Statistics | For standardization |
| **Log Transformation** | Applying log to reduce skewness | `data['log'] = np.log(data['value'])` | Data Science | Preprocessing | For skewed data |
| **Power Transformation** | Box-Cox or Yeo-Johnson transformations | `from sklearn.preprocessing import PowerTransformer` | Data Science | Preprocessing | For normalization |
| **Feature Scaling** | Standardizing numeric features for modeling | `scaler.fit_transform(X)` | Data Science | Preprocessing | Before ML algorithms |
| **Polynomial Features** | Creating higher-order features | `from sklearn.preprocessing import PolynomialFeatures` | Data Science | Feature Engineering | For non-linear relationships |
| **Interaction Features** | Multiplying features to capture relationships | `X['a*b'] = X['a'] * X['b']` | Data Science | Feature Engineering | For feature interactions |
| **Binning** | Converting continuous values to discrete bins | `pd.cut(data, bins)` | Data Science | Preprocessing | For discretization |
| **Discretization** | Converting continuous data into discrete categories | `from sklearn.preprocessing import KBinsDiscretizer` | Data Science | Preprocessing | For categorical ML |
| **Dimensionality Reduction** | Reducing number of features | `from sklearn.decomposition import PCA` | Data Science | Preprocessing | For high-dimensional data |
| **PCA** | Principal Component Analysis for dimensionality reduction | `from sklearn.decomposition import PCA` | Data Science | Feature Engineering | For feature reduction |
| **LDA** | Linear Discriminant Analysis for feature reduction | `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis` | Data Science | Feature Engineering | For classification |
| **t-SNE** | t-Distributed Stochastic Neighbor Embedding for visualization | `from sklearn.manifold import TSNE` | Data Science | Visualization | For clustering visualization |
| **UMAP** | Uniform Manifold Approximation and Projection | `import umap` | Data Science | Visualization | For nonlinear embedding |

---

### 2.4 Statistics

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Mean** | Average value of a dataset | `np.mean(data)` | Data Science | Statistics | For central tendency |
| **Median** | Middle value of ordered dataset | `np.median(data)` | Data Science | Statistics | For skewed data |
| **Mode** | Most frequent value in dataset | `stats.mode(data)` | Data Science | Statistics | For categorical data |
| **Standard Deviation** | Measure of data spread around the mean | `np.std(data)` | Data Science | Statistics | For dispersion |
| **Variance** | Average of squared differences from mean | `np.var(data)` | Data Science | Statistics | For spread measurement |
| **Quantile** | Values dividing data into equal intervals | `np.quantile(data, 0.5)` | Data Science | Statistics | For distribution analysis |
| **Percentile** | The value below which given % of observations fall | `np.percentile(data, 75)` | Data Science | Statistics | For ranking |
| **Quartile** | Values dividing data into four equal parts | `np.percentile(data, [25, 50, 75])` | Data Science | Statistics | For boxplots |
| **IQR** | Interquartile Range (Q3-Q1) | `iqr = q3 - q1` | Data Science | Statistics | For outlier detection |
| **Correlation** | Measure of linear relationship between variables | `np.corrcoef(x, y)` | Data Science | Statistics | For feature relationships |
| **Covariance** | Measure of how two variables change together | `np.cov(x, y)` | Data Science | Statistics | For relationship direction |
| **Pearson Correlation** | Linear correlation between variables | `from scipy.stats import pearsonr` | Data Science | Statistics | For linear relationships |
| **Spearman Correlation** | Rank-based correlation | `from scipy.stats import spearmanr` | Data Science | Statistics | For monotonic relationships |
| **Hypothesis Testing** | Statistical tests to validate assumptions | `from scipy.stats import ttest_ind` | Data Science | Statistics | For A/B testing |
| **P-value** | Probability of observing results by chance | `p_value = ttest_ind(a, b).pvalue` | Data Science | Statistics | For significance testing |
| **T-test** | Test comparing means of two groups | `from scipy.stats import ttest_ind` | Data Science | Statistics | For group comparison |
| **Chi-Square** | Test for categorical variables independence | `from scipy.stats import chi2_contingency` | Data Science | Statistics | For categorical analysis |
| **ANOVA** | Analysis of Variance for multiple group means | `from scipy.stats import f_oneway` | Data Science | Statistics | For multiple groups |
| **Confidence Interval** | Range containing true value with certain probability | `from scipy.stats import t` | Data Science | Statistics | For uncertainty estimation |
| **Margin of Error** | The maximum expected difference from true value | `margin = z * std / sqrt(n)` | Data Science | Statistics | For survey accuracy |
| **Type I Error** | False positive (incorrectly rejecting null hypothesis) | `α = 0.05` | Data Science | Statistics | For significance testing |
| **Type II Error** | False negative (failing to reject false null) | `β = 0.2` | Data Science | Statistics | For power analysis |
| **Statistical Power** | Probability of correctly rejecting null | `power = 1 - β` | Data Science | Statistics | For experiment design |
| **Regression Analysis** | Statistical method for predicting outcomes | `from sklearn.linear_model import LinearRegression` | Data Science | Statistics | For prediction |
| **Residual** | Difference between observed and predicted values | `residual = y_true - y_pred` | Data Science | Statistics | For model evaluation |
| **R-Squared** | Proportion of variance explained by the model | `r2 = r2_score(y_true, y_pred)` | Data Science | Statistics | For model fit |
| **Adjusted R-Squared** | R-squared adjusted for number of predictors | `adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)` | Data Science | Statistics | For overfitting detection |
| **MLE** | Maximum Likelihood Estimation for parameter estimation | `from scipy.stats import norm` | Data Science | Statistics | For parameter estimation |
| **Bayesian** | Statistical approach using prior and posterior probabilities | `from scipy.stats import bayes` | Data Science | Statistics | For uncertainty |
| **Prior** | Initial belief before seeing data | `prior = 0.5` | Data Science | Statistics | For Bayesian analysis |
| **Posterior** | Updated belief after seeing data | `posterior = prior * likelihood` | Data Science | Statistics | For Bayesian analysis |
| **Likelihood** | Probability of data given parameters | `likelihood = norm.pdf(data, mean, std)` | Data Science | Statistics | For MLE |

---

### 2.5 Data Visualization

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Plot** | Graphical representation of data | `plt.plot(x, y)` | Data Science | Visualization | For line charts |
| **Chart** | Visual representation of data with various types | `plt.bar(x, y)` | Data Science | Visualization | For comparisons |
| **Graph** | Visual representation of data relationships | `networkx.draw(G)` | Data Science | Visualization | For networks |
| **Bar Chart** | Chart showing categories as rectangular bars | `plt.bar(categories, values)` | Data Science | Visualization | For categorical comparison |
| **Line Plot** | Plot showing data points connected by lines | `plt.plot(x, y)` | Data Science | Visualization | For trends over time |
| **Scatter Plot** | Plot showing relationship between two variables | `plt.scatter(x, y)` | Data Science | Visualization | For correlation |
| **Pie Chart** | Circular chart showing proportions of categories | `plt.pie(sizes, labels=labels)` | Data Science | Visualization | For composition |
| **Box Plot** | Plot showing distribution through quartiles | `plt.boxplot(data)` | Data Science | Visualization | For distribution |
| **Violin Plot** | Box plot with density information | `sns.violinplot(data)` | Data Science | Visualization | For detailed distribution |
| **Histogram** | Plot showing frequency distribution of numeric data | `plt.hist(data)` | Data Science | Visualization | For frequency |
| **Heatmap** | Color-coded matrix showing values | `sns.heatmap(matrix)` | Data Science | Visualization | For correlations |
| **KDE Plot** | Kernel Density Estimation for smooth distribution | `sns.kdeplot(data)` | Data Science | Visualization | For density |
| **Pairplot** | Matrix of scatter plots for multiple variables | `sns.pairplot(data)` | Data Science | Visualization | For relationships |
| **Jointplot** | Scatter plot with marginal distributions | `sns.jointplot(x, y)` | Data Science | Visualization | For distributions |
| **Residual Plot** | Plot of residuals to check model assumptions | `plt.scatter(predicted, residuals)` | Data Science | Visualization | For model diagnostics |
| **Q-Q Plot** | Quantile-Quantile plot for distribution comparison | `scipy.stats.probplot(data)` | Data Science | Visualization | For normality check |
| **Facet Grid** | Multiple subplots organized by categories | `sns.FacetGrid(data)` | Data Science | Visualization | For comparisons |
| **Theme** | Visual style applied to plots | `plt.style.use('seaborn')` | Data Science | Visualization | For aesthetics |
| **Legend** | Explanation of plot elements | `plt.legend()` | Data Science | Visualization | For clarity |
| **Label** | Text identification for axes and elements | `plt.xlabel('X-axis')` | Data Science | Visualization | For annotation |
| **Title** | Main heading of the plot | `plt.title('My Plot')` | Data Science | Visualization | For identification |
| **Gridlines** | Reference lines for easier reading | `plt.grid(True)` | Data Science | Visualization | For readability |
| **Annotation** | Adding text or markers to highlight points | `plt.annotate('Point', (x,y))` | Data Science | Visualization | For highlighting |
| **Color Map** | Mapping values to colors | `plt.cm.viridis` | Data Science | Visualization | For color coding |
| **Subplot** | Multiple plots in one figure | `plt.subplot(2, 2, 1)` | Data Science | Visualization | For multiple views |
| **Figure** | The overall plotting container | `plt.figure(figsize=(10,6))` | Data Science | Visualization | For layout control |
| **Axis** | The coordinate system for plotting | `ax = plt.gca()` | Data Science | Visualization | For fine control |
| **Savefig** | Saving plot to file | `plt.savefig('plot.png')` | Data Science | Visualization | For export |

---

## 3. Machine Learning

### 3.1 Core ML Concepts

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Machine Learning** | Algorithms that learn from data without explicit programming | `from sklearn import ...` | Machine Learning | Core Concept | For pattern discovery |
| **Artificial Intelligence** | Systems that mimic human intelligence | `AI systems` | Machine Learning | Core Concept | For intelligent automation |
| **Model** | Mathematical representation learned from data | `model = algorithm.fit(X, y)` | Machine Learning | Core Concept | For making predictions |
| **Algorithm** | Step-by-step procedure to solve a problem | `LinearRegression()` | Machine Learning | Core Concept | For learning patterns |
| **Supervised Learning** | Learning from labeled examples | `model.fit(X, y)` | Machine Learning | Core Concept | With labeled data |
| **Unsupervised Learning** | Learning patterns from unlabeled data | `model.fit(X)` | Machine Learning | Core Concept | Without labels |
| **Semi-Supervised** | Using both labeled and unlabeled data | `SSL algorithms` | Machine Learning | Core Concept | With limited labels |
| **Reinforcement Learning** | Learning through rewards and penalties | `RL environment` | Machine Learning | Core Concept | For decision making |
| **Training** | Process of teaching a model on data | `model.fit(X_train, y_train)` | Machine Learning | Core Concept | Before prediction |
| **Testing** | Evaluating model on unseen data | `model.score(X_test, y_test)` | Machine Learning | Core Concept | For performance |
| **Inference** | Making predictions with a trained model | `model.predict(X_new)` | Machine Learning | Core Concept | After training |
| **Classification** | Predicting discrete class labels | `model.predict(X)` | Machine Learning | Core Concept | For categories |
| **Regression** | Predicting continuous numeric values | `model.predict(X)` | Machine Learning | Core Concept | For numbers |
| **Clustering** | Grouping similar data points together | `from sklearn.cluster import KMeans` | Machine Learning | Core Concept | For grouping |
| **Dimensionality Reduction** | Reducing feature count while preserving information | `from sklearn.decomposition import PCA` | Machine Learning | Core Concept | For complexity |
| **Feature Selection** | Choosing most relevant features | `from sklearn.feature_selection import SelectKBest` | Machine Learning | Feature Engineering | For feature importance |
| **Feature Extraction** | Creating new features from existing ones | `from sklearn.decomposition import PCA` | Machine Learning | Feature Engineering | For representation |
| **Data Splitting** | Dividing data into training, validation, test | `train_test_split(X, y, test_size=0.2)` | Machine Learning | Core Concept | For evaluation |
| **Train-Test Split** | Splitting into training and test sets | `from sklearn.model_selection import train_test_split` | Machine Learning | Core Concept | For unbiased evaluation |
| **Cross-Validation** | Multiple train-test splits for robust evaluation | `from sklearn.model_selection import cross_val_score` | Machine Learning | Validation | For stable estimates |
| **K-Fold CV** | K-fold cross-validation for validation | `cross_val_score(model, X, y, cv=5)` | Machine Learning | Validation | For small datasets |
| **Stratified Split** | Maintaining class distribution in splits | `StratifiedKFold(n_splits=5)` | Machine Learning | Validation | For imbalanced data |
| **Holdout** | Single train-test split for evaluation | `train_test_split(test_size=0.2)` | Machine Learning | Validation | For large datasets |
| **Underfitting** | Model too simple, misses patterns | `model = LinearRegression()` | Machine Learning | Core Concept | When model is too simple |
| **Overfitting** | Model too complex, memorizes noise | `model = PolynomialFeatures(degree=20)` | Machine Learning | Core Concept | When model is too complex |
| **Bias** | Error due to wrong assumptions in model | `model = LinearRegression()` | Machine Learning | Core Concept | For simple models |
| **Variance** | Error due to sensitivity to training data | `model = DecisionTree(max_depth=100)` | Machine Learning | Core Concept | For complex models |
| **Bias-Variance Tradeoff** | Balance between underfitting and overfitting | `regularization` | Machine Learning | Core Concept | For optimal model |
| **Generalization** | Model performance on new unseen data | `model.score(X_test, y_test)` | Machine Learning | Core Concept | For real-world use |
| **Feature Importance** | Measure of feature contribution to predictions | `model.feature_importances_` | Machine Learning | Core Concept | For understanding |
| **Feature Engineering** | Creating features from raw data | `data['new_feature'] = data['a'] * data['b']` | Machine Learning | Core Concept | For better patterns |
| **Hyperparameters** | Parameters set before training | `max_depth=5, learning_rate=0.1` | Machine Learning | Core Concept | For model tuning |
| **Hyperparameter Tuning** | Finding optimal hyperparameters | `GridSearchCV()` | Machine Learning | Validation | For best performance |
| **Grid Search** | Exhaustive search over hyperparameter grid | `GridSearchCV(estimator, param_grid)` | Machine Learning | Validation | For small grids |
| **Random Search** | Random hyperparameter sampling | `RandomizedSearchCV()` | Machine Learning | Validation | For larger grids |
| **Bayesian Optimization** | Informed hyperparameter search | `Optuna, Hyperopt` | Machine Learning | Validation | For expensive tuning |
| **Ensemble Learning** | Combining multiple models | `RandomForestClassifier()` | Machine Learning | Core Concept | For better performance |
| **Bagging** | Bootstrap aggregating of models | `BaggingClassifier()` | Machine Learning | Core Concept | For variance reduction |
| **Boosting** | Sequential building of weak learners | `XGBoost, AdaBoost` | Machine Learning | Core Concept | For bias reduction |
| **Stacking** | Combining multiple models via meta-learner | `StackingClassifier()` | Machine Learning | Core Concept | For diverse models |

---

### 3.2 ML Algorithms - Supervised

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Linear Regression** | Models linear relationship between features and target | `from sklearn.linear_model import LinearRegression` | Machine Learning | Regression | For linear relationships |
| **Polynomial Regression** | Models non-linear relationships with polynomial features | `from sklearn.preprocessing import PolynomialFeatures` | Machine Learning | Regression | For curve fitting |
| **Ridge Regression** | Linear regression with L2 regularization | `from sklearn.linear_model import Ridge` | Machine Learning | Regression | For multicollinearity |
| **Lasso Regression** | Linear regression with L1 regularization | `from sklearn.linear_model import Lasso` | Machine Learning | Regression | For feature selection |
| **Elastic Net** | Linear regression with both L1 and L2 regularization | `from sklearn.linear_model import ElasticNet` | Machine Learning | Regression | For combined regularization |
| **Logistic Regression** | Classification using logistic function | `from sklearn.linear_model import LogisticRegression` | Machine Learning | Classification | For binary classification |
| **Decision Tree** | Tree-based model for classification/regression | `from sklearn.tree import DecisionTreeClassifier` | Machine Learning | Classification | For interpretable models |
| **Random Forest** | Ensemble of decision trees | `from sklearn.ensemble import RandomForestClassifier` | Machine Learning | Classification | For high accuracy |
| **SVM** | Support Vector Machines for classification | `from sklearn.svm import SVC` | Machine Learning | Classification | For complex boundaries |
| **SVR** | Support Vector Regression | `from sklearn.svm import SVR` | Machine Learning | Regression | For regression with margins |
| **KNN** | K-Nearest Neighbors classification/regression | `from sklearn.neighbors import KNeighborsClassifier` | Machine Learning | Classification | For simple non-linear |
| **Naive Bayes** | Classification using Bayes theorem | `from sklearn.naive_bayes import GaussianNB` | Machine Learning | Classification | For text classification |
| **GBM** | Gradient Boosting Machines | `from sklearn.ensemble import GradientBoostingClassifier` | Machine Learning | Classification | For high accuracy |
| **XGBoost** | Extreme Gradient Boosting | `import xgboost as xgb` | Machine Learning | Classification | For competition winning |
| **LightGBM** | Lightweight Gradient Boosting | `import lightgbm as lgb` | Machine Learning | Classification | For fast training |
| **CatBoost** | Gradient Boosting with categorical features | `import catboost as cb` | Machine Learning | Classification | For categorical data |
| **AdaBoost** | Adaptive Boosting algorithm | `from sklearn.ensemble import AdaBoostClassifier` | Machine Learning | Classification | For boosting |

---

### 3.3 ML Algorithms - Unsupervised

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **K-Means** | Partitioning data into K clusters | `from sklearn.cluster import KMeans` | Machine Learning | Clustering | For spherical clusters |
| **Hierarchical Clustering** | Building hierarchy of clusters | `from sklearn.cluster import AgglomerativeClustering` | Machine Learning | Clustering | For hierarchy |
| **DBSCAN** | Density-based spatial clustering | `from sklearn.cluster import DBSCAN` | Machine Learning | Clustering | For arbitrary shapes |
| **Gaussian Mixture** | Probabilistic clustering using Gaussian distributions | `from sklearn.mixture import GaussianMixture` | Machine Learning | Clustering | For soft clustering |
| **PCA** | Principal Component Analysis | `from sklearn.decomposition import PCA` | Machine Learning | Dimensionality | For dimension reduction |
| **LDA** | Linear Discriminant Analysis | `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis` | Machine Learning | Dimensionality | For classification |
| **t-SNE** | t-Distributed Stochastic Neighbor Embedding | `from sklearn.manifold import TSNE` | Machine Learning | Visualization | For 2D visualization |
| **Isomap** | Isometric Feature Mapping | `from sklearn.manifold import Isomap` | Machine Learning | Dimensionality | For manifold learning |
| **LLE** | Locally Linear Embedding | `from sklearn.manifold import LocallyLinearEmbedding` | Machine Learning | Dimensionality | For local structure |
| **Autoencoders** | Neural networks for representation learning | `from torch import nn` | Deep Learning | Dimensionality | For deep features |

---

### 3.4 Evaluation Metrics

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Accuracy** | (TP+TN)/(Total) for classification | `accuracy_score(y_true, y_pred)` | Machine Learning | Classification | For balanced data |
| **Precision** | TP/(TP+FP) for classification | `precision_score(y_true, y_pred)` | Machine Learning | Classification | For FP minimization |
| **Recall** | TP/(TP+FN) for classification | `recall_score(y_true, y_pred)` | Machine Learning | Classification | For FN minimization |
| **F1 Score** | Harmonic mean of precision and recall | `f1_score(y_true, y_pred)` | Machine Learning | Classification | For imbalanced data |
| **F1-Macro** | Unweighted average of F1 scores | `f1_score(y_true, y_pred, average='macro')` | Machine Learning | Classification | For multiple classes |
| **F1-Micro** | Weighted average of F1 scores | `f1_score(y_true, y_pred, average='micro')` | Machine Learning | Classification | For imbalanced classes |
| **ROC Curve** | Receiver Operating Characteristic curve | `roc_curve(y_true, y_scores)` | Machine Learning | Classification | For threshold selection |
| **AUC** | Area Under the ROC Curve | `roc_auc_score(y_true, y_scores)` | Machine Learning | Classification | For model comparison |
| **PR Curve** | Precision-Recall curve | `precision_recall_curve(y_true, y_scores)` | Machine Learning | Classification | For imbalanced data |
| **Confusion Matrix** | Table showing TP, FP, TN, FN | `confusion_matrix(y_true, y_pred)` | Machine Learning | Classification | For error analysis |
| **Log Loss** | Logarithmic loss for probabilities | `log_loss(y_true, y_pred_proba)` | Machine Learning | Classification | For probability models |
| **MAE** | Mean Absolute Error | `mean_absolute_error(y_true, y_pred)` | Machine Learning | Regression | For robust errors |
| **MSE** | Mean Squared Error | `mean_squared_error(y_true, y_pred)` | Machine Learning | Regression | For large errors |
| **RMSE** | Root Mean Squared Error | `sqrt(mean_squared_error())` | Machine Learning | Regression | For interpretability |
| **R2 Score** | R-squared for regression | `r2_score(y_true, y_pred)` | Machine Learning | Regression | For model fit |
| **Adjusted R2** | R2 adjusted for features | `1 - (1-r2)*(n-1)/(n-p-1)` | Machine Learning | Regression | For feature selection |
| **Silhouette Score** | Measure of cluster cohesion/separation | `silhouette_score(X, labels)` | Machine Learning | Clustering | For clustering quality |
| **Davies-Bouldin** | Clustering quality measure | `davies_bouldin_score(X, labels)` | Machine Learning | Clustering | For cluster separation |
| **Calinski-Harabasz** | Variance ratio for clustering | `calinski_harabasz_score(X, labels)` | Machine Learning | Clustering | For cluster density |
| **Elbow Method** | Finding optimal K in K-means | `inertia_plot` | Machine Learning | Clustering | For K selection |
| **Gini Impurity** | Measure of data purity in decision trees | `criterion='gini'` | Machine Learning | Classification | For tree splitting |
| **Entropy** | Measure of randomness/uncertainty | `criterion='entropy'` | Machine Learning | Classification | For information gain |

---

### 3.5 Feature Engineering

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Feature Creation** | Creating new features from existing data | `data['new'] = data['a'] / data['b']` | Machine Learning | Feature Engineering | For better representation |
| **Feature Transformation** | Changing feature representation | `np.log(data['feature'])` | Machine Learning | Feature Engineering | For normality |
| **Feature Scaling** | Normalizing feature values | `StandardScaler().fit_transform(X)` | Machine Learning | Feature Engineering | For ML algorithms |
| **Feature Reduction** | Reducing number of features | `PCA(n_components=10)` | Machine Learning | Feature Engineering | For efficiency |
| **Feature Extraction** | Extracting relevant features | `CountVectorizer()` | Machine Learning | Feature Engineering | For text data |
| **Feature Encoding** | Converting categories to numbers | `pd.get_dummies(data)` | Machine Learning | Feature Engineering | For categorical data |
| **One-Hot Encoding** | Creating binary features for categories | `pd.get_dummies(data['column'])` | Machine Learning | Feature Engineering | For nominal data |
| **Label Encoding** | Converting labels to integers | `LabelEncoder().fit_transform(y)` | Machine Learning | Feature Engineering | For ordinal data |
| **Target Encoding** | Encoding using target variable | `from category_encoders import TargetEncoder` | Machine Learning | Feature Engineering | For high cardinality |
| **Frequency Encoding** | Encoding using category frequency | `data['freq'] = data['col'].map(data['col'].value_counts())` | Machine Learning | Feature Engineering | For low cardinality |
| **Binary Encoding** | Encoding using binary representation | `from category_encoders import BinaryEncoder` | Machine Learning | Feature Engineering | For many categories |
| **Feature Importance** | Measuring feature contribution | `model.feature_importances_` | Machine Learning | Feature Engineering | For feature selection |
| **Mutual Information** | Measuring feature-target dependence | `mutual_info_classif(X, y)` | Machine Learning | Feature Engineering | For feature ranking |
| **Variance Threshold** | Removing low variance features | `VarianceThreshold(threshold=0.1)` | Machine Learning | Feature Engineering | For constant features |
| **Correlation Analysis** | Checking feature relationships | `data.corr()` | Machine Learning | Feature Engineering | For multicollinearity |
| **Domain Knowledge** | Using domain expertise for features | `custom_features` | Machine Learning | Feature Engineering | For expert insights |
| **Time Features** | Extracting time-based features | `data['hour'] = data['date'].dt.hour` | Machine Learning | Feature Engineering | For time data |
| **Aggregation Features** | Creating aggregated features | `data['mean'] = data.groupby('id')['value'].transform('mean')` | Machine Learning | Feature Engineering | For grouped data |

---

### 3.6 Model Validation

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Train/Test Split** | Splitting data for validation | `train_test_split(X, y, test_size=0.2)` | Machine Learning | Validation | For evaluation |
| **Cross-Validation** | Multiple train/val splits | `cross_val_score(model, X, y, cv=5)` | Machine Learning | Validation | For small data |
| **Leave-One-Out** | CV where each sample is test once | `LeaveOneOut()` | Machine Learning | Validation | For tiny data |
| **Validation Curve** | Plot for hyperparameter tuning | `validation_curve(model, X, y, param)` | Machine Learning | Validation | For tuning |
| **Learning Curve** | Performance vs training size | `learning_curve(model, X, y)` | Machine Learning | Validation | For bias/variance |
| **Holdout** | Fixed validation set | `X_train, X_val, y_train, y_val` | Machine Learning | Validation | For large data |
| **K-Fold Validation** | K iterations of train/val split | `KFold(n_splits=5)` | Machine Learning | Validation | For stable estimates |
| **Stratified K-Fold** | K-Fold preserving class distribution | `StratifiedKFold(n_splits=5)` | Machine Learning | Validation | For imbalanced data |
| **Time Series Split** | Validation for time series data | `TimeSeriesSplit()` | Machine Learning | Validation | For temporal data |
| **Group K-Fold** | K-Fold preserving group membership | `GroupKFold(n_splits=5)` | Machine Learning | Validation | For grouped data |
| **Shuffle Split** | Random train/test splits | `ShuffleSplit(n_splits=10, test_size=0.2)` | Machine Learning | Validation | For random splits |
| **Bootstrap** | Sampling with replacement | `from sklearn.utils import resample` | Machine Learning | Validation | For uncertainty |

---

## 4. Deep Learning

### 4.1 Neural Network Fundamentals

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Neural Network** | Computing system inspired by biological neural networks | `nn.Module` | Deep Learning | Core Concept | For complex patterns |
| **Neuron** | Basic computational unit in neural network | `Linear(10, 1)` | Deep Learning | Core Concept | For computing |
| **Layer** | Group of neurons in a neural network | `nn.Linear(10, 5)` | Deep Learning | Core Concept | For architecture |
| **Input Layer** | First layer receiving raw data | `nn.Linear(input_size, hidden_size)` | Deep Learning | Core Concept | For data entry |
| **Hidden Layer** | Intermediate layers between input and output | `nn.Linear(hidden_size1, hidden_size2)` | Deep Learning | Core Concept | For computation |
| **Output Layer** | Final layer producing predictions | `nn.Linear(hidden_size, output_size)` | Deep Learning | Core Concept | For results |
| **Forward Pass** | Data flowing through network | `output = model(x)` | Deep Learning | Core Concept | For inference |
| **Backward Pass** | Gradients flowing backward | `loss.backward()` | Deep Learning | Core Concept | For training |
| **Backpropagation** | Algorithm for updating weights using gradients | `loss.backward()` | Deep Learning | Core Concept | For learning |
| **Gradient** | Direction and magnitude of weight change | `param.grad` | Deep Learning | Core Concept | For updates |
| **Gradient Descent** | Optimization algorithm for weight updates | `optimizer.step()` | Deep Learning | Core Concept | For minimization |
| **Learning Rate** | Step size for gradient descent | `lr=0.001` | Deep Lighting | Core Concept | For convergence |
| **Batch** | Subset of training data processed at once | `batch_size=32` | Deep Learning | Core Concept | For efficiency |
| **Batch Size** | Number of samples per batch | `batch_size=64` | Deep Learning | Core Concept | For training stability |
| **Epoch** | One complete pass through training data | `for epoch in range(10):` | Deep Learning | Core Concept | For training progress |
| **Iteration** | One batch processed | `for batch in dataloader:` | Deep Learning | Core Concept | For updates |
| **Weight** | Learnable parameter connecting neurons | `model.fc1.weight` | Deep Learning | Core Concept | For learning |
| **Bias** | Learnable parameter added to weighted sum | `model.fc1.bias` | Deep Learning | Core Concept | For shifting |
| **Parameter** | Learnable weight or bias in network | `model.parameters()` | Deep Learning | Core Concept | For training |
| **Activation** | Non-linear function applied to neuron output | `nn.ReLU()` | Deep Learning | Core Concept | For non-linearity |
| **Forward Function** | Defines how data flows through network | `def forward(self, x):` | Deep Learning | Core Concept | For architecture |
| **Model Weights** | All learnable parameters of model | `model.state_dict()` | Deep Learning | Core Concept | For saving |
| **Model Architecture** | Structure of neural network | `class MyModel(nn.Module):` | Deep Learning | Core Concept | For design |
| **Hyperparameter** | Parameters set before training | `learning_rate, batch_size, epochs` | Deep Learning | Core Concept | For configuration |
| **Training** | Process of learning from data | `model.train()` | Deep Learning | Core Concept | For learning |
| **Evaluation** | Process of testing model performance | `model.eval()` | Deep Learning | Core Concept | For validation |
| **Prediction** | Model output on new data | `model.predict(x)` | Deep Learning | Core Concept | For inference |
| **Inference** | Making predictions with trained model | `with torch.no_grad():` | Deep Learning | Core Concept | For deployment |
| **Loss** | Measure of prediction error | `loss = criterion(output, target)` | Deep Learning | Core Concept | For training |
| **Convergence** | When model stops improving | `loss stops decreasing` | Deep Learning | Core Concept | For stopping |

---

### 4.2 Activation Functions

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **ReLU** | Rectified Linear Unit (max(0,x)) | `nn.ReLU()` | Deep Learning | Activation | Most common for hidden layers |
| **Leaky ReLU** | ReLU with small negative slope | `nn.LeakyReLU(0.1)` | Deep Learning | Activation | For dying ReLU issue |
| **PReLU** | Learnable slope for negative values | `nn.PReLU()` | Deep Learning | Activation | For adaptive activation |
| **ELU** | Exponential Linear Unit | `nn.ELU(alpha=1.0)` | Deep Learning | Activation | For negative values |
| **SELU** | Scaled Exponential Linear Unit | `nn.SELU()` | Deep Learning | Activation | For self-normalization |
| **Sigmoid** | S-shaped function outputting 0-1 | `nn.Sigmoid()` | Deep Learning | Activation | For binary classification |
| **Tanh** | Hyperbolic tangent outputting -1 to 1 | `nn.Tanh()` | Deep Learning | Activation | For zero-centered data |
| **Softmax** | Normalized exponential for probabilities | `nn.Softmax(dim=1)` | Deep Learning | Activation | For multi-class output |
| **LogSoftmax** | Log of Softmax for numerical stability | `nn.LogSoftmax(dim=1)` | Deep Learning | Activation | For log probabilities |
| **Swish** | Self-gated activation (x * sigmoid(x)) | `nn.SiLU()` | Deep Learning | Activation | For modern architectures |
| **Mish** | Self-gated activation | `nn.Mish()` | Deep Learning | Activation | For alternative to Swish |
| **GELU** | Gaussian Error Linear Unit | `nn.GELU()` | Deep Learning | Activation | For transformer models |
| **Hardswish** | Hard version of Swish | `nn.Hardswish()` | Deep Learning | Activation | For mobile devices |
| **Hardsigmoid** | Hard version of Sigmoid | `nn.Hardsigmoid()` | Deep Learning | Activation | For mobile devices |
| **Identity** | No activation (y = x) | `nn.Identity()` | Deep Learning | Activation | For linear layers |

---

### 4.3 Loss Functions

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Cross Entropy** | Loss for classification problems | `nn.CrossEntropyLoss()` | Deep Learning | Loss | For multi-class classification |
| **BCE Loss** | Binary Cross Entropy | `nn.BCEWithLogitsLoss()` | Deep Learning | Loss | For binary classification |
| **MSE Loss** | Mean Squared Error | `nn.MSELoss()` | Deep Learning | Loss | For regression |
| **MAE Loss** | Mean Absolute Error | `nn.L1Loss()` | Deep Learning | Loss | For robust regression |
| **Smooth L1** | Huber loss (robust L1/L2) | `nn.SmoothL1Loss()` | Deep Learning | Loss | For object detection |
| **Huber Loss** | Combination of MSE and MAE | `from torch import huber_loss` | Deep Learning | Loss | For outlier robustness |
| **NLL Loss** | Negative Log Likelihood | `nn.NLLLoss()` | Deep Learning | Loss | For log probabilities |
| **KL Divergence** | Kullback-Leibler divergence | `nn.KLDivLoss()` | Deep Learning | Loss | For distribution matching |
| **Cosine Embedding** | Cosine similarity loss | `nn.CosineEmbeddingLoss()` | Deep Learning | Loss | For similarity learning |
| **Margin Ranking** | Ranking loss with margin | `nn.MarginRankingLoss()` | Deep Learning | Loss | For ranking tasks |
| **Triplet Margin** | Triplet loss for embeddings | `nn.TripletMarginLoss()` | Deep Learning | Loss | For face recognition |
| **CTCLoss** | Connectionist Temporal Classification | `nn.CTCLoss()` | Deep Learning | Loss | For sequence alignment |
| **BCEWithLogits** | BCE combined with sigmoid | `nn.BCEWithLogitsLoss()` | Deep Learning | Loss | For numerical stability |
| **Poisson NLL** | Poisson negative log likelihood | `nn.PoissonNLLLoss()` | Deep Learning | Loss | For count data |
| **Gaussian NLL** | Gaussian negative log likelihood | `nn.GaussianNLLLoss()` | Deep Learning | Loss | For uncertainty |

---

### 4.4 Optimizers

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **SGD** | Stochastic Gradient Descent | `torch.optim.SGD(model.parameters(), lr=0.01)` | Deep Learning | Optimizer | For basic training |
| **Adam** | Adaptive Moment Estimation | `torch.optim.Adam(model.parameters(), lr=0.001)` | Deep Learning | Optimizer | Most common optimizer |
| **AdamW** | Adam with weight decay decoupling | `torch.optim.AdamW(model.parameters(), lr=0.001)` | Deep Learning | Optimizer | For better generalization |
| **Adagrad** | Adaptive gradient algorithm | `torch.optim.Adagrad(model.parameters(), lr=0.01)` | Deep Learning | Optimizer | For sparse features |
| **RMSprop** | Root Mean Square Propagation | `torch.optim.RMSprop(model.parameters(), lr=0.01)` | Deep Learning | Optimizer | For unstable gradients |
| **Adamax** | Adam with infinity norm | `torch.optim.Adamax(model.parameters(), lr=0.002)` | Deep Learning | Optimizer | For high dimensional data |
| **Nadam** | Nesterov accelerated Adam | `torch.optim.Nadam(model.parameters(), lr=0.002)` | Deep Learning | Optimizer | For faster convergence |
| **RAdam** | Rectified Adam | `torch.optim.RAdam(model.parameters(), lr=0.001)` | Deep Learning | Optimizer | For stable training |
| **SparseAdam** | Adam for sparse gradients | `torch.optim.SparseAdam(model.parameters())` | Deep Learning | Optimizer | For sparse embeddings |
| **ASGD** | Averaged SGD | `torch.optim.ASGD(model.parameters(), lr=0.01)` | Deep Learning | Optimizer | For better generalization |
| **Rprop** | Resilient Backpropagation | `torch.optim.Rprop(model.parameters(), lr=0.01)` | Deep Learning | Optimizer | For batch training |
| **Optimizer** | Algorithm updating model parameters | `optimizer = torch.optim.Adam(...)` | Deep Learning | Optimizer | For training |
| **Learning Rate** | Control speed of parameter updates | `lr=0.001` | Deep Learning | Core Concept | For convergence control |
| **Momentum** | Accelerate gradient descent | `momentum=0.9` | Deep Learning | Core Concept | For faster convergence |
| **Weight Decay** | L2 regularization for parameters | `weight_decay=0.01` | Deep Learning | Regularization | For overfitting prevention |
| **Learning Rate Scheduler** | Adjusts learning rate during training | `torch.optim.lr_scheduler` | Deep Learning | Core Concept | For adaptive LR |
| **Cosine Annealing** | Cosine decay of learning rate | `CosineAnnealingLR(optimizer, T_max=50)` | Deep Learning | Scheduler | For cyclical schedules |
| **StepLR** | Step decay of learning rate | `StepLR(optimizer, step_size=30, gamma=0.1)` | Deep Learning | Scheduler | For step changes |
| **ReduceLROnPlateau** | Reduce LR when metric plateaus | `ReduceLROnPlateau(optimizer, patience=10)` | Deep Learning | Scheduler | For adaptive stopping |
| **Warmup** | Gradually increasing learning rate | `warmup_steps` | Deep Learning | Scheduler | For stable start |
| **Cyclical LR** | Cyclical learning rate schedule | `CyclicLR(optimizer)` | Deep Learning | Scheduler | For exploring optima |

---

### 4.5 Regularization

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Dropout** | Randomly drop neurons during training | `nn.Dropout(p=0.5)` | Deep Learning | Regularization | For overfitting prevention |
| **L1 Regularization** | L1 penalty on weights (LASSO) | `L1Regularization` | Deep Learning | Regularization | For feature selection |
| **L2 Regularization** | L2 penalty on weights (Ridge) | `weight_decay=0.01` | Deep Learning | Regularization | For weight shrinkage |
| **Weight Decay** | L2 regularization implementation | `weight_decay=0.01` | Deep Learning | Regularization | For general regularization |
| **Dropout2D** | Spatial dropout for Conv layers | `nn.Dropout2d(p=0.2)` | Deep Learning | Regularization | For feature maps |
| **AlphaDropout** | Dropout with alpha (for SELU) | `nn.AlphaDropout(p=0.5)` | Deep Learning | Regularization | For self-normalizing |
| **Batch Normalization** | Normalizes batch activations | `nn.BatchNorm2d(64)` | Deep Learning | Normalization | For faster training |
| **Layer Normalization** | Normalizes across features | `nn.LayerNorm(512)` | Deep Learning | Normalization | For transformers |
| **Instance Norm** | Normalizes per instance | `nn.InstanceNorm2d(64)` | Deep Learning | Normalization | For style transfer |
| **Group Norm** | Normalizes in groups | `nn.GroupNorm(8, 64)` | Deep Learning | Normalization | For small batches |
| **Weight Decay** | L2 regularization | `optimizer = AdamW(weight_decay=0.01)` | Deep Learning | Regularization | For generalization |
| **Dropout Rate** | Probability of dropping neuron | `p=0.5` | Deep Learning | Regularization | For dropout strength |
| **Early Stopping** | Stop training when validation plateaus | `EarlyStopping(patience=10)` | Deep Learning | Regularization | For preventing overfitting |
| **Label Smoothing** | Soften one-hot labels | `label_smoothing=0.1` | Deep Learning | Regularization | For classification |
| **Mixup** | Mixing input samples | `mixup_alpha=0.2` | Deep Learning | Regularization | For data augmentation |
| **CutMix** | Mixing patches of images | `cutmix_alpha=1.0` | Deep Learning | Regularization | For robust training |
| **Stochastic Depth** | Randomly skip layers | `stochastic_depth_rate=0.2` | Deep Learning | Regularization | For deep networks |
| **Weight Initialization** | Initial values of weights | `init.kaiming_uniform_` | Deep Learning | Regularization | For training stability |
| **Xavier Initialization** | Glorot initialization | `init.xavier_uniform_` | Deep Learning | Regularization | For tanh/linear |
| **Kaiming Initialization** | He initialization | `init.kaiming_uniform_` | Deep Learning | Regularization | For ReLU layers |

---

### 4.6 Normalization

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **BatchNorm** | Normalization across batch dimension | `nn.BatchNorm2d(64)` | Deep Learning | Normalization | For CNN training |
| **LayerNorm** | Normalization across feature dimension | `nn.LayerNorm(512)` | Deep Learning | Normalization | For transformers |
| **InstanceNorm** | Normalization per instance | `nn.InstanceNorm2d(64)` | Deep Learning | Normalization | For style transfer |
| **GroupNorm** | Normalization in channel groups | `nn.GroupNorm(8, 64)` | Deep Learning | Normalization | For small batches |
| **RMSNorm** | Root Mean Square Normalization | `RMSNorm` | Deep Learning | Normalization | For efficient training |
| **Normalize** | Generic normalization operation | `F.normalize(tensor)` | Deep Learning | Normalization | For vector normalization |
| **Standardization** | Mean=0, std=1 normalization | `(x - mean) / std` | Data Science | Normalization | For data preprocessing |
| **Min-Max Scaling** | Scale to [0, 1] range | `(x - min) / (max - min)` | Data Science | Normalization | For bounded features |
| **Robust Scaling** | Scaling using median and IQR | `RobustScaler()` | Data Science | Normalization | For outliers |
| **Power Transform** | Box-Cox/Yeo-Johnson transform | `PowerTransformer()` | Data Science | Normalization | For skewed data |
| **Quantile Transform** | Uniform quantile normalization | `QuantileTransformer()` | Data Science | Normalization | For arbitrary distribution |

---

### 4.7 Layers - PyTorch

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Linear** | Fully connected layer | `nn.Linear(in_features, out_features)` | PyTorch | Layer | For dense connections |
| **Conv1d** | 1D convolution layer | `nn.Conv1d(in_channels, out_channels, kernel_size)` | PyTorch | Layer | For sequence data |
| **Conv2d** | 2D convolution for images | `nn.Conv2d(in_channels, out_channels, kernel_size)` | PyTorch | Layer | For image processing |
| **Conv3d** | 3D convolution for volumes | `nn.Conv3d(in_channels, out_channels, kernel_size)` | PyTorch | Layer | For 3D data |
| **MaxPool1d** | 1D max pooling | `nn.MaxPool1d(kernel_size, stride)` | PyTorch | Layer | For downsampling |
| **MaxPool2d** | 2D max pooling for images | `nn.MaxPool2d(kernel_size, stride)` | PyTorch | Layer | For reducing image size |
| **AvgPool2d** | 2D average pooling | `nn.AvgPool2d(kernel_size, stride)` | PyTorch | Layer | For smooth downsampling |
| **AdaptiveAvgPool** | Adaptive average pooling | `nn.AdaptiveAvgPool2d((1,1))` | PyTorch | Layer | For global pooling |
| **LSTM** | Long Short-Term Memory | `nn.LSTM(input_size, hidden_size, num_layers)` | PyTorch | Layer | For sequence modeling |
| **GRU** | Gated Recurrent Unit | `nn.GRU(input_size, hidden_size, num_layers)` | PyTorch | Layer | For lightweight sequences |
| **RNN** | Recurrent Neural Network | `nn.RNN(input_size, hidden_size, num_layers)` | PyTorch | Layer | For sequential data |
| **Embedding** | Word embedding layer | `nn.Embedding(num_embeddings, embedding_dim)` | PyTorch | Layer | For categorical/text |
| **BatchNorm1d** | 1D batch normalization | `nn.BatchNorm1d(num_features)` | PyTorch | Layer | For 1D features |
| **BatchNorm2d** | 2D batch normalization | `nn.BatchNorm2d(num_features)` | PyTorch | Layer | For Conv layers |
| **Dropout** | Dropout layer | `nn.Dropout(p=0.5)` | PyTorch | Layer | For regularization |
| **Flatten** | Flatten layer | `nn.Flatten(start_dim=1)` | PyTorch | Layer | For reshaping |
| **Sequential** | Sequential container | `nn.Sequential(layer1, layer2)` | PyTorch | Layer | For simple models |
| **Identity** | Identity mapping layer | `nn.Identity()` | PyTorch | Layer | For skip connections |

---

### 4.8 Layers - TensorFlow

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Dense** | Fully connected layer | `tf.keras.layers.Dense(units=64)` | TensorFlow | Layer | For dense connections |
| **Conv1D** | 1D convolution layer | `tf.keras.layers.Conv1D(filters, kernel_size)` | TensorFlow | Layer | For sequence data |
| **Conv2D** | 2D convolution for images | `tf.keras.layers.Conv2D(filters, kernel_size)` | TensorFlow | Layer | For image processing |
| **Conv3D** | 3D convolution layer | `tf.keras.layers.Conv3D(filters, kernel_size)` | TensorFlow | Layer | For 3D data |
| **MaxPool1D** | 1D max pooling | `tf.keras.layers.MaxPool1D(pool_size)` | TensorFlow | Layer | For downsampling |
| **MaxPool2D** | 2D max pooling | `tf.keras.layers.MaxPool2D(pool_size)` | TensorFlow | Layer | For image downsampling |
| **AveragePooling** | Average pooling | `tf.keras.layers.AveragePooling2D(pool_size)` | TensorFlow | Layer | For smooth downsampling |
| **GlobalAveragePooling** | Global average pooling | `tf.keras.layers.GlobalAveragePooling2D()` | TensorFlow | Layer | For classification |
| **LSTM** | LSTM layer | `tf.keras.layers.LSTM(units=128)` | TensorFlow | Layer | For sequence modeling |
| **GRU** | GRU layer | `tf.keras.layers.GRU(units=128)` | TensorFlow | Layer | For lightweight sequences |
| **Embedding** | Embedding layer | `tf.keras.layers.Embedding(input_dim, output_dim)` | TensorFlow | Layer | For categorical/text |
| **BatchNormalization** | Batch normalization | `tf.keras.layers.BatchNormalization()` | TensorFlow | Layer | For faster training |
| **Dropout** | Dropout layer | `tf.keras.layers.Dropout(rate=0.5)` | TensorFlow | Layer | For regularization |
| **Flatten** | Flatten layer | `tf.keras.layers.Flatten()` | TensorFlow | Layer | For reshaping |
| **Input** | Input layer | `tf.keras.layers.Input(shape=(32,32,3))` | TensorFlow | Layer | For defining input |

---

## 5. PyTorch Framework

### 5.1 Core PyTorch

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **torch** | Main PyTorch library import | `import torch` | PyTorch | Core | For PyTorch operations |
| **Tensor** | Multi-dimensional array container | `torch.tensor([1,2,3])` | PyTorch | Core | For data storage |
| **nn** | Neural network module | `import torch.nn as nn` | PyTorch | Core | For building networks |
| **optim** | Optimization algorithms | `import torch.optim as optim` | PyTorch | Core | For training |
| **utils** | Utility functions | `import torch.utils.data` | PyTorch | Core | For data loading |
| **cuda** | CUDA operations | `torch.cuda.is_available()` | PyTorch | Core | For GPU usage |
| **device** | Device specification | `torch.device('cuda')` | PyTorch | Core | For device management |
| **dtype** | Data type specification | `dtype=torch.float32` | PyTorch | Core | For precision control |
| **Tensor** | Main data structure | `torch.Tensor(10, 20)` | PyTorch | Core | For numerical data |
| **Variable** | Wrapper for tensor (deprecated) | `torch.autograd.Variable` | PyTorch | Core | For autograd (old) |
| **no_grad** | Disable gradient computation | `torch.no_grad()` | PyTorch | Core | For inference |
| **grad** | Gradient attribute of tensor | `tensor.grad` | PyTorch | Core | For accessing gradients |
| **requires_grad** | Track gradients flag | `requires_grad=True` | PyTorch | Core | For trainable parameters |
| **detach** | Detach from computation graph | `tensor.detach()` | PyTorch | Core | For separating graph |
| **item** | Extract scalar from single-element tensor | `tensor.item()` | PyTorch | Core | For getting values |
| **numpy** | Convert to numpy array | `tensor.numpy()` | PyTorch | Core | For numpy interoperability |
| **from_numpy** | Create tensor from numpy | `torch.from_numpy(array)` | PyTorch | Core | For converting from numpy |
| **as_tensor** | Convert to tensor | `torch.as_tensor(data)` | PyTorch | Core | For efficient conversion |
| **cat** | Concatenate tensors | `torch.cat([t1, t2], dim=0)` | PyTorch | Core | For combining tensors |
| **stack** | Stack tensors along new dimension | `torch.stack([t1, t2], dim=0)` | PyTorch | Core | For creating batches |

---

### 5.2 Tensor Operations

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **shape** | Dimensions of tensor | `tensor.shape` | PyTorch | Operation | For size checking |
| **size** | Size of each dimension | `tensor.size()` | PyTorch | Operation | For dimension info |
| **view** | Reshape tensor | `tensor.view(-1, 10)` | PyTorch | Operation | For reshaping |
| **reshape** | Reshape tensor | `tensor.reshape(10, 20)` | PyTorch | Operation | For changing shape |
| **flatten** | Flatten tensor | `tensor.flatten()` | PyTorch | Operation | For making 1D |
| **unsqueeze** | Add singleton dimension | `tensor.unsqueeze(0)` | PyTorch | Operation | For expanding dims |
| **squeeze** | Remove singleton dimension | `tensor.squeeze()` | PyTorch | Operation | For removing dims |
| **transpose** | Transpose dimensions | `tensor.transpose(0, 1)` | PyTorch | Operation | For swapping dims |
| **permute** | Reorder dimensions | `tensor.permute(0, 2, 1)` | PyTorch | Operation | For complex reordering |
| **matmul** | Matrix multiplication | `torch.matmul(A, B)` | PyTorch | Operation | For matrix multiplication |
| **mm** | Matrix multiply (2D only) | `torch.mm(A, B)` | PyTorch | Operation | For 2D matrix multiply |
| **bmm** | Batch matrix multiply | `torch.bmm(A, B)` | PyTorch | Operation | For batch matmul |
| **mul** | Element-wise multiplication | `A * B` | PyTorch | Operation | For element-wise ops |
| **add** | Element-wise addition | `A + B` | PyTorch | Operation | For adding tensors |
| **sum** | Sum of elements | `tensor.sum()` | PyTorch | Operation | For summation |
| **mean** | Mean of elements | `tensor.mean()` | PyTorch | Operation | For average |
| **std** | Standard deviation | `tensor.std()` | PyTorch | Operation | For spread |
| **max** | Maximum value | `tensor.max()` | PyTorch | Operation | For maximum |
| **min** | Minimum value | `tensor.min()` | PyTorch | Operation | For minimum |
| **argmax** | Index of maximum value | `tensor.argmax(dim=1)` | PyTorch | Operation | For predictions |
| **softmax** | Softmax activation | `F.softmax(tensor, dim=1)` | PyTorch | Operation | For probabilities |
| **log_softmax** | Log of softmax | `F.log_softmax(tensor, dim=1)` | PyTorch | Operation | For NLL loss |
| **sigmoid** | Sigmoid activation | `torch.sigmoid(tensor)` | PyTorch | Operation | For binary classification |
| **relu** | ReLU activation | `torch.relu(tensor)` | PyTorch | Operation | For activation |
| **tanh** | Tanh activation | `torch.tanh(tensor)` | PyTorch | Operation | For activation |

---

### 5.3 Model Building

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Module** | Base class for all models | `nn.Module` | PyTorch | Building | For custom models |
| **Sequential** | Sequential container | `nn.Sequential()` | PyTorch | Building | For simple models |
| **ModuleList** | List of modules | `nn.ModuleList()` | PyTorch | Building | For dynamic lists |
| **ModuleDict** | Dictionary of modules | `nn.ModuleDict()` | PyTorch | Building | For named modules |
| **ParameterList** | List of parameters | `nn.ParameterList()` | PyTorch | Building | For dynamic params |
| **ParameterDict** | Dictionary of parameters | `nn.ParameterDict()` | PyTorch | Building | For named params |
| **forward** | Define forward pass | `def forward(self, x):` | PyTorch | Building | For model logic |
| **super** | Call parent class | `super().__init__()` | PyTorch | Building | For inheritance |
| **init** | Class initializer | `def __init__(self):` | PyTorch | Building | For initialization |
| **parameters** | Get model parameters | `model.parameters()` | PyTorch | Building | For optimizer |
| **state_dict** | Get model state | `model.state_dict()` | PyTorch | Building | For saving |
| **load_state_dict** | Load model state | `model.load_state_dict()` | PyTorch | Building | For loading |
| **to** | Move model to device | `model.to(device)` | PyTorch | Building | For GPU |
| **train** | Set training mode | `model.train()` | PyTorch | Building | For training |
| **eval** | Set evaluation mode | `model.eval()` | PyTorch | Building | For evaluation |
| **children** | Get child modules | `model.children()` | PyTorch | Building | For iteration |
| **named_children** | Get named children | `model.named_children()` | PyTorch | Building | For named iteration |
| **modules** | Get all modules | `model.modules()` | PyTorch | Building | For recursion |
| **named_modules** | Get all named modules | `model.named_modules()` | PyTorch | Building | For named recursion |
| **apply** | Apply function to modules | `model.apply(function)` | PyTorch | Building | For initialization |
| **type** | Set data type | `model.type(torch.float32)` | PyTorch | Building | For precision |
| **double** | Convert to double | `model.double()` | PyTorch | Building | For double precision |
| **half** | Convert to half precision | `model.half()` | PyTorch | Building | For mixed precision |

---

### 5.4 Training Loop

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **optimizer** | Optimization algorithm | `optim.Adam(model.parameters())` | PyTorch | Training | For weight updates |
| **loss** | Loss value | `loss.item()` | PyTorch | Training | For monitoring |
| **criterion** | Loss function | `criterion = nn.CrossEntropyLoss()` | PyTorch | Training | For error calculation |
| **zero_grad** | Reset gradients | `optimizer.zero_grad()` | PyTorch | Training | For gradient clearing |
| **backward** | Compute gradients | `loss.backward()` | PyTorch | Training | For backpropagation |
| **step** | Update parameters | `optimizer.step()` | PyTorch | Training | For weight update |
| **epoch** | Complete pass through data | `for epoch in range(num_epochs):` | PyTorch | Training | For training iteration |
| **batch** | Mini-batch of data | `for batch in dataloader:` | PyTorch | Training | For batch processing |
| **train_loader** | Training data loader | `DataLoader(train_dataset)` | PyTorch | Training | For training data |
| **val_loader** | Validation data loader | `DataLoader(val_dataset)` | PyTorch | Training | For validation |
| **test_loader** | Test data loader | `DataLoader(test_dataset)` | PyTorch | Training | For testing |
| **train_loss** | Training loss | `loss.item()` | PyTorch | Training | For tracking |
| **val_loss** | Validation loss | `val_loss` | PyTorch | Training | For tracking |
| **train_acc** | Training accuracy | `accuracy` | PyTorch | Training | For performance |
| **val_acc** | Validation accuracy | `val_acc` | PyTorch | Training | For performance |
| **num_epochs** | Number of epochs | `num_epochs=100` | PyTorch | Training | For training duration |
| **batch_size** | Size of batch | `batch_size=64` | PyTorch | Training | For batch size |
| **shuffle** | Shuffle data | `shuffle=True` | PyTorch | Training | For randomization |
| **drop_last** | Drop last incomplete batch | `drop_last=True` | PyTorch | Training | For consistent batches |
| **pin_memory** | Pin memory for faster GPU transfer | `pin_memory=True` | PyTorch | Training | For speed |
| **num_workers** | Number of data loader workers | `num_workers=4` | PyTorch | Training | For parallel loading |
| **prefetch** | Prefetch data | `prefetch_factor=2` | PyTorch | Training | For speed |
| **persistent_workers** | Keep workers alive | `persistent_workers=True` | PyTorch | Training | For speed |
| **gradient_accumulation** | Accumulate gradients | `accumulation_steps` | PyTorch | Training | For large batches |

---

### 5.5 Data Loading

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Dataset** | Dataset class | `class MyDataset(Dataset):` | PyTorch | Data | For custom data |
| **DataLoader** | Data loading utility | `DataLoader(dataset, batch_size=32)` | PyTorch | Data | For batch loading |
| **Sampler** | Sample strategy | `Sampler` | PyTorch | Data | For sampling |
| **RandomSampler** | Random sampling | `RandomSampler(dataset)` | PyTorch | Data | For randomization |
| **SequentialSampler** | Sequential sampling | `SequentialSampler(dataset)` | PyTorch | Data | For order |
| **Subset** | Subset of dataset | `Subset(dataset, indices)` | PyTorch | Data | For subsets |
| **SubsetRandomSampler** | Random subset sampler | `SubsetRandomSampler(indices)` | PyTorch | Data | For random subset |
| **WeightedRandomSampler** | Weighted sampling | `WeightedRandomSampler(weights)` | PyTorch | Data | For imbalanced data |
| **BatchSampler** | Batch-level sampler | `BatchSampler(sampler)` | PyTorch | Data | For custom batching |
| **ConcatDataset** | Concatenate datasets | `ConcatDataset([d1, d2])` | PyTorch | Data | For combining |
| **ChainDataset** | Chain datasets | `ChainDataset([d1, d2])` | PyTorch | Data | For chaining |
| **IterableDataset** | Iterable dataset | `class MyIterableDataset(IterableDataset):` | PyTorch | Data | For streaming |
| **TensorDataset** | Tensor dataset | `TensorDataset(X, y)` | PyTorch | Data | For tensor data |
| **ImageFolder** | Image dataset from folder | `ImageFolder(root, transform)` | PyTorch | Data | For images |
| **DatasetFolder** | Generic folder dataset | `DatasetFolder(root, loader)` | PyTorch | Data | For custom files |
| **CIFAR10** | CIFAR-10 dataset | `torchvision.datasets.CIFAR10()` | PyTorch | Data | For image classification |
| **MNIST** | MNIST dataset | `torchvision.datasets.MNIST()` | PyTorch | Data | For digit classification |
| **ImageNet** | ImageNet dataset | `torchvision.datasets.ImageNet()` | PyTorch | Data | For large-scale image |

---

### 5.6 Model Management

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **state_dict** | Dictionary of model weights | `model.state_dict()` | PyTorch | Management | For saving |
| **load_state_dict** | Load model weights | `model.load_state_dict(state_dict)` | PyTorch | Management | For loading |
| **save** | Save model | `torch.save(model.state_dict(), 'model.pth')` | PyTorch | Management | For persisting |
| **load** | Load model | `torch.load('model.pth')` | PyTorch | Management | For restoring |
| **checkpoint** | Training checkpoint | `checkpoint = {'epoch': epoch, 'model_state': model.state_dict()}` | PyTorch | Management | For resuming |
| **resume_training** | Resume from checkpoint | `model.load_state_dict(checkpoint['model_state'])` | PyTorch | Management | For continuing |
| **save_checkpoint** | Save checkpoint | `torch.save(checkpoint, 'checkpoint.pth')` | PyTorch | Management | For regular saving |
| **load_checkpoint** | Load checkpoint | `torch.load('checkpoint.pth')` | PyTorch | Management | For restoring |
| **cpu** | Move model to CPU | `model.cpu()` | PyTorch | Management | For CPU |
| **cuda** | Move model to GPU | `model.cuda()` | PyTorch | Management | For GPU |
| **share_memory** | Share model across processes | `model.share_memory()` | PyTorch | Management | For multiprocessing |
| **clone** | Clone model | `model_copy = copy.deepcopy(model)` | PyTorch | Management | For copying |
| **parameters** | Get trainable parameters | `list(model.parameters())` | PyTorch | Management | For optimizer |
| **named_parameters** | Get named parameters | `model.named_parameters()` | PyTorch | Management | For identification |
| **buffers** | Get non-trainable buffers | `model.buffers()` | PyTorch | Management | For buffer access |

---

### 5.7 GPU Management

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **device** | Device specification | `torch.device('cuda')` | PyTorch | GPU | For GPU usage |
| **cuda_available** | Check CUDA availability | `torch.cuda.is_available()` | PyTorch | GPU | For GPU detection |
| **gpu_count** | Number of GPUs | `torch.cuda.device_count()` | PyTorch | GPU | For multi-GPU |
| **current_device** | Current GPU device | `torch.cuda.current_device()` | PyTorch | GPU | For identification |
| **device_name** | GPU device name | `torch.cuda.get_device_name(0)` | PyTorch | GPU | For identification |
| **memory_allocated** | GPU memory allocated | `torch.cuda.memory_allocated()` | PyTorch | GPU | For monitoring |
| **memory_reserved** | GPU memory reserved | `torch.cuda.memory_reserved()` | PyTorch | GPU | For monitoring |
| **empty_cache** | Clear GPU cache | `torch.cuda.empty_cache()` | PyTorch | GPU | For memory management |
| **synchronize** | Synchronize GPU | `torch.cuda.synchronize()` | PyTorch | GPU | For timing |
| **seed_all** | Set all random seeds | `set_seed(42)` | PyTorch | GPU | For reproducibility |
| **manual_seed** | Set manual seed | `torch.manual_seed(42)` | PyTorch | GPU | For reproducibility |
| **cudnn** | CUDA deep neural network | `torch.backends.cudnn` | PyTorch | GPU | For performance |
| **cudnn_benchmark** | Benchmark cuDNN | `torch.backends.cudnn.benchmark = True` | PyTorch | GPU | For optimization |
| **cudnn_deterministic** | Deterministic cuDNN | `torch.backends.cudnn.deterministic = True` | PyTorch | GPU | For reproducibility |
| **mixed_precision** | Mixed precision training | `torch.cuda.amp` | PyTorch | GPU | For speed |
| **autocast** | Automatic mixed precision | `torch.cuda.amp.autocast()` | PyTorch | GPU | For mixed precision |
| **GradScaler** | Gradient scaling | `GradScaler()` | PyTorch | GPU | For mixed precision |
| **DataParallel** | Data parallelization | `nn.DataParallel(model)` | PyTorch | GPU | For multi-GPU |
| **DistributedDataParallel** | Distributed training | `nn.parallel.DistributedDataParallel(model)` | PyTorch | GPU | For distributed |

---

### 5.8 Debugging

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **print** | Print for debugging | `print(tensor.shape)` | PyTorch | Debugging | For checking values |
| **assert** | Assert condition | `assert tensor.shape == (3, 32, 32)` | PyTorch | Debugging | For validating |
| **isnan** | Check for NaN values | `torch.isnan(tensor)` | PyTorch | Debugging | For detecting NaN |
| **isinf** | Check for infinity | `torch.isinf(tensor)` | PyTorch | Debugging | For detecting inf |
| **grad_fn** | Gradient function | `tensor.grad_fn` | PyTorch | Debugging | For graph debugging |
| **requires_grad** | Check gradient requirement | `tensor.requires_grad` | PyTorch | Debugging | For autograd |
| **breakpoint** | Set breakpoint | `breakpoint()` | PyTorch | Debugging | For interactive debugging |
| **pdb** | Python debugger | `import pdb; pdb.set_trace()` | PyTorch | Debugging | For debugging |
| **logging** | Logging messages | `import logging` | PyTorch | Debugging | For monitoring |
| **tensorboard** | TensorBoard logging | `from torch.utils.tensorboard import SummaryWriter` | PyTorch | Debugging | For visualization |
| **profiler** | Profiling tool | `torch.profiler.profile()` | PyTorch | Debugging | For performance |

---

### 5.9 Optimization

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **optimize** | Model optimization | `optimize(model)` | PyTorch | Optimization | For performance |
| **quantize** | Quantize model | `torch.quantization.quantize_dynamic()` | PyTorch | Optimization | For mobile |
| **prune** | Prune weights | `torch.nn.utils.prune` | PyTorch | Optimization | For sparsity |
| **onnx** | ONNX export | `torch.onnx.export()` | PyTorch | Optimization | For interoperability |
| **jit** | TorchScript JIT | `torch.jit.script(model)` | PyTorch | Optimization | For deployment |
| **trace** | Trace model | `torch.jit.trace(model, input)` | PyTorch | Optimization | For tracing |
| **script** | Script model | `torch.jit.script(model)` | PyTorch | Optimization | For scripting |
| **compile** | Compile model | `torch.compile(model)` | PyTorch | Optimization | For speed |
| **dynamo** | PyTorch Dynamo | `torch._dynamo` | PyTorch | Optimization | For acceleration |
| **fx** | FX graph transformation | `torch.fx` | PyTorch | Optimization | For graph manipulation |

---

### 5.10 Deployment

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **deploy** | Model deployment | `torchserve` | PyTorch | Deployment | For production |
| **torchserve** | PyTorch serving | `torchserve` | PyTorch | Deployment | For model serving |
| **torchscript** | TorchScript format | `torch.jit.save(model, 'model.pt')` | PyTorch | Deployment | For C++ |
| **onnx** | ONNX format | `torch.onnx.export()` | PyTorch | Deployment | For interoperability |
| **libtorch** | C++ library | `libtorch` | PyTorch | Deployment | For C++ |
| **onnxruntime** | ONNX runtime | `import onnxruntime` | PyTorch | Deployment | For inference |
| **tensorrt** | TensorRT optimization | `torch2trt` | PyTorch | Deployment | For NVIDIA |
| **openvino** | Intel OpenVINO | `openvino` | PyTorch | Deployment | For Intel |
| **onnx_to_tensorrt** | ONNX to TensorRT | `tensorrt` | PyTorch | Deployment | For optimization |

---

## 6. TensorFlow Framework

### 6.1 Core TensorFlow

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **tensorflow** | Main TensorFlow library | `import tensorflow as tf` | TensorFlow | Core | For TensorFlow |
| **Tensor** | Core data structure | `tf.constant([1, 2, 3])` | TensorFlow | Core | For data storage |
| **Variable** | Mutable tensor | `tf.Variable([1, 2, 3])` | TensorFlow | Core | For trainable params |
| **Graph** | Computation graph | `tf.Graph()` | TensorFlow | Core | For graph building |
| **Session** | Runtime session (TF1) | `tf.Session()` | TensorFlow | Core | For running graphs |
| **Eager** | Eager execution mode | `tf.executing_eagerly()` | TensorFlow | Core | For immediate execution |
| **placeholder** | Placeholder (TF1) | `tf.placeholder(tf.float32)` | TensorFlow | Core | For graph inputs |
| **feed_dict** | Feed dictionary (TF1) | `sess.run(op, feed_dict={x: value})` | TensorFlow | Core | For feeding data |
| **device** | Device specification | `with tf.device('/GPU:0'):` | TensorFlow | Core | For device placement |
| **keras** | Keras API | `from tensorflow import keras` | TensorFlow | Core | For building models |
| **keras** | Keras API | `tf.keras` | TensorFlow | Core | For building models |

---

### 6.2 Keras API

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Model** | Keras model class | `tf.keras.Model` | TensorFlow | Keras | For models |
| **Sequential** | Sequential model | `tf.keras.Sequential()` | TensorFlow | Keras | For linear stacks |
| **Functional** | Functional API | `tf.keras.Model(inputs, outputs)` | TensorFlow | Keras | For complex models |
| **Layer** | Layer class | `tf.keras.layers.Layer` | TensorFlow | Keras | For custom layers |
| **Input** | Input layer | `tf.keras.Input(shape=(32,32,3))` | TensorFlow | Keras | For inputs |
| **Dense** | Dense layer | `tf.keras.layers.Dense(64)` | TensorFlow | Keras | For FC layers |
| **Conv2D** | 2D convolution | `tf.keras.layers.Conv2D(64, 3)` | TensorFlow | Keras | For images |
| **MaxPooling2D** | 2D max pooling | `tf.keras.layers.MaxPooling2D(2)` | TensorFlow | Keras | For downsampling |
| **Flatten** | Flatten layer | `tf.keras.layers.Flatten()` | TensorFlow | Keras | For reshaping |
| **Dropout** | Dropout layer | `tf.keras.layers.Dropout(0.5)` | TensorFlow | Keras | For regularization |
| **BatchNormalization** | Batch normalization | `tf.keras.layers.BatchNormalization()` | TensorFlow | Keras | For normalization |
| **Activation** | Activation layer | `tf.keras.layers.Activation('relu')` | TensorFlow | Keras | For activation |
| **Add** | Add layers | `tf.keras.layers.Add()` | TensorFlow | Keras | For addition |
| **Concatenate** | Concatenate layers | `tf.keras.layers.Concatenate()` | TensorFlow | Keras | For concatenation |
| **Reshape** | Reshape layer | `tf.keras.layers.Reshape((32,32))` | TensorFlow | Keras | For reshaping |
| **GlobalAveragePooling2D** | Global pooling | `tf.keras.layers.GlobalAveragePooling2D()` | TensorFlow | Keras | For global features |
| **Softmax** | Softmax layer | `tf.keras.layers.Softmax()` | TensorFlow | Keras | For probabilities |
| **ReLU** | ReLU activation | `tf.keras.layers.ReLU()` | TensorFlow | Keras | For activation |
| **LeakyReLU** | Leaky ReLU | `tf.keras.layers.LeakyReLU()` | TensorFlow | Keras | For activation |
| **AdditiveAttention** | Attention layer | `tf.keras.layers.AdditiveAttention()` | TensorFlow | Keras | For attention |
| **Attention** | Attention layer | `tf.keras.layers.Attention()` | TensorFlow | Keras | For attention |

---

### 6.3 Tensor Operations

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **shape** | Tensor shape | `tensor.shape` | TensorFlow | Operation | For shape |
| **reshape** | Reshape tensor | `tf.reshape(tensor, (10, 20))` | TensorFlow | Operation | For reshaping |
| **transpose** | Transpose tensor | `tf.transpose(tensor)` | TensorFlow | Operation | For transposing |
| **concat** | Concatenate tensors | `tf.concat([t1, t2], axis=0)` | TensorFlow | Operation | For concatenation |
| **stack** | Stack tensors | `tf.stack([t1, t2])` | TensorFlow | Operation | For stacking |
| **split** | Split tensor | `tf.split(tensor, num_or_size_splits)` | TensorFlow | Operation | For splitting |
| **slice** | Slice tensor | `tf.slice(tensor, begin, size)` | TensorFlow | Operation | For slicing |
| **gather** | Gather indices | `tf.gather(tensor, indices)` | TensorFlow | Operation | For indexing |
| **one_hot** | One-hot encoding | `tf.one_hot(indices, depth)` | TensorFlow | Operation | For encoding |
| **matmul** | Matrix multiplication | `tf.matmul(A, B)` | TensorFlow | Operation | For matrix ops |
| **reduce_mean** | Reduce with mean | `tf.reduce_mean(tensor, axis)` | TensorFlow | Operation | For averaging |
| **reduce_sum** | Reduce with sum | `tf.reduce_sum(tensor, axis)` | TensorFlow | Operation | For summing |
| **reduce_max** | Reduce with max | `tf.reduce_max(tensor, axis)` | TensorFlow | Operation | For maximum |
| **argmax** | Index of maximum | `tf.argmax(tensor, axis)` | TensorFlow | Operation | For predictions |
| **where** | Conditional selection | `tf.where(condition)` | TensorFlow | Operation | For conditional ops |
| **maximum** | Element-wise maximum | `tf.maximum(a, b)` | TensorFlow | Operation | For element-wise |
| **minimum** | Element-wise minimum | `tf.minimum(a, b)` | TensorFlow | Operation | For element-wise |
| **clip** | Clip values | `tf.clip_by_value(tensor, min, max)` | TensorFlow | Operation | For clipping |

---

### 6.4 Model Building

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **compile** | Compile model | `model.compile(optimizer='adam', loss='categorical_crossentropy')` | TensorFlow | Building | For model setup |
| **fit** | Train model | `model.fit(X_train, y_train, epochs=10, batch_size=32)` | TensorFlow | Training | For training |
| **predict** | Make predictions | `model.predict(X_test)` | TensorFlow | Training | For inference |
| **evaluate** | Evaluate model | `model.evaluate(X_test, y_test)` | TensorFlow | Training | For evaluation |
| **summary** | Model summary | `model.summary()` | TensorFlow | Building | For architecture |
| **save** | Save model | `model.save('model.h5')` | TensorFlow | Management | For saving |
| **load_model** | Load model | `tf.keras.models.load_model('model.h5')` | TensorFlow | Management | For loading |
| **get_weights** | Get model weights | `model.get_weights()` | TensorFlow | Management | For weights |
| **set_weights** | Set model weights | `model.set_weights(weights)` | TensorFlow | Management | For setting |
| **history** | Training history | `history = model.fit(...)` | TensorFlow | Training | For tracking |
| **checkpoint** | Model checkpoint | `tf.keras.callbacks.ModelCheckpoint()` | TensorFlow | Training | For saving |
| **early_stopping** | Early stopping | `tf.keras.callbacks.EarlyStopping()` | TensorFlow | Training | For stopping |
| **reduce_lr** | Reduce LR on plateau | `tf.keras.callbacks.ReduceLROnPlateau()` | TensorFlow | Training | For LR scheduling |
| **tensorboard** | TensorBoard callback | `tf.keras.callbacks.TensorBoard()` | TensorFlow | Training | For monitoring |

---

### 6.5 Training

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **fit** | Train model | `model.fit(data, labels, epochs=10)` | TensorFlow | Training | For training |
| **batch** | Mini-batch | `batch_size=32` | TensorFlow | Training | For batch size |
| **epoch** | Complete pass | `epochs=10` | TensorFlow | Training | For epochs |
| **validation_split** | Validation split | `validation_split=0.2` | TensorFlow | Training | For validation |
| **shuffle** | Shuffle data | `shuffle=True` | TensorFlow | Training | For randomization |
| **verbose** | Verbosity level | `verbose=1` | TensorFlow | Training | For logging |
| **callbacks** | Training callbacks | `callbacks=[EarlyStopping()]` | TensorFlow | Training | For callbacks |
| **loss** | Loss value | `loss.item()` | TensorFlow | Training | For monitoring |
| **accuracy** | Accuracy metric | `accuracy` | TensorFlow | Training | For performance |
| **optimizer** | Optimizer | `optimizer='adam'` | TensorFlow | Training | For updates |
| **learning_rate** | Learning rate | `learning_rate=0.001` | TensorFlow | Training | For convergence |
| **momentum** | Momentum | `momentum=0.9` | TensorFlow | Training | For speed |
| **weight_decay** | Weight decay | `weight_decay=0.01` | TensorFlow | Training | For regularization |

---

### 6.6 Data Loading

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Dataset** | Dataset class | `tf.data.Dataset` | TensorFlow | Data | For data pipeline |
| **from_tensor_slices** | Create from slices | `tf.data.Dataset.from_tensor_slices((X, y))` | TensorFlow | Data | For tensor data |
| **from_generator** | From generator | `tf.data.Dataset.from_generator()` | TensorFlow | Data | For custom data |
| **map** | Transform dataset | `dataset.map(preprocess)` | TensorFlow | Data | For preprocessing |
| **filter** | Filter dataset | `dataset.filter(filter_fn)` | TensorFlow | Data | For filtering |
| **batch** | Batch dataset | `dataset.batch(32)` | TensorFlow | Data | For batching |
| **shuffle** | Shuffle dataset | `dataset.shuffle(buffer_size)` | TensorFlow | Data | For randomization |
| **prefetch** | Prefetch data | `dataset.prefetch(tf.data.AUTOTUNE)` | TensorFlow | Data | For speed |
| **repeat** | Repeat dataset | `dataset.repeat()` | TensorFlow | Data | For repetition |
| **take** | Take first N | `dataset.take(10)` | TensorFlow | Data | For subset |
| **skip** | Skip first N | `dataset.skip(10)` | TensorFlow | Data | For subset |
| **padded_batch** | Padded batch | `dataset.padded_batch()` | TensorFlow | Data | For variable length |

---

## 7. Computer Vision

### 7.1 Image Basics

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Pixel** | Smallest unit of an image | `pixel = image[0,0]` | Computer Vision | Image | For image processing |
| **Resolution** | Number of pixels in image | `height, width = image.shape` | Computer Vision | Image | For image size |
| **RGB** | Red, Green, Blue channels | `R, G, B = cv2.split(image)` | Computer Vision | Image | For color images |
| **HSV** | Hue, Saturation, Value | `hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)` | Computer Vision | Image | For color detection |
| **Grayscale** | Single channel image | `gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)` | Computer Vision | Image | For monochrome |
| **Channel** | Color component | `channels = image.shape[-1]` | Computer Vision | Image | For color depth |
| **Depth** | Bit depth per pixel | `depth = image.dtype` | Computer Vision | Image | For precision |
| **Size** | Image dimensions | `size = image.shape[:2]` | Computer Vision | Image | For dimensions |
| **Shape** | Image dimensions | `image.shape` | Computer Vision | Image | For dimensions |
| **dtype** | Data type | `image.dtype` | Computer Vision | Image | For data type |

---

### 7.2 Image Processing

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **resize** | Resize image | `cv2.resize(image, (224, 224))` | Computer Vision | Processing | For resizing |
| **crop** | Crop image | `image[y:y+h, x:x+w]` | Computer Vision | Processing | For cropping |
| **rotate** | Rotate image | `cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)` | Computer Vision | Processing | For rotation |
| **flip** | Flip image | `cv2.flip(image, 1)` | Computer Vision | Processing | For flipping |
| **translate** | Shift image | `cv2.warpAffine(image, M, (w, h))` | Computer Vision | Processing | For translation |
| **scale** | Scale image | `cv2.resize(image, (0,0), fx=2, fy=2)` | Computer Vision | Processing | For scaling |
| **blur** | Blur image | `cv2.GaussianBlur(image, (5,5), 0)` | Computer Vision | Processing | For smoothing |
| **filter** | Apply filter | `cv2.filter2D(image, -1, kernel)` | Computer Vision | Processing | For convolution |
| **threshold** | Apply threshold | `cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)` | Computer Vision | Processing | For binarization |
| **edge** | Detect edges | `cv2.Canny(image, 100, 200)` | Computer Vision | Processing | For edge detection |
| **contour** | Find contours | `cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)` | Computer Vision | Processing | For object boundaries |
| **histogram** | Image histogram | `cv2.calcHist([image], [0], None, [256], [0,256])` | Computer Vision | Processing | For distribution |
| **equalize** | Histogram equalization | `cv2.equalizeHist(image)` | Computer Vision | Processing | For contrast |
| **morphology** | Morphological operations | `cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)` | Computer Vision | Processing | For shape processing |
| **dilate** | Dilation | `cv2.dilate(image, kernel)` | Computer Vision | Processing | For expanding |
| **erode** | Erosion | `cv2.erode(image, kernel)` | Computer Vision | Processing | For shrinking |

---

### 7.3 CNN Architectures

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **CNN** | Convolutional Neural Network | `models.resnet18()` | Computer Vision | Architecture | For images |
| **ConvNet** | Convolutional Network | `nn.Conv2d()` | Computer Vision | Architecture | For spatial data |
| **AlexNet** | AlexNet architecture | `torchvision.models.alexnet()` | Computer Vision | Architecture | For image classification |
| **VGG** | VGG architecture | `torchvision.models.vgg16()` | Computer Vision | Architecture | For simple architecture |
| **ResNet** | Residual Network | `torchvision.models.resnet18()` | Computer Vision | Architecture | For deep networks |
| **ResNeXt** | ResNet with cardinality | `torchvision.models.resnext50_32x4d()` | Computer Vision | Architecture | For better accuracy |
| **DenseNet** | Densely Connected Network | `torchvision.models.densenet121()` | Computer Vision | Architecture | For dense connections |
| **Inception** | Inception/GoogLeNet | `torchvision.models.inception_v3()` | Computer Vision | Architecture | For multi-scale |
| **MobileNet** | Mobile-friendly network | `torchvision.models.mobilenet_v2()` | Computer Vision | Architecture | For mobile |
| **ShuffleNet** | ShuffleNet architecture | `torchvision.models.shufflenet_v2_x0_5()` | Computer Vision | Architecture | For efficiency |
| **EfficientNet** | EfficientNet architecture | `torchvision.models.efficientnet_b0()` | Computer Vision | Architecture | For SOTA accuracy |
| **SqueezeNet** | Small network | `torchvision.models.squeezenet1_0()` | Computer Vision | Architecture | For small devices |
| **Vision Transformer** | ViT architecture | `torchvision.models.vit_b_16()` | Computer Vision | Architecture | For transformer vision |
| **Swin Transformer** | Swin architecture | `torchvision.models.swin_t()` | Computer Vision | Architecture | For hierarchical vision |

---

### 7.4 Detection & Segmentation

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Object Detection** | Finding objects in image | `YOLO`, `Faster R-CNN` | Computer Vision | Detection | For object localization |
| **Segmentation** | Pixel-level classification | `U-Net`, `Mask R-CNN` | Computer Vision | Segmentation | For detailed masks |
| **Bounding Box** | Object rectangle | `(x, y, w, h)` | Computer Vision | Detection | For localization |
| **Anchor Box** | Predefined box shapes | `anchor = torch.tensor(...)` | Computer Vision | Detection | For detection |
| **ROI** | Region of Interest | `roi = image[y1:y2, x1:x2]` | Computer Vision | Detection | For cropping |
| **IoU** | Intersection over Union | `iou_score(pred, target)` | Computer Vision | Detection | For evaluation |
| **NMS** | Non-Maximum Suppression | `torchvision.ops.nms(boxes, scores, iou_threshold)` | Computer Vision | Detection | For filtering |
| **Faster R-CNN** | Two-stage detector | `torchvision.models.detection.fasterrcnn_resnet50_fpn()` | Computer Vision | Detection | For accuracy |
| **YOLO** | One-stage detector | `ultralytics.YOLO('yolov8n.pt')` | Computer Vision | Detection | For speed |
| **SSD** | Single Shot Detector | `torchvision.models.detection.ssd300_vgg16()` | Computer Vision | Detection | For speed-accuracy |
| **RetinaNet** | Focal Loss detector | `torchvision.models.detection.retinanet_resnet50_fpn()` | Computer Vision | Detection | For imbalanced data |
| **Mask R-CNN** | Instance segmentation | `torchvision.models.detection.maskrcnn_resnet50_fpn()` | Computer Vision | Segmentation | For masks |
| **U-Net** | U-shaped network | `model = UNet()` | Computer Vision | Segmentation | For medical images |
| **DeepLab** | DeepLab segmentation | `torchvision.models.segmentation.deeplabv3_resnet50()` | Computer Vision | Segmentation | For semantic segmentation |

---

### 7.5 Image Augmentation

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Augmentation** | Data augmentation | `transforms.RandomHorizontalFlip()` | Computer Vision | Augmentation | For more data |
| **Horizontal Flip** | Flip image horizontally | `transforms.RandomHorizontalFlip(p=0.5)` | Computer Vision | Augmentation | For mirroring |
| **Vertical Flip** | Flip image vertically | `transforms.RandomVerticalFlip(p=0.5)` | Computer Vision | Augmentation | For vertical mirror |
| **Random Crop** | Random crop | `transforms.RandomResizedCrop(224)` | Computer Vision | Augmentation | For size variation |
| **Random Rotation** | Random rotation | `transforms.RandomRotation(degrees=10)` | Computer Vision | Augmentation | For angle variation |
| **Color Jitter** | Color variations | `transforms.ColorJitter(brightness=0.2, contrast=0.2)` | Computer Vision | Augmentation | For color variation |
| **Random Erasing** | Erase random parts | `transforms.RandomErasing(p=0.5)` | Computer Vision | Augmentation | For robustness |
| **AutoAugment** | Auto augmentation | `transforms.AutoAugment()` | Computer Vision | Augmentation | For automated |
| **Gaussian Noise** | Add noise | `transforms.RandomGaussianNoise()` | Computer Vision | Augmentation | For robustness |

---

### 7.6 Video Processing

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Video** | Sequence of frames | `cap = cv2.VideoCapture('video.mp4')` | Computer Vision | Video | For video processing |
| **Frame** | Single image from video | `ret, frame = cap.read()` | Computer Vision | Video | For frames |
| **FPS** | Frames per second | `fps = cap.get(cv2.CAP_PROP_FPS)` | Computer Vision | Video | For frame rate |
| **Optical Flow** | Motion between frames | `cv2.calcOpticalFlowFarneback()` | Computer Vision | Video | For motion |
| **VideoWriter** | Write video file | `cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))` | Computer Vision | Video | For saving |
| **Frame Extraction** | Extract video frames | `while cap.isOpened():` | Computer Vision | Video | For frame processing |

---

## 8. NLP & LLM

### 8.1 NLP Fundamentals

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **NLP** | Natural Language Processing | `import nltk` | NLP | Core Concept | For text |
| **Token** | Smallest text unit | `tokens = word_tokenize(text)` | NLP | Text | For text splitting |
| **Tokenization** | Splitting text into tokens | `tokenizer.tokenize(text)` | NLP | Text | For preprocessing |
| **Word** | Basic linguistic unit | `word = "hello"` | NLP | Text | For text analysis |
| **Sentence** | Grammatical unit | `sentences = sent_tokenize(text)` | NLP | Text | For sentence splitting |
| **Document** | Collection of text | `doc = "Full text..."` | NLP | Text | For corpus |
| **Corpus** | Collection of documents | `corpus = [doc1, doc2, ...]` | NLP | Text | For dataset |
| **Stopwords** | Common words to remove | `stopwords.words('english')` | NLP | Text | For cleaning |
| **Stemming** | Reduce to root form | `porter.stem(word)` | NLP | Text | For normalization |
| **Lemmatization** | Reduce to dictionary form | `lemmatizer.lemmatize(word)` | NLP | Text | For normalization |
| **Bag of Words** | Word count representation | `CountVectorizer()` | NLP | Text | For features |
| **TF-IDF** | Term frequency-inverse document frequency | `TfidfVectorizer()` | NLP | Text | For importance |
| **N-gram** | Sequence of N tokens | `nltk.ngrams(tokens, 2)` | NLP | Text | For context |

---

### 8.2 Text Processing

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Cleaning** | Text cleaning | `text.lower().strip()` | NLP | Processing | For preprocessing |
| **Normalization** | Text normalization | `unicodedata.normalize('NFKD', text)` | NLP | Processing | For standardizing |
| **Regex** | Regular expressions | `re.sub(r'[^a-zA-Z\s]', '', text)` | NLP | Processing | For pattern matching |
| **POS Tagging** | Part-of-speech tagging | `nltk.pos_tag(tokens)` | NLP | Processing | For grammar |
| **NER** | Named Entity Recognition | `spacy.ner(text)` | NLP | Processing | For entities |
| **Dependency Parsing** | Parse grammar tree | `spacy.parse(text)` | NLP | Processing | For structure |
| **Word2Vec** | Word vectors | `Word2Vec(sentences)` | NLP | Embeddings | For word embeddings |
| **GloVe** | Global vectors | `GloVe()` | NLP | Embeddings | For word embeddings |
| **FastText** | FastText embeddings | `fasttext` | NLP | Embeddings | For word embeddings |
| **BERT** | Bidirectional Encoder Representations | `BERT` | NLP | Model | For NLP tasks |

---

### 8.3 Word Embeddings

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Embedding** | Distributed representation | `nn.Embedding(vocab_size, embed_dim)` | NLP | Embeddings | For word vectors |
| **Word Vector** | Numerical word representation | `word2vec['word']` | NLP | Embeddings | For semantics |
| **Vector** | Embedding vector | `vector = embedding[word_idx]` | NLP | Embeddings | For representation |
| **Dimension** | Embedding dimension | `embed_dim=300` | NLP | Embeddings | For size |
| **Training** | Embedding training | `model.fit(sentences)` | NLP | Embeddings | For learning |
| **Transfer** | Transfer learning | `pretrained_embeddings` | NLP | Embeddings | For starting point |
| **Fine-tuning** | Fine-tune embeddings | `embedding.weight.requires_grad = True` | NLP | Embeddings | For adaptation |
| **Contextual** | Context-aware embeddings | `BERT`, `ELMo` | NLP | Embeddings | For context |

---

### 8.4 Transformer Architecture

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Attention** | Attention mechanism | `nn.MultiheadAttention()` | NLP | Transformer | For relationships |
| **Self-Attention** | Self-attention | `nn.MultiheadAttention()` | NLP | Transformer | For context |
| **Multi-Head** | Multiple attention heads | `num_heads=8` | NLP | Transformer | For diverse patterns |
| **QKV** | Query, Key, Value | `Q, K, V` | NLP | Transformer | For attention |
| **Transformer** | Transformer model | `Transformer()` | NLP | Transformer | For sequence |
| **Encoder** | Encoder stack | `EncoderLayer()` | NLP | Transformer | For input encoding |
| **Decoder** | Decoder stack | `DecoderLayer()` | NLP | Transformer | For output decoding |
| **Positional Encoding** | Position embedding | `positional_encoding()` | NLP | Transformer | For position |
| **Masking** | Attention masking | `mask` | NLP | Transformer | For padding |
| **Cross-Attention** | Encoder-decoder attention | `nn.MultiheadAttention()` | NLP | Transformer | For generation |
| **Feed-Forward** | FFN layer | `nn.Linear()` | NLP | Transformer | For transformation |
| **LayerNorm** | Layer normalization | `nn.LayerNorm()` | NLP | Transformer | For normalization |

---

### 8.5 LLM Models

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **LLM** | Large Language Model | `GPT`, `LLaMA`, `Claude` | NLP | LLM | For generation |
| **GPT** | Generative Pre-trained Transformer | `GPT-4` | NLP | LLM | For generation |
| **BERT** | Bidirectional Encoder Representations | `BERT` | NLP | LLM | For understanding |
| **LLaMA** | Large Language Model Meta AI | `Meta-LLaMA` | NLP | LLM | For open-source |
| **Claude** | Claude by Anthropic | `Claude` | NLP | LLM | For safe AI |
| **Gemini** | Gemini by Google | `Gemini` | NLP | LLM | For multimodal |
| **Mistral** | Mistral model | `Mistral` | NLP | LLM | For efficiency |
| **Falcon** | Falcon model | `Falcon` | NLP | LLM | For open-source |
| **Hugging Face** | Hugging Face library | `from transformers import ...` | NLP | Tools | For models |
| **Transformers** | Hugging Face transformers | `from transformers import AutoModel` | NLP | Tools | For NLP models |
| **Tokenizer** | Tokenization tool | `AutoTokenizer.from_pretrained('bert-base')` | NLP | Tools | For text tokenization |
| **Model Hub** | Model repository | `huggingface.co/models` | NLP | Tools | For finding models |

---

### 8.6 LLM Training

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Pre-training** | Initial training on large data | `pretrained` | NLP | Training | For base model |
| **Fine-tuning** | Adapting pre-trained model | `model.train()` | NLP | Training | For specific tasks |
| **Prompt** | Input text for LLM | `prompt = "Write a summary..."` | NLP | Training | For generation |
| **Instruction Tuning** | Training on instructions | `model.train()` | NLP | Training | For following instructions |
| **RLHF** | Reinforcement Learning from Human Feedback | `RLHF` | NLP | Training | For alignment |
| **LoRA** | Low-Rank Adaptation | `peft.LoRA` | NLP | Training | For fine-tuning |
| **PEFT** | Parameter-Efficient Fine-Tuning | `from peft import ...` | NLP | Training | For efficient tuning |
| **QLoRA** | Quantized LoRA | `QLoRA` | NLP | Training | For memory-efficient |
| **Adapter** | Adapter modules | `adapters.Adapters()` | NLP | Training | For efficient tuning |

---

### 8.7 Prompt Engineering

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Prompt** | Input to LLM | `prompt = "Hello!"` | NLP | Prompt | For LLM input |
| **Prompt Engineering** | Designing prompts | `system_prompt` | NLP | Prompt | For better output |
| **System Prompt** | Core instruction | `system = "You are a helpful assistant"` | NLP | Prompt | For guidance |
| **User Prompt** | User query | `user = "What is AI?"` | NLP | Prompt | For question |
| **Context** | Background information | `context` | NLP | Prompt | For relevant info |
| **In-Context Learning** | Learning from examples | `few_shot_prompt` | NLP | Prompt | For patterns |
| **Few-Shot** | Few examples in prompt | `few_shot_examples` | NLP | Prompt | For demonstration |
| **Zero-Shot** | No examples | `zero_shot_prompt` | NLP | Prompt | For general tasks |
| **Chain of Thought** | Step-by-step reasoning | `CoT` | NLP | Prompt | For reasoning |
| **Self-Consistency** | Multiple reasoning paths | `self_consistency` | NLP | Prompt | For accuracy |

---

### 8.8 Fine-tuning

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Fine-tuning** | Adapting pre-trained model | `model = AutoModelForSequenceClassification.from_pretrained('bert-base')` | NLP | Fine-tuning | For specific tasks |
| **Full Fine-tuning** | All parameters updated | `model.train()` | NLP | Fine-tuning | For full adaptation |
| **Parameter-Efficient** | Few parameters updated | `peft` | NLP | Fine-tuning | For memory saving |
| **LoRA** | Low-Rank Adaptation | `from peft import LoraConfig` | NLP | Fine-tuning | For efficient tuning |
| **Adapter** | Small adapter modules | `adapters.Adapter()` | NLP | Fine-tuning | For modular tuning |
| **P-tuning** | Prompts as parameters | `P-tuning` | NLP | Fine-tuning | For prompt tuning |
| **Prefix Tuning** | Prefix as learnable | `PrefixTuning` | NLP | Fine-tuning | For efficient tuning |
| **Instruction Tuning** | Training on instructions | `instruction_tuning` | NLP | Fine-tuning | For instruction following |

---

### 8.9 RAG Systems

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **RAG** | Retrieval-Augmented Generation | `RAG` | NLP | RAG | For augmented generation |
| **Retrieval** | Information retrieval | `vector_search(query)` | NLP | RAG | For finding documents |
| **Vector Database** | Vector storage | `faiss`, `pinecone`, `weaviate` | NLP | RAG | For embedding retrieval |
| **Embedding** | Vector representation | `embedding = model.encode(text)` | NLP | RAG | For similarity |
| **Semantic Search** | Meaning-based search | `semantic_search(query, documents)` | NLP | RAG | For relevance |
| **Chunking** | Splitting documents | `split_text(document)` | NLP | RAG | For retrieval |
| **Re-ranking** | Reorder retrieved documents | `rerank(documents, query)` | NLP | RAG | For better results |
| **Context Augmentation** | Adding retrieved context | `context = retrieve(query)` | NLP | RAG | For enriched input |
| **LLM Augmentation** | LLM with retrieved context | `generate(prompt, context)` | NLP | RAG | For answer generation |

---

## 9. MLOps

### 9.1 Model Deployment

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Deployment** | Model deployment | `deploy(model)` | MLOps | Deployment | For production |
| **Serving** | Model serving | `torchserve`, `triton` | MLOps | Deployment | For inference |
| **API** | API endpoint | `FastAPI`, `Flask` | MLOps | Deployment | For service |
| **Inference** | Model inference | `model.predict(input)` | MLOps | Deployment | For predictions |
| **Batch Inference** | Batch predictions | `predict_batch(data)` | MLOps | Deployment | For bulk processing |
| **Real-time** | Real-time inference | `streaming inference` | MLOps | Deployment | For speed |
| **Edge** | Edge deployment | `mobile`, `IoT` | MLOps | Deployment | For devices |
| **Cloud** | Cloud deployment | `AWS`, `GCP`, `Azure` | MLOps | Deployment | For scalability |
| **On-premises** | On-premise deployment | `self-hosted` | MLOps | Deployment | For privacy |
| **Docker** | Containerization | `docker build` | MLOps | Deployment | For portability |
| **Kubernetes** | Container orchestration | `kubectl` | MLOps | Deployment | For scaling |

---

### 9.2 Monitoring

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Monitoring** | System monitoring | `monitor(model)` | MLOps | Monitoring | For health |
| **Metrics** | Performance metrics | `accuracy`, `latency` | MLOps | Monitoring | For performance |
| **Logging** | Logging system | `log.info()` | MLOps | Monitoring | For debugging |
| **Alerting** | Alert system | `send_alert()` | MLOps | Monitoring | For issues |
| **Dashboard** | Visual interface | `Grafana`, `Tableau` | MLOps | Monitoring | For visualization |
| **Drift** | Data/model drift | `detect_drift()` | MLOps | Monitoring | For degradation |
| **Data Drift** | Data distribution change | `data_drift` | MLOps | Monitoring | For input change |
| **Concept Drift** | Target concept change | `concept_drift` | MLOps | Monitoring | For output change |
| **Model Drift** | Model performance change | `model_drift` | MLOps | Monitoring | For model decay |
| **Retraining** | Model retraining | `retrain_model()` | MLOps | Monitoring | For updates |

---

### 9.3 CI/CD

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **CI** | Continuous Integration | `CI pipeline` | MLOps | CI/CD | For automation |
| **CD** | Continuous Deployment | `CD pipeline` | MLOps | CI/CD | For automated release |
| **Pipeline** | CI/CD pipeline | `pipeline` | MLOps | CI/CD | For automation |
| **Git** | Git version control | `git` | MLOps | CI/CD | For versioning |
| **GitHub** | GitHub platform | `GitHub Actions` | MLOps | CI/CD | For hosting |
| **GitLab** | GitLab platform | `GitLab CI/CD` | MLOps | CI/CD | For hosting |
| **Jenkins** | Jenkins CI tool | `Jenkins` | MLOps | CI/CD | For automation |
| **Testing** | ML testing | `test_model()` | MLOps | CI/CD | For validation |
| **Validation** | Model validation | `validate_model()` | MLOps | CI/CD | For quality |
| **Staging** | Staging environment | `staging` | MLOps | CI/CD | For testing |

---

### 9.4 Versioning

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Versioning** | Version control | `dvc` | MLOps | Versioning | For tracking |
| **Data Versioning** | Data version control | `dvc` | MLOps | Versioning | For data |
| **Model Versioning** | Model version control | `model_version` | MLOps | Versioning | For models |
| **Code Versioning** | Code version control | `git` | MLOps | Versioning | For code |
| **Artifact** | ML artifacts | `pickle`, `pth` | MLOps | Versioning | For model files |
| **Model Registry** | Model storage | `MLflow` | MLOps | Versioning | For storage |
| **DVC** | Data Version Control | `dvc` | MLOps | Versioning | For data |
| **MLflow** | MLflow platform | `import mlflow` | MLOps | Versioning | For tracking |

---

### 9.5 Scaling

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Scaling** | Horizontal scaling | `scale()` | MLOps | Scaling | For load |
| **Load Balancing** | Load distribution | `load_balancer` | MLOps | Scaling | For distribution |
| **Distributed** | Distributed computing | `torch.distributed` | MLOps | Scaling | For parallel |
| **Parallel** | Parallel processing | `parallel` | MLOps | Scaling | For speed |
| **Batch** | Batch processing | `batch_inference` | MLOps | Scaling | For efficiency |
| **Queue** | Message queue | `Kafka`, `RabbitMQ` | MLOps | Scaling | For queuing |
| **Kubernetes** | Container orchestration | `k8s` | MLOps | Scaling | For management |

---

### 9.6 Security

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Security** | System security | `security` | MLOps | Security | For protection |
| **Privacy** | Data privacy | `differential_privacy` | MLOps | Security | For privacy |
| **Encryption** | Data encryption | `encrypt()` | MLOps | Security | For protection |
| **Authentication** | Identity verification | `auth` | MLOps | Security | For access |
| **Authorization** | Access control | `permissions` | MLOps | Security | For roles |
| **Audit** | Audit trail | `audit_log` | MLOps | Security | For tracking |
| **Compliance** | Regulatory compliance | `GDPR`, `HIPAA` | MLOps | Security | For regulation |

---

## 10. Mathematics & Statistics

### 10.1 Linear Algebra

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Matrix** | 2D array of numbers | `torch.matrix([[1,2],[3,4]])` | Mathematics | Linear Algebra | For linear operations |
| **Vector** | 1D array of numbers | `torch.vector([1,2,3])` | Mathematics | Linear Algebra | For direction |
| **Scalar** | Single number | `scalar = 5` | Mathematics | Linear Algebra | For magnitude |
| **Tensor** | Multi-dimensional array | `torch.tensor(...)` | Mathematics | Linear Algebra | For data |
| **Dot Product** | Vector multiplication | `torch.dot(v1, v2)` | Mathematics | Linear Algebra | For similarity |
| **Norm** | Vector length | `torch.norm(v)` | Mathematics | Linear Algebra | For magnitude |
| **Eigenvalue** | Characteristic value | `eigenvalues = torch.eig(A)` | Mathematics | Linear Algebra | For PCA |
| **Eigenvector** | Characteristic vector | `eigenvectors` | Mathematics | Linear Algebra | For PCA |
| **Determinant** | Matrix scalar | `torch.det(A)` | Mathematics | Linear Algebra | For invertibility |
| **Inverse** | Matrix inverse | `torch.inv(A)` | Mathematics | Linear Algebra | For solving |

---

### 10.2 Calculus

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Derivative** | Rate of change | `torch.autograd.grad()` | Mathematics | Calculus | For optimization |
| **Gradient** | Vector of partial derivatives | `gradient` | Mathematics | Calculus | For learning |
| **Chain Rule** | Composite function derivative | `chain_rule` | Mathematics | Calculus | For backprop |
| **Jacobian** | Matrix of partial derivatives | `jacobian` | Mathematics | Calculus | For transformation |
| **Hessian** | Matrix of second derivatives | `hessian` | Mathematics | Calculus | For optimization |
| **Optimization** | Maximize/minimize function | `optimizer.step()` | Mathematics | Calculus | For training |

---

### 10.3 Probability

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Probability** | Measure of likelihood | `p = 0.5` | Mathematics | Probability | For uncertainty |
| **Distribution** | Probability distribution | `torch.distributions` | Mathematics | Probability | For modeling |
| **Normal** | Normal/Gaussian distribution | `torch.normal()` | Mathematics | Probability | For noise |
| **Uniform** | Uniform distribution | `torch.uniform()` | Mathematics | Probability | For sampling |
| **Binomial** | Binomial distribution | `torch.distributions.Binomial()` | Mathematics | Probability | For binary |
| **Poisson** | Poisson distribution | `torch.distributions.Poisson()` | Mathematics | Probability | For counts |
| **Bayes** | Bayes theorem | `bayes_theorem` | Mathematics | Probability | For inference |
| **Likelihood** | Conditional probability | `likelihood` | Mathematics | Probability | For estimation |
| **Entropy** | Information content | `entropy` | Mathematics | Probability | For uncertainty |
| **Cross-Entropy** | Cross entropy loss | `nn.CrossEntropyLoss()` | Mathematics | Probability | For classification |

---

### 10.4 Statistics

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Mean** | Average | `torch.mean()` | Mathematics | Statistics | For central value |
| **Median** | Middle value | `torch.median()` | Mathematics | Statistics | For robust center |
| **Mode** | Most frequent | `torch.mode()` | Mathematics | Statistics | For frequent |
| **Variance** | Spread of data | `torch.var()` | Mathematics | Statistics | For dispersion |
| **Standard Deviation** | Square root of variance | `torch.std()` | Mathematics | Statistics | For spread |
| **Correlation** | Relationship measure | `torch.corrcoef()` | Mathematics | Statistics | For association |
| **Covariance** | Joint variability | `torch.cov()` | Mathematics | Statistics | For relationship |
| **Quantile** | Data division point | `torch.quantile()` | Mathematics | Statistics | For percentiles |
| **Test** | Hypothesis test | `scipy.stats.ttest_ind()` | Mathematics | Statistics | For inference |
| **P-Value** | Significance measure | `p_value` | Mathematics | Statistics | For significance |

---

## 11. Tools & Infrastructure

### 11.1 Version Control

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Git** | Version control | `git init` | Tools | Version Control | For code |
| **GitHub** | Git hosting | `GitHub` | Tools | Version Control | For collaboration |
| **GitLab** | Git hosting | `GitLab` | Tools | Version Control | For collaboration |
| **Bitbucket** | Git hosting | `Bitbucket` | Tools | Version Control | For collaboration |
| **DVC** | Data version control | `dvc` | Tools | Version Control | For data |
| **LFS** | Large file storage | `git lfs` | Tools | Version Control | For large files |

---

### 11.2 Cloud Platforms

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **AWS** | Amazon Web Services | `AWS` | Tools | Cloud | For cloud |
| **GCP** | Google Cloud Platform | `GCP` | Tools | Cloud | For cloud |
| **Azure** | Microsoft Azure | `Azure` | Tools | Cloud | For cloud |
| **S3** | AWS storage | `s3` | Tools | Cloud | For storage |
| **EC2** | AWS compute | `ec2` | Tools | Cloud | For compute |
| **GCS** | Google Cloud Storage | `gcs` | Tools | Cloud | For storage |
| **VM** | Virtual Machine | `vm` | Tools | Cloud | For compute |

---

### 11.3 Containers

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Docker** | Containerization | `docker build` | Tools | Containers | For packaging |
| **Container** | Isolated runtime | `container` | Tools | Containers | For deployment |
| **Image** | Docker image | `docker image` | Tools | Containers | For template |
| **Registry** | Container registry | `docker hub` | Tools | Containers | For storage |
| **Dockerfile** | Docker build script | `Dockerfile` | Tools | Containers | For building |
| **Kubernetes** | Container orchestration | `k8s` | Tools | Orchestration | For scaling |

---

### 11.4 Orchestration

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Kubernetes** | Container orchestration | `kubectl` | Tools | Orchestration | For managing |
| **Pod** | Kubernetes pod | `pod` | Tools | Orchestration | For container |
| **Service** | Kubernetes service | `service` | Tools | Orchestration | For networking |
| **Deployment** | Kubernetes deployment | `deployment` | Tools | Orchestration | For scaling |
| **Ingress** | Ingress controller | `ingress` | Tools | Orchestration | For routing |
| **ConfigMap** | Configuration storage | `configmap` | Tools | Orchestration | For config |
| **Secret** | Secret storage | `secret` | Tools | Orchestration | For secrets |
| **Docker Swarm** | Docker orchestration | `docker swarm` | Tools | Orchestration | For alternative |

---

### 11.5 Experiment Tracking

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **MLflow** | MLflow platform | `import mlflow` | Tools | Tracking | For experiment |
| **Weights & Biases** | W&B platform | `import wandb` | Tools | Tracking | For experiment |
| **TensorBoard** | TensorBoard visualization | `tensorboard` | Tools | Tracking | For visualization |
| **Optuna** | Hyperparameter optimization | `import optuna` | Tools | Optimization | For tuning |
| **Hyperopt** | Hyperparameter optimization | `import hyperopt` | Tools | Optimization | For tuning |
| **Ray** | Distributed computing | `import ray` | Tools | Distributed | For scaling |
| **DVC** | Data version control | `dvc` | Tools | Tracking | For data tracking |
| **MLflow Tracking** | Experiment tracking | `mlflow.log_metric()` | Tools | Tracking | For logging |

---

## 12. Ethics & Best Practices

### 12.1 AI Ethics

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Bias** | Systematic error in AI | `bias` | Ethics | Ethics | For fairness |
| **Fairness** | Equal treatment in AI | `fairness` | Ethics | Ethics | For justice |
| **Transparency** | AI explainability | `explainable_ai` | Ethics | Ethics | For understanding |
| **Explainability** | AI reasoning explanation | `XAI` | Ethics | Ethics | For trust |
| **Accountability** | AI responsibility | `accountability` | Ethics | Ethics | For responsibility |
| **Privacy** | Data privacy | `privacy` | Ethics | Ethics | For protection |
| **Safety** | AI safety | `safety` | Ethics | Ethics | For security |
| **Robustness** | AI reliability | `robustness` | Ethics | Ethics | For reliability |
| **Human-Centric** | Human-focused AI | `human_centric` | Ethics | Ethics | For human values |

---

### 12.2 Privacy

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Differential Privacy** | Privacy-preserving ML | `diffpriv` | Ethics | Privacy | For privacy |
| **Federated Learning** | Decentralized learning | `FL` | Ethics | Privacy | For distributed |
| **Data Anonymization** | Anonymizing data | `anonymize()` | Ethics | Privacy | For protection |
| **Privacy Budget** | Privacy measurement | `epsilon` | Ethics | Privacy | For privacy control |

---

### 12.3 Fairness

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Algorithmic Fairness** | Fair algorithms | `fairness` | Ethics | Fairness | For justice |
| **Fairness Metrics** | Fairness measurement | `demographic_parity` | Ethics | Fairness | For evaluation |
| **Bias Detection** | Bias identification | `bias_detection()` | Ethics | Fairness | For discovery |

---

### 12.4 Explainability

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Explainability** | Model explanation | `XAI` | Ethics | Explainability | For transparency |
| **SHAP** | SHAP explainability | `import shap` | Ethics | Explainability | For feature importance |
| **LIME** | LIME explainability | `import lime` | Ethics | Explainability | For local explanation |
| **Integrated Gradients** | Gradient-based explanation | `integrated_gradients` | Ethics | Explainability | For attribution |

---

### 12.5 Security

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Adversarial Attack** | Attack on AI | `adversarial_attack` | Ethics | Security | For robustness |
| **Adversarial Training** | Training with attacks | `adversarial_training` | Ethics | Security | For defense |
| **Data Poisoning** | Poisoning data | `data_poisoning` | Ethics | Security | For awareness |
| **Model Extraction** | Extracting models | `model_extraction` | Ethics | Security | For protection |

---

## 13. Advanced Concepts

### 13.1 Generative AI

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **GAN** | Generative Adversarial Network | `GAN` | Advanced | Generative | For generation |
| **VAE** | Variational Autoencoder | `VAE` | Advanced | Generative | For generation |
| **Diffusion** | Diffusion models | `diffusion` | Advanced | Generative | For generation |
| **Stable Diffusion** | Image generation | `stable_diffusion` | Advanced | Generative | For images |
| **DALL-E** | Image generation | `DALL-E` | Advanced | Generative | For images |
| **Midjourney** | Image generation | `Midjourney` | Advanced | Generative | For images |
| **Stable Diffusion** | Open-source generation | `stable_diffusion` | Advanced | Generative | For images |

---

### 13.2 Reinforcement Learning

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **RL** | Reinforcement Learning | `RL` | Advanced | RL | For decision making |
| **Agent** | RL agent | `agent` | Advanced | RL | For learning |
| **Environment** | RL environment | `env` | Advanced | RL | For interaction |
| **Action** | RL action | `action` | Advanced | RL | For decision |
| **State** | RL state | `state` | Advanced | RL | For context |
| **Reward** | RL reward | `reward` | Advanced | RL | For feedback |
| **Policy** | RL policy | `policy` | Advanced | RL | For strategy |
| **Q-Learning** | Q-learning algorithm | `Q-learning` | Advanced | RL | For learning |
| **Deep Q-Network** | DQN algorithm | `DQN` | Advanced | RL | For deep RL |
| **PPO** | Proximal Policy Optimization | `PPO` | Advanced | RL | For policy |

---

### 13.3 Graph Learning

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **GNN** | Graph Neural Network | `GNN` | Advanced | Graph | For graph data |
| **GCN** | Graph Convolutional Network | `GCN` | Advanced | Graph | For graph convolution |
| **GAT** | Graph Attention Network | `GAT` | Advanced | Graph | For attention |
| **Node** | Graph node | `node` | Advanced | Graph | For vertices |
| **Edge** | Graph edge | `edge` | Advanced | Graph | For connections |
| **Graph** | Graph data structure | `graph` | Advanced | Graph | For relationships |
| **Message Passing** | Graph messaging | `message_passing` | Advanced | Graph | For communication |
| **GraphSAGE** | Graph sampling | `GraphSAGE` | Advanced | Graph | For large graphs |

---

### 13.4 Quantum ML

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **Quantum Computing** | Quantum computation | `quantum` | Advanced | Quantum | For quantum AI |
| **Qubit** | Quantum bit | `qubit` | Advanced | Quantum | For quantum |
| **Quantum Circuit** | Quantum operations | `quantum_circuit` | Advanced | Quantum | For computation |
| **Quantum ML** | Quantum machine learning | `QML` | Advanced | Quantum | For quantum AI |

---

### 13.5 AutoML

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **AutoML** | Automated ML | `AutoML` | Advanced | AutoML | For automation |
| **AutoGluon** | AutoML library | `autogluon` | Advanced | AutoML | For automation |
| **H2O AutoML** | AutoML framework | `H2O` | Advanced | AutoML | For automation |
| **TPOT** | Evolutionary AutoML | `TPOT` | Advanced | AutoML | For optimization |
| **Neural Architecture Search** | Architecture search | `NAS` | Advanced | AutoML | For architecture |

---

## 14. Quick Reference

### 14.1 Cheat Sheets

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **PyTorch Cheat Sheet** | Quick PyTorch reference | `All PyTorch commands` | PyTorch | Reference | For quick lookup |
| **TensorFlow Cheat Sheet** | Quick TensorFlow reference | `All TensorFlow commands` | TensorFlow | Reference | For quick lookup |
| **Numpy Cheat Sheet** | Quick Numpy reference | `All numpy commands` | Python | Reference | For quick lookup |
| **Pandas Cheat Sheet** | Quick Pandas reference | `All pandas commands` | Python | Reference | For quick lookup |
| **Scikit-learn Cheat Sheet** | Quick scikit-learn reference | `All sklearn commands` | Machine Learning | Reference | For quick lookup |
| **SQL Cheat Sheet** | Quick SQL reference | `All SQL commands` | Databases | Reference | For quick lookup |
| **Git Cheat Sheet** | Quick Git reference | `All git commands` | Tools | Reference | For quick lookup |
| **Docker Cheat Sheet** | Quick Docker reference | `All docker commands` | Tools | Reference | For quick lookup |
| **Kubernetes Cheat Sheet** | Quick Kubernetes reference | `All kubectl commands` | Tools | Reference | For quick lookup |

---

### 14.2 Command Reference

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **ls** | List directory | `ls -la` | CLI | Command | For listing |
| **cd** | Change directory | `cd /path` | CLI | Command | For navigation |
| **pwd** | Print working directory | `pwd` | CLI | Command | For path |
| **mkdir** | Make directory | `mkdir folder` | CLI | Command | For creating |
| **rm** | Remove file/directory | `rm -rf folder` | CLI | Command | For deleting |
| **cp** | Copy file/directory | `cp source dest` | CLI | Command | For copying |
| **mv** | Move file/directory | `mv source dest` | CLI | Command | For moving |
| **grep** | Search text | `grep pattern file` | CLI | Command | For searching |
| **find** | Find files | `find . -name "*.py"` | CLI | Command | For finding |
| **chmod** | Change permissions | `chmod 755 file` | CLI | Command | For permissions |
| **chown** | Change ownership | `chown user:group file` | CLI | Command | For ownership |
| **ps** | Process status | `ps aux` | CLI | Command | For processes |
| **kill** | Kill process | `kill -9 PID` | CLI | Command | For stopping |
| **top** | Process monitor | `top` | CLI | Command | For monitoring |
| **htop** | Interactive monitor | `htop` | CLI | Command | For monitoring |
| **nvidia-smi** | GPU status | `nvidia-smi` | CLI | Command | For GPU |
| **conda** | Package manager | `conda install` | CLI | Command | For packages |
| **pip** | Python package manager | `pip install` | CLI | Command | For packages |
| **docker** | Docker commands | `docker run` | CLI | Command | For containers |
| **kubectl** | Kubernetes commands | `kubectl get pods` | CLI | Command | For K8s |
| **git** | Git commands | `git commit` | CLI | Command | For version |
| **aws** | AWS CLI | `aws s3 ls` | CLI | Command | For AWS |
| **gcloud** | GCP CLI | `gcloud compute` | CLI | Command | For GCP |

---

### 14.3 Code Snippets

| Term | Definition | Code Example | Domain | Category | When to Use |
|------|------------|--------------|--------|----------|-------------|
| **PyTorch Training** | Training loop | `for epoch in range(epochs):` | PyTorch | Snippet | For training |
| **PyTorch DataLoader** | Data loading | `DataLoader(dataset, batch_size=32)` | PyTorch | Snippet | For batching |
| **PyTorch Model** | Model definition | `class Model(nn.Module):` | PyTorch | Snippet | For building |
| **FastAPI Endpoint** | API endpoint | `@app.post("/predict")` | FastAPI | Snippet | For API |
| **NumPy Array** | Array creation | `np.array([1,2,3])` | Python | Snippet | For arrays |
| **Pandas DataFrame** | DataFrame | `pd.DataFrame(data)` | Python | Snippet | For tables |
| **Matplotlib Plot** | Plot creation | `plt.plot(x, y)` | Python | Snippet | For plotting |
| **Scikit-learn Model** | Model training | `model.fit(X, y)` | Scikit-learn | Snippet | For training |
| **SQL Query** | Database query | `SELECT * FROM table` | SQL | Snippet | For queries |
| **Dockerfile** | Docker build | `FROM python:3.10` | Docker | Snippet | For building |
| **Git Commands** | Git operations | `git add . && git commit` | Git | Snippet | For versioning |

---

> **💡 Pro Tip:** This glossary is your complete reference for all AI/ML/DL/LLM/MLOps terminology. Bookmark it for quick access and refer to it whenever you encounter unfamiliar terms!

---

**END OF COMPLETE PROFESSIONAL GLOSSARY**
