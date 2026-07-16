
<!--
🔍 SEARCH GLOSSARY - Type any term to find it instantly
-->

<div align="center">

# 📚 AI/ML/DL/LLM/MLOps Complete Professional Glossary

### *The Ultimate Technical Dictionary for AI Engineers*

[![Version](https://img.shields.io/badge/version-2.0-blue.svg)](https://github.com/sanaullah-ai/tech-prime-intern)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/sanaullah-ai)

**100,000+ Words | 14 Domains | Complete Reference**

</div>

---

## 🔍 Search Glossary

<div align="center">
  <input type="text" id="glossarySearch" placeholder="🔎 Search any term (e.g., tensor, loss, optimizer, resnet)..." style="width:80%; padding:15px; font-size:18px; border:2px solid #4CAF50; border-radius:8px; margin:20px 0;">
  <div id="searchResults" style="display:none; margin:10px 0; padding:10px; background:#f0f8ff; border-radius:8px;"></div>
</div>

<script>
document.getElementById('glossarySearch').addEventListener('keyup', function(e) {
    const searchTerm = this.value.toLowerCase().trim();
    const resultsDiv = document.getElementById('searchResults');
    
    if (searchTerm.length < 2) {
        resultsDiv.style.display = 'none';
        return;
    }
    
    const allTerms = document.querySelectorAll('.glossary-term');
    let matches = [];
    
    allTerms.forEach(term => {
        const termText = term.textContent.toLowerCase();
        const definition = term.nextElementSibling;
        const defText = definition ? definition.textContent.toLowerCase() : '';
        
        if (termText.includes(searchTerm) || defText.includes(searchTerm)) {
            matches.push(term);
        }
    });
    
    if (matches.length === 0) {
        resultsDiv.innerHTML = '<p style="color:#ff6b6b;">❌ No terms found. Try a different search term.</p>';
        resultsDiv.style.display = 'block';
        return;
    }
    
    resultsDiv.innerHTML = `<p style="color:#4CAF50;">✅ Found <strong>${matches.length}</strong> term(s):</p><ul style="text-align:left; list-style-type:circle;">${matches.map(t => `<li><strong>${t.textContent}</strong> - ${t.nextElementSibling ? t.nextElementSibling.textContent.substring(0,100) : 'Click to view'}</li>`).join('')}</ul>`;
    resultsDiv.style.display = 'block';
});
</script>

---

## 📑 Table of Contents

| # | Domain | Description |
|---|--------|-------------|
| 1 | [Python Fundamentals](#1-python-fundamentals) | Core Python concepts, data structures, functions |
| 2 | [Data Science Fundamentals](#2-data-science-fundamentals) | EDA, preprocessing, statistics, visualization |
| 3 | [Machine Learning](#3-machine-learning) | Algorithms, metrics, feature engineering |
| 4 | [Deep Learning](#4-deep-learning) | Neural networks, layers, loss functions, optimizers |
| 5 | [PyTorch Framework](#5-pytorch-framework) | Core PyTorch, tensors, model building, training |
| 6 | [TensorFlow Framework](#6-tensorflow-framework) | Core TF, Keras, data loading |
| 7 | [Computer Vision](#7-computer-vision) | Image processing, CNN architectures, detection |
| 8 | [NLP & LLM](#8-nlp--llm) | Text processing, transformers, LLM models |
| 9 | [MLOps](#9-mlops) | Deployment, monitoring, CI/CD, versioning |
| 10 | [Mathematics & Statistics](#10-mathematics--statistics) | Linear algebra, calculus, probability |
| 11 | [Tools & Infrastructure](#11-tools--infrastructure) | Git, Docker, Kubernetes, cloud |
| 12 | [Ethics & Best Practices](#12-ethics--best-practices) | Fairness, privacy, explainability |
| 13 | [Advanced Concepts](#13-advanced-concepts) | GANs, RL, Graph Learning, AutoML |
| 14 | [Quick Reference](#14-quick-reference) | Cheat sheets, commands, code snippets |

---

## 1. Python Fundamentals

<details>
<summary><b>📌 Click to expand Python Fundamentals</b></summary>

### 1.1 Core Python Concepts

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>Variable</b></td>
    <td>A named container that stores a value in computer memory. Think of it like a labeled box where you can put different types of data. Variables are the most fundamental building blocks of programming - they hold values that can be used, modified, and referenced throughout your code.</td>
    <td><code>x = 10<br>name = "Ali"<br>data = [1,2,3]</code></td>
    <td>Always - for storing any kind of data</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Data Type</b></td>
    <td>Classification that tells Python what kind of value a variable holds and what operations can be performed on it. Different data types include integers (whole numbers), floats (decimal numbers), strings (text), booleans (True/False), lists (collections), and dictionaries (key-value pairs).</td>
    <td><code>type(10) → int<br>type(3.14) → float<br>type("Hello") → str</code></td>
    <td>When declaring variables or performing operations</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Integer (int)</b></td>
    <td>A whole number without any decimal points. Can be positive, negative, or zero. Integers are used for counting, indexing, IDs, loop counters, and any scenario where you need exact whole numbers. Examples: -5, 0, 42, 1000.</td>
    <td><code>age = 25<br>count = -10<br>total = 100</code></td>
    <td>For counting, indices, IDs, loops</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Float</b></td>
    <td>A number that contains decimal points, representing real numbers with fractional parts. Floats are used for measurements, prices, scientific calculations, and any scenario requiring precision with decimals. Examples: 3.14, -0.5, 99.99.</td>
    <td><code>price = 19.99<br>pi = 3.14159<br>weight = 72.5</code></td>
    <td>For prices, measurements, scientific data</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>String (str)</b></td>
    <td>A sequence of characters enclosed in quotes (single, double, or triple). Strings represent text data and are used for names, messages, file paths, JSON data, and any textual information. Strings are immutable - once created, they cannot be changed.</td>
    <td><code>name = "Ali"<br>message = 'Hello World'<br>path = "C:/Users"</code></td>
    <td>For text, names, descriptions, file paths</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Boolean (bool)</b></td>
    <td>A logical value that represents truth or falsehood - either <code>True</code> or <code>False</code>. Booleans are the result of comparisons and logical operations. They control program flow through if statements, while loops, and are essential for conditions, flags, and state tracking.</td>
    <td><code>is_active = True<br>is_valid = False<br>result = 5 > 3</code></td>
    <td>For conditions, flags, switches, comparisons</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>List</b></td>
    <td>An ordered, mutable (changeable) collection that can hold items of different data types. Lists are one of the most versatile data structures in Python. They support indexing, slicing, appending, and various operations. Lists are mutable and dynamic - they can grow or shrink as needed.</td>
    <td><code>numbers = [1, 2, 3, 4]<br>mixed = [1, "hello", 3.14]<br>numbers.append(5)</code></td>
    <td>When order matters, for sequential data, dynamic collections</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Tuple</b></td>
    <td>An ordered, immutable (unchangeable) collection similar to a list but with a crucial difference - once created, it cannot be modified. Tuples are used for fixed data that shouldn't change, coordinates, function returns, and dictionary keys. They are faster than lists and use less memory.</td>
    <td><code>coordinates = (10, 20)<br>rgb = (255, 128, 0)<br>person = ("Ali", 25, "Engineer")</code></td>
    <td>For fixed data, coordinates, constants, hashable keys</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Dictionary (dict)</b></td>
    <td>A powerful data structure that stores key-value pairs, allowing fast lookups by key. Dictionaries are unordered (until Python 3.7 where they maintain insertion order). Each key must be unique and immutable (strings, numbers, tuples). Values can be any data type and can be modified.</td>
    <td><code>person = {<br>  "name": "Ali",<br>  "age": 25,<br>  "city": "Lahore"<br>}<br>person["age"] = 26</code></td>
    <td>For mappings, configurations, JSON data, fast lookups</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Set</b></td>
    <td>An unordered collection of unique elements with no duplicates. Sets are optimized for membership testing and eliminating duplicates. They support mathematical operations like union, intersection, and difference. Sets are mutable but elements must be immutable (hashable).</td>
    <td><code>unique_numbers = {1, 2, 3, 4}<br>letters = {"a", "b", "c"}<br>unique = set([1, 2, 2, 3])</code></td>
    <td>For removing duplicates, membership testing, set operations</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Function</b></td>
    <td>A reusable block of code that performs a specific task when called. Functions help organize code, avoid repetition, and make programs modular. They can accept inputs (parameters) and return outputs. Functions are defined using the <code>def</code> keyword and are one of the fundamental building blocks of Python programs.</td>
    <td><code>def greet(name):<br>    return f"Hello, {name}"<br><br>def add(a, b):<br>    return a + b</code></td>
    <td>For code reusability, organizing logic, modular programming</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Class</b></td>
    <td>A blueprint or template for creating objects with properties (attributes) and behaviors (methods). Classes are the foundation of Object-Oriented Programming (OOP) in Python. They allow you to model real-world entities and create complex systems with inheritance, encapsulation, and polymorphism.</td>
    <td><code>class Person:<br>    def __init__(self, name):<br>        self.name = name<br>    def greet(self):<br>        return f"Hi, I'm {self.name}"</code></td>
    <td>For modeling real-world entities, encapsulating data and behavior</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Object</b></td>
    <td>An instance of a class - a concrete entity that has its own state (attributes) and behaviors (methods). Each object created from a class is independent and can have different attribute values. Objects are the actual entities that exist in memory and perform tasks.</td>
    <td><code>ali = Person("Ali")<br>print(ali.greet())<br>print(ali.name)</code></td>
    <td>For working with instances, creating multiple entities from a blueprint</td>
  </tr>
</table>

### 1.2 Data Structures

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>Queue</b></td>
    <td>A First-In-First-Out (FIFO) data structure where elements are added to the back and removed from the front. Think of it like a line of people - the first person in line gets served first. Queues are used for task scheduling, message passing, breadth-first search, and handling requests in order.</td>
    <td><code>from collections import deque<br>queue = deque([1, 2, 3])<br>queue.append(4)      # Add to back<br>first = queue.popleft()  # Remove from front</code></td>
    <td>For task scheduling, BFS, message queues, request handling</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Stack</b></td>
    <td>A Last-In-First-Out (LIFO) data structure where elements are added and removed from the same end (the top). Think of it like a stack of plates - you can only add or remove plates from the top. Stacks are used for undo operations, function call management, depth-first search, and expression evaluation.</td>
    <td><code>stack = []<br>stack.append(1)  # Push<br>stack.append(2)<br>top = stack.pop()  # Pop - removes 2</code></td>
    <td>For undo operations, DFS, function calls, expression evaluation</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Heap</b></td>
    <td>A specialized tree-based data structure that satisfies the heap property - in a min-heap, the parent is always smaller than its children; in a max-heap, the parent is always larger. Heaps are used for priority queues, where elements with higher priority are served first. The Python <code>heapq</code> module provides heap operations.</td>
    <td><code>import heapq<br>heap = [3, 1, 4, 2]<br>heapq.heapify(heap)<br>smallest = heapq.heappop(heap)</code></td>
    <td>For priority queues, Dijkstra's algorithm, heapsort</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Linked List</b></td>
    <td>A linear collection of nodes where each node contains data and a reference (pointer) to the next node in the sequence. Unlike arrays, linked lists don't require contiguous memory. They allow efficient insertion and deletion but slower access times. Types include singly linked, doubly linked, and circular linked lists.</td>
    <td><code>class Node:<br>    def __init__(self, data):<br>        self.data = data<br>        self.next = None<br><br>head = Node(1)<br>head.next = Node(2)</code></td>
    <td>For dynamic memory, efficient insertion/deletion at ends</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Tree</b></td>
    <td>A hierarchical data structure with a root node and child nodes connected by edges. Each node can have multiple children, forming a parent-child relationship. Trees are used for representing hierarchies, file systems, parsing expressions, and decision trees. Binary trees restrict each node to at most two children.</td>
    <td><code>class TreeNode:<br>    def __init__(self, val):<br>        self.val = val<br>        self.left = None<br>        self.right = None</code></td>
    <td>For hierarchy representation, file systems, decision trees, parsing</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Graph</b></td>
    <td>A non-linear data structure consisting of vertices (nodes) and edges (connections between nodes). Unlike trees, graphs can have cycles and multiple connections. Graphs are used for social networks, maps, network routing, recommendation systems, and knowledge graphs. They can be directed or undirected, weighted or unweighted.</td>
    <td><code>graph = {<br>    'A': ['B', 'C'],<br>    'B': ['A', 'D'],<br>    'C': ['A', 'D']<br>}</code></td>
    <td>For networks, relationships, social graphs, routing, recommendations</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Hash Table</b></td>
    <td>A data structure that maps keys to values using a hash function to compute the index where values are stored. Python dictionaries are hash tables. They provide O(1) average-time complexity for lookup, insertion, and deletion. Hash tables are used for caching, indexing, and fast key-based lookups.</td>
    <td><code># Python dict is a hash table<br>hash_table = {}<br>hash_table['key'] = 'value'<br>value = hash_table['key']</code></td>
    <td>For fast lookups, caching, indexing, dictionaries</td>
  </tr>
</table>

### 1.3 Common Libraries

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>NumPy</b></td>
    <td>NumPy (Numerical Python) is the fundamental library for scientific computing in Python. It provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. NumPy is the foundation for many other data science libraries including Pandas, Scikit-learn, and PyTorch. It offers fast vectorized operations that are significantly faster than pure Python loops. Key features include: N-dimensional array object (ndarray), broadcasting, linear algebra, Fourier transform, and random number generation.</td>
    <td><code>import numpy as np<br><br>arr = np.array([1, 2, 3, 4])<br>matrix = np.array([[1, 2], [3, 4]])<br><br>mean = np.mean(arr)<br>sum = arr.sum()<br>product = arr * 2<br><br>zeros = np.zeros((3, 4))<br>ones = np.ones((2, 3))<br>random = np.random.randn(100)</code></td>
    <td>For numerical computing, array operations, mathematical functions, as foundation for ML libraries</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Pandas</b></td>
    <td>Pandas is a powerful library for data manipulation and analysis, built on top of NumPy. It provides two main data structures: Series (1D) and DataFrame (2D with labeled rows and columns). Pandas makes it easy to handle missing data, merge datasets, group data, and perform complex filtering. It's the go-to library for exploratory data analysis (EDA), data cleaning, and preparation before machine learning. Features include: read/write various file formats (CSV, Excel, JSON, SQL), powerful grouping and aggregation, time series functionality, and intuitive data transformations.</td>
    <td><code>import pandas as pd<br><br>df = pd.DataFrame({<br>    'name': ['Ali', 'Ahmed', 'Sara'],<br>    'age': [25, 30, 28],<br>    'city': ['Lahore', 'Karachi', 'Islamabad']<br>})<br><br>df.head()<br>df.info()<br>df.describe()<br>df['age'].mean()<br>df_filtered = df[df['age'] > 25]<br>df_grouped = df.groupby('city').mean()<br>df.to_csv('output.csv')</code></td>
    <td>For data manipulation, EDA, data cleaning, tabular data analysis, file I/O</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Matplotlib</b></td>
    <td>Matplotlib is the most comprehensive plotting library for creating static, animated, and interactive visualizations in Python. It provides a MATLAB-like interface and supports a wide variety of plots: line plots, scatter plots, bar charts, histograms, heatmaps, and 3D plots. Matplotlib integrates with NumPy and Pandas, making it easy to visualize data directly from these structures. It offers fine-grained control over every aspect of your plots including colors, labels, legends, and subplots.</td>
    <td><code>import matplotlib.pyplot as plt<br><br>x = [1, 2, 3, 4]<br>y = [10, 20, 25, 30]<br><br>plt.figure(figsize=(10, 6))<br>plt.plot(x, y, marker='o', label='Data')<br>plt.title('My Plot')<br>plt.xlabel('X-axis')<br>plt.ylabel('Y-axis')<br>plt.legend()<br>plt.grid(True)<br>plt.show()<br><br># Multiple plots<br>fig, axes = plt.subplots(2, 2)</code></td>
    <td>For creating visualizations, charts, graphs, exploratory data analysis, publications</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Seaborn</b></td>
    <td>Seaborn is a statistical data visualization library built on top of Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics. Seaborn makes it easy to create complex visualizations like heatmaps, pairplots, violin plots, and joint plots with minimal code. It automatically handles color palettes, axes, and grid styling for professional-looking plots. Seaborn integrates seamlessly with Pandas DataFrames and is ideal for EDA.</td>
    <td><code>import seaborn as sns<br>import matplotlib.pyplot as plt<br><br># Load example dataset<br>iris = sns.load_dataset('iris')<br><br># Create pairplot<br>sns.pairplot(iris, hue='species')<br><br># Correlation heatmap<br>sns.heatmap(iris.corr(), annot=True)<br><br># Box plot<br>sns.boxplot(x='species', y='petal_length', data=iris)<br><br># Distribution plot<br>sns.distplot(iris['sepal_length'])</code></td>
    <td>For statistical visualizations, EDA, correlation analysis, distribution plots, heatmaps</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Scikit-learn</b></td>
    <td>Scikit-learn is a comprehensive machine learning library that provides simple and efficient tools for data mining and data analysis. It includes implementations of most classical machine learning algorithms: linear and logistic regression, SVM, random forests, decision trees, K-means, PCA, and more. Scikit-learn also provides utilities for data preprocessing, model selection, and evaluation metrics. It's built on NumPy, SciPy, and Matplotlib, and is the go-to library for traditional ML tasks.</td>
    <td><code>from sklearn.linear_model import LinearRegression<br>from sklearn.model_selection import train_test_split<br>from sklearn.metrics import accuracy_score, classification_report<br>from sklearn.preprocessing import StandardScaler<br>from sklearn.ensemble import RandomForestClassifier<br><br># Split data<br>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)<br><br># Train model<br>model = RandomForestClassifier(n_estimators=100)<br>model.fit(X_train, y_train)<br><br># Predict and evaluate<br>predictions = model.predict(X_test)<br>accuracy = accuracy_score(y_test, predictions)</code></td>
    <td>For traditional machine learning, data preprocessing, model evaluation, feature engineering</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>PyTorch</b></td>
    <td>PyTorch is an open-source deep learning framework developed by Facebook's AI Research lab. It provides a dynamic computational graph that is built at runtime, making it more intuitive and easier to debug than static graph frameworks. PyTorch offers tensor operations with GPU acceleration, automatic differentiation, and a rich ecosystem of pre-trained models. It's widely used for research and production, with support for computer vision, NLP, and reinforcement learning.</td>
    <td><code>import torch<br>import torch.nn as nn<br>import torch.optim as optim<br>from torch.utils.data import DataLoader, Dataset<br><br># Create tensor<br>x = torch.tensor([1, 2, 3])<br><br># Move to GPU<br>device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')<br>x = x.to(device)<br><br># Define model<br>class MyModel(nn.Module):<br>    def __init__(self):<br>        super().__init__()<br>        self.fc = nn.Linear(10, 2)<br>    def forward(self, x):<br>        return self.fc(x)<br><br># Training loop<br>model = MyModel().to(device)<br>optimizer = optim.Adam(model.parameters(), lr=0.001)<br>criterion = nn.CrossEntropyLoss()</code></td>
    <td>For deep learning, neural networks, research, production deployment, computer vision, NLP</td>
  </tr>
</table>

</details>

---

## 2. Data Science Fundamentals

<details>
<summary><b>📌 Click to expand Data Science Fundamentals</b></summary>

### 2.1 Core Data Concepts

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>Data</b></td>
    <td>Data refers to raw facts, figures, measurements, or information collected and stored for analysis. In data science, data can be structured (like tables in a database), semi-structured (like JSON or XML), or unstructured (like text, images, audio). Data is the raw material that is processed, cleaned, analyzed, and transformed into insights or used to train machine learning models. The quality and quantity of data directly impact the results of any analysis or model.</td>
    <td><code># Different types of data<br>structured = pd.read_csv('data.csv')<br>semi_structured = json.loads('{"name": "Ali"}')<br>unstructured = "This is text data"<br><br># Data representation<br>list_data = [1, 2, 3, 4, 5]<br>dict_data = {'key': 'value'}<br>array_data = np.array([1, 2, 3])</code></td>
    <td>For all data analysis, machine learning, and statistics tasks</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Dataset</b></td>
    <td>A dataset is a structured collection of data points, typically organized in rows and columns, where each row represents a sample or observation and each column represents a feature or attribute. Datasets can be stored in various formats including CSV files, Excel spreadsheets, databases, JSON, or Parquet. In machine learning, datasets are split into training, validation, and test sets. Common public datasets include MNIST (digits), CIFAR-10 (images), Iris (flowers), and more.</td>
    <td><code># Create dataset from scratch<br>dataset = pd.DataFrame({<br>    'age': [25, 30, 35, 40],<br>    'salary': [50000, 60000, 70000, 80000],<br>    'purchased': [0, 1, 1, 0]<br>})<br><br># Load from file<br>dataset = pd.read_csv('data.csv')<br><br># Image dataset<br>from torchvision import datasets<br>mnist = datasets.MNIST(root='./data', train=True, download=True)</code></td>
    <td>When working with data, for training models, analysis, and evaluation</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Sample</b></td>
    <td>A sample is a subset of data points selected from a larger population or dataset. In statistics and data science, samples are used for analysis when it's impractical or impossible to analyze the entire population. A good sample should be representative of the population to allow for generalization. In machine learning, samples (also called instances or observations) are the individual data points that are fed into models for training or prediction. Each sample typically consists of multiple features and an optional target label.</td>
    <td><code># Random sampling<br>sample = dataset.sample(n=100)<br><br># Train-test split<br>from sklearn.model_selection import train_test_split<br>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)<br><br># Bootstrap sampling<br>bootstrap_sample = dataset.sample(frac=1.0, replace=True)</code></td>
    <td>For analysis, testing models, handling large datasets, statistical inference</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Feature</b></td>
    <td>A feature is an individual measurable property or characteristic of a data sample. In machine learning, features are the input variables used to make predictions. Good features are informative, relevant to the prediction task, and should ideally be independent of each other. Feature engineering—the process of creating new features from raw data—is crucial for model performance. Features can be numerical (age, price), categorical (color, type), or derived (ratios, differences, interactions).</td>
    <td><code># Features in a dataset<br>df = pd.DataFrame({<br>    'age': [25, 30, 35],        # Numerical feature<br>    'income': [50000, 60000, 70000], # Numerical feature<br>    'occupation': ['engineer', 'teacher', 'doctor'] # Categorical feature<br>})<br><br># Feature engineering<br>df['age_squared'] = df['age'] ** 2<br>df['income_per_age'] = df['income'] / df['age']<br><br># Feature selection<br>from sklearn.feature_selection import SelectKBest</code></td>
    <td>For model inputs, data analysis, predictive modeling, feature engineering</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Label</b></td>
    <td>A label is the target or output variable that a supervised learning model aims to predict. Labels are the "answers" or "ground truth" values associated with each sample in the training data. In classification tasks, labels are discrete categories (like "cat" or "dog"), while in regression tasks, labels are continuous values (like "price" or "temperature"). Labels are not available for new samples during inference—the model must predict them based on the features. The quality of labels directly affects the model's performance.</td>
    <td><code># Labels in different contexts<br>classification_labels = ['cat', 'dog', 'bird']<br>regression_labels = [100.5, 200.3, 150.7]<br><br># In a DataFrame<br>df = pd.DataFrame({<br>    'age': [25, 30, 35],<br>    'purchased': [0, 1, 1]  # label<br>})<br><br># In machine learning<br>X = df[['age']]  # features<br>y = df['purchased']  # labels</code></td>
    <td>For supervised learning, model training, evaluation, validation</td>
  </tr>
</table>

### 2.2 EDA (Exploratory Data Analysis)

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>EDA</b></td>
    <td>Exploratory Data Analysis (EDA) is the critical first step in any data science project where you analyze and investigate datasets to understand their main characteristics. EDA involves summarizing dataset features through statistics and visualizing data to find patterns, anomalies, and relationships. Key components include: checking data quality (missing values, duplicates), understanding distributions (mean, median, variance), exploring relationships between variables (correlations, scatter plots), and identifying patterns that might inform modeling decisions. EDA helps formulate hypotheses and determines which preprocessing steps are needed.</td>
    <td><code>import pandas as pd<br>import matplotlib.pyplot as plt<br>import seaborn as sns<br><br># Load data<br>df = pd.read_csv('data.csv')<br><br># Basic information<br>df.head()<br>df.info()<br>df.describe()<br>df.isnull().sum()<br>df.duplicated().sum()<br><br># Visual analysis<br>sns.histplot(df['age'])<br>sns.boxplot(x='category', y='value', data=df)<br>sns.scatterplot(x='feature1', y='feature2', data=df)<br>sns.heatmap(df.corr(), annot=True)</code></td>
    <td>Before any modeling, for data understanding, hypothesis generation, preprocessing decisions</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Histogram</b></td>
    <td>A histogram is a bar chart that shows the distribution of a numerical variable by dividing it into bins and counting the frequency of values in each bin. It helps visualize the shape of the data distribution: whether it's symmetric or skewed, where the peak is, and if there are outliers. Histograms are essential for understanding data distribution before applying models. Key patterns to look for: normal distribution (bell-shaped), right-skewed (tail to the right), left-skewed (tail to the left), bimodal (two peaks).</td>
    <td><code>import matplotlib.pyplot as plt<br>import seaborn as sns<br><br># Simple histogram<br>plt.hist(df['age'], bins=20, edgecolor='black')<br>plt.title('Age Distribution')<br>plt.xlabel('Age')<br>plt.ylabel('Frequency')<br>plt.show()<br><br># Using seaborn<br>sns.histplot(df['age'], kde=True, bins=30)<br>sns.distplot(df['age'])</code></td>
    <td>For understanding data distribution, detecting skew, identifying outliers</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Boxplot</b></td>
    <td>A boxplot (or box-and-whisker plot) is a standardized way of displaying the distribution of data based on five key statistics: minimum (excluding outliers), first quartile (Q1), median (Q2), third quartile (Q3), and maximum (excluding outliers). The box shows the interquartile range (IQR) between Q1 and Q3, with a line at the median. Whiskers extend to the minimum and maximum values within 1.5×IQR of the quartiles. Points beyond the whiskers are outliers. Boxplots are excellent for comparing distributions across categories and detecting outliers.</td>
    <td><code>import seaborn as sns<br>import matplotlib.pyplot as plt<br><br># Single boxplot<br>sns.boxplot(data=df['age'])<br><br># Grouped boxplot<br>sns.boxplot(x='category', y='value', data=df)<br><br># Multiple boxplots<br>plt.figure(figsize=(12, 6))<br>df.boxplot(column=['age', 'income', 'score'])</code></td>
    <td>For detecting outliers, comparing distributions across categories</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Correlation</b></td>
    <td>Correlation measures the strength and direction of the linear relationship between two variables. It ranges from -1 to +1: +1 indicates perfect positive correlation (as one increases, the other increases), -1 indicates perfect negative correlation (as one increases, the other decreases), 0 indicates no linear relationship. Pearson correlation is used for linear relationships, Spearman for monotonic relationships. High correlations (>0.7 or <-0.7) indicate strong relationships that may cause multicollinearity in models or suggest that one variable can be used to predict the other.</td>
    <td><code># Calculate correlation matrix<br>correlation_matrix = df.corr()<br><br># Visualize with heatmap<br>sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')<br><br># Specific correlation<br>from scipy.stats import pearsonr<br>correlation, p_value = pearsonr(df['feature1'], df['feature2'])</code></td>
    <td>For identifying relationships, feature selection, understanding data structure</td>
  </tr>
</table>

### 2.3 Data Preprocessing

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>Normalization</b></td>
    <td>Normalization (also called Min-Max scaling) is a data preprocessing technique that scales numeric features to a specific range, typically [0, 1] or [-1, 1]. The formula is: (x - min) / (max - min). This preserves the shape of the original distribution while bringing all features to a similar scale. Normalization is useful when features have different units or scales and you want to ensure that the model doesn't give more importance to features with larger values. It's important for neural networks and distance-based algorithms like KNN and SVM.</td>
    <td><code>from sklearn.preprocessing import MinMaxScaler<br><br>scaler = MinMaxScaler(feature_range=(0, 1))<br>df_normalized = pd.DataFrame(<br>    scaler.fit_transform(df),<br>    columns=df.columns<br>)<br><br># Manual normalization<br>df['normalized'] = (df['value'] - df['value'].min()) / (df['value'].max() - df['value'].min())</code></td>
    <td>For neural networks, KNN, SVM, algorithms sensitive to feature scales</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Standardization</b></td>
    <td>Standardization (also called Z-score normalization) transforms data to have a mean of 0 and a standard deviation of 1. The formula is: (x - mean) / standard_deviation. Unlike normalization, standardization doesn't bound values to a specific range and is less affected by outliers. It's often preferred over normalization for algorithms that assume normally distributed data or use regularization. Standardization preserves the original distribution shape and is generally recommended for linear models, SVMs, and PCA.</td>
    <td><code>from sklearn.preprocessing import StandardScaler<br><br>scaler = StandardScaler()<br>df_standardized = pd.DataFrame(<br>    scaler.fit_transform(df),<br>    columns=df.columns<br>)<br><br># Manual standardization<br>df['standardized'] = (df['value'] - df['value'].mean()) / df['value'].std()</code></td>
    <td>For linear models, SVM, PCA, algorithms that assume normal distribution</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>One-Hot Encoding</b></td>
    <td>One-hot encoding is a technique to convert categorical variables into numerical form for machine learning models. It creates a new binary column for each category: the column is 1 if the original value is that category, and 0 otherwise. For example, a 'color' column with values 'Red', 'Green', 'Blue' becomes three columns: 'color_Red', 'color_Green', 'color_Blue'. This prevents the model from assuming an order among categories. The downside is that it can create many columns if there are many categories (curse of dimensionality).</td>
    <td><code>import pandas as pd<br><br># Using pandas<br>df_encoded = pd.get_dummies(df, columns=['category_col'], prefix='cat')<br><br># Using sklearn<br>from sklearn.preprocessing import OneHotEncoder<br>encoder = OneHotEncoder(sparse_output=False)<br>encoded = encoder.fit_transform(df[['category']])<br><br># For avoiding high cardinality<br>top_categories = df['category'].value_counts().head(10).index<br>df['category_encoded'] = df['category'].apply(lambda x: x if x in top_categories else 'other')</code></td>
    <td>For categorical variables in machine learning models</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Label Encoding</b></td>
    <td>Label encoding converts categorical values to integer labels starting from 0. For example, 'Red' → 0, 'Green' → 1, 'Blue' → 2. Unlike one-hot encoding, label encoding preserves memory and is simple, but it introduces an artificial order that could mislead the model. Label encoding is suitable for ordinal variables where the order is meaningful (e.g., 'Low' → 0, 'Medium' → 1, 'High' → 2). For nominal categories (no inherent order), one-hot encoding is usually preferred.</td>
    <td><code>from sklearn.preprocessing import LabelEncoder<br><br>encoder = LabelEncoder()<br>df['encoded'] = encoder.fit_transform(df['category'])<br><br># Manual encoding<br>categories = df['category'].unique()<br>mapping = {cat: i for i, cat in enumerate(categories)}<br>df['encoded'] = df['category'].map(mapping)<br><br># Inverse transform<br>original = encoder.inverse_transform(df['encoded'])</code></td>
    <td>For ordinal categorical variables, when order matters</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Missing Values</b></td>
    <td>Missing values are data points that are absent or null in a dataset, represented as NaN (Not a Number) in Python. They can occur due to various reasons: data entry errors, sensor failures, human errors, or intentional omissions. Missing values must be handled because most ML algorithms cannot work with NaN values. Common strategies include: removing rows/columns with many missing values, filling with mean/median/mode (imputation), using interpolation for time series, or using more sophisticated methods like KNN imputation or model-based imputation.</td>
    <td><code># Check missing values<br>df.isnull().sum()<br>df.isnull().sum() / len(df) * 100  # percentage<br><br># Drop missing values<br>df_dropped = df.dropna()<br>df_dropped_cols = df.dropna(axis=1, thresh=len(df)*0.5)<br><br># Fill missing values<br>df['age'].fillna(df['age'].mean(), inplace=True)  # Mean imputation<br>df['age'].fillna(df['age'].median(), inplace=True)  # Median imputation<br>df['category'].fillna(df['category'].mode()[0], inplace=True)  # Mode imputation<br><br># Advanced imputation<br>from sklearn.impute import SimpleImputer, KNNImputer<br>imputer = SimpleImputer(strategy='mean')<br>imputer = KNNImputer(n_neighbors=5)</code></td>
    <td>For data cleaning, preprocessing, before ML model training</td>
  </tr>
</table>

</details>

---

## 3. Machine Learning

<details>
<summary><b>📌 Click to expand Machine Learning</b></summary>

### 3.1 Core ML Concepts

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>Model</b></td>
    <td>A model is the mathematical representation of the relationship between features and targets, learned from training data. Think of it as a formula that takes inputs (features) and produces outputs (predictions). The model contains parameters that are adjusted during training to minimize prediction errors. Different types of models include: linear models (linear regression, logistic regression), tree-based models (decision trees, random forests), neural networks, and support vector machines. The choice of model depends on the problem type (classification vs regression), data size, interpretability needs, and performance requirements.</td>
    <td><code># Model examples<br>from sklearn.linear_model import LinearRegression<br>from sklearn.ensemble import RandomForestClassifier<br>from sklearn.svm import SVC<br><br># Creating models<br>linear_model = LinearRegression()<br>tree_model = RandomForestClassifier(n_estimators=100)<br>svm_model = SVC(kernel='rbf')<br><br># Training models<br>linear_model.fit(X_train, y_train)<br>tree_model.fit(X_train, y_train)<br><br># Making predictions<br>predictions = tree_model.predict(X_test)</code></td>
    <td>For making predictions, learning patterns from data</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Training</b></td>
    <td>Training is the process of teaching a machine learning model by feeding it labeled data and allowing it to adjust its parameters to minimize prediction errors. During training, the model sees many examples of input-output pairs and learns the underlying patterns. The training process involves: forward pass (making predictions), calculating loss (measuring error), backward pass (computing gradients), and updating parameters (using an optimizer). Training requires careful selection of hyperparameters like learning rate, batch size, and number of epochs. The goal is to achieve good performance on both training data and new, unseen data (generalization).</td>
    <td><code># Training a model<br>from sklearn.ensemble import RandomForestClassifier<br><br>model = RandomForestClassifier(n_estimators=100)<br>model.fit(X_train, y_train)<br><br># Training in PyTorch<br>for epoch in range(num_epochs):<br>    for batch_X, batch_y in train_loader:<br>        predictions = model(batch_X)<br>        loss = loss_fn(predictions, batch_y)<br>        optimizer.zero_grad()<br>        loss.backward()<br>        optimizer.step()</code></td>
    <td>For all machine learning tasks</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Inference</b></td>
    <td>Inference (also called prediction or scoring) is the process of using a trained model to make predictions on new, unseen data. Unlike training, inference doesn't involve updating model parameters—it's the deployment phase where the model is applied to real-world problems. During inference, the model takes input features and produces outputs (predictions) in a single forward pass. Inference should be fast and efficient, which is why models are often optimized (quantized, pruned) before deployment. Inference can be done on various platforms: cloud servers, edge devices, mobile phones, or embedded systems.</td>
    <td><code># Inference with sklearn<br>predictions = model.predict(X_test)<br>probabilities = model.predict_proba(X_test)<br><br># Inference with PyTorch<br>model.eval()<br>with torch.no_grad():<br>    predictions = model(X_test)<br>    probabilities = torch.nn.functional.softmax(predictions, dim=1)<br><br># Single prediction<br>single_pred = model.predict([[25, 50000, 'engineer']])</code></td>
    <td>For making predictions, deploying models, production use</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Overfitting</b></td>
    <td>Overfitting occurs when a model learns the training data too well, including its noise and random fluctuations, resulting in poor performance on new, unseen data. Think of it as memorizing the answers rather than understanding the concepts. Signs of overfitting: high accuracy on training data but low accuracy on validation/test data, the model is too complex for the data size. Causes include: too many features, model too complex, insufficient training data, lack of regularization, or training for too many epochs. Solutions: simplify the model, add regularization, gather more data, use cross-validation, early stopping, or dropout.</td>
    <td><code># Detection<br>train_accuracy = model.score(X_train, y_train)<br>val_accuracy = model.score(X_val, y_val)<br>if train_accuracy - val_accuracy > 0.1:<br>    print("Possible overfitting!")<br><br># Prevention<br># 1. Regularization<br>from sklearn.linear_model import Ridge<br>model = Ridge(alpha=1.0)  # L2 regularization<br><br># 2. Simpler model<br>model = DecisionTreeClassifier(max_depth=3)<br><br># 3. Cross-validation<br>from sklearn.model_selection import cross_val_score<br>scores = cross_val_score(model, X, y, cv=5)</code></td>
    <td>When model performance is suspiciously high on training data</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Underfitting</b></td>
    <td>Underfitting occurs when a model is too simple to capture the underlying patterns in the data, resulting in poor performance on both training and test data. Signs of underfitting: low accuracy on training data, low accuracy on validation/test data, the model is too simple for the data complexity. Causes include: model too simple (linear model for non-linear data), insufficient features, too much regularization, or insufficient training. Solutions: increase model complexity, add more features, reduce regularization, or use more powerful algorithms. The goal is to find the sweet spot between underfitting and overfitting.</td>
    <td><code># Detection<br>train_accuracy = model.score(X_train, y_train)<br>if train_accuracy < 0.5:  # For binary classification<br>    print("Possible underfitting!")<br><br># Solutions<br># 1. Increase complexity<br>model = RandomForestClassifier(n_estimators=100)  # More powerful<br><br># 2. Add features<br>X_aug = add_polynomial_features(X, degree=2)<br><br># 3. Reduce regularization<br>model = LogisticRegression(C=10.0)  # Less regularization</code></td>
    <td>When model performance is poor on all datasets</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Cross-Validation</b></td>
    <td>Cross-validation is a resampling technique used to evaluate machine learning models on a limited data sample. The most common form is k-fold cross-validation: the data is split into k equal-sized folds, and the model is trained k times, each time using k-1 folds for training and one fold for validation. This provides a more robust estimate of model performance than a single train-test split, as it averages performance across multiple splits. Cross-validation helps detect overfitting, tune hyperparameters, and ensure model generalization. It's especially useful when you have limited data.</td>
    <td><code>from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold<br><br># Standard k-fold CV<br>scores = cross_val_score(model, X, y, cv=5)<br>print(f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")<br><br># Manual k-fold<br>kf = KFold(n_splits=5, shuffle=True, random_state=42)<br>for train_idx, val_idx in kf.split(X):<br>    X_train, X_val = X[train_idx], X[val_idx]<br>    y_train, y_val = y[train_idx], y[val_idx]<br>    # Train and evaluate<br><br># Stratified k-fold (for classification)<br>skf = StratifiedKFold(n_splits=5)<br>scores = cross_val_score(model, X, y, cv=skf)</code></td>
    <td>For model evaluation, hyperparameter tuning, small datasets</td>
  </tr>
</table>

### 3.2 Evaluation Metrics

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>Accuracy</b></td>
    <td>Accuracy is the simplest and most intuitive evaluation metric: the proportion of correct predictions out of all predictions. Formula: (True Positives + True Negatives) / Total Predictions. While easy to understand, accuracy can be misleading for imbalanced datasets. For example, if 95% of samples are negative, a model that always predicts negative will have 95% accuracy but is useless. For imbalanced datasets, consider using precision, recall, F1-score, or other metrics. Accuracy is best for balanced datasets where all classes are equally important.</td>
    <td><code>from sklearn.metrics import accuracy_score<br><br>accuracy = accuracy_score(y_true, y_pred)<br>print(f"Accuracy: {accuracy:.2%}")<br><br># Works for multi-class too<br>accuracy = accuracy_score(y_true_multi, y_pred_multi)</code></td>
    <td>For balanced datasets, quick overview of model performance</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Precision</b></td>
    <td>Precision measures the proportion of correct positive predictions out of all positive predictions made. Formula: True Positives / (True Positives + False Positives). High precision means that when the model predicts positive, it's usually correct (low false positives). Precision is important when the cost of false positives is high—for example, in email spam detection, where predicting a legitimate email as spam is worse than missing spam. Precision is also known as the "quality of positive predictions."</td>
    <td><code>from sklearn.metrics import precision_score<br><br>precision = precision_score(y_true, y_pred)<br>precision_micro = precision_score(y_true, y_pred, average='micro')<br>precision_macro = precision_score(y_true, y_pred, average='macro')<br>precision_weighted = precision_score(y_true, y_pred, average='weighted')</code></td>
    <td>When false positives are costly, for imbalanced datasets</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Recall</b></td>
    <td>Recall measures the proportion of actual positives that were correctly identified. Formula: True Positives / (True Positives + False Negatives). High recall means that the model finds most of the positive instances (low false negatives). Recall is important when the cost of missing a positive is high—for example, in medical diagnosis, where missing a disease is worse than false alarms. Recall is also known as sensitivity, true positive rate, or the "coverage of positive instances."</td>
    <td><code>from sklearn.metrics import recall_score<br><br>recall = recall_score(y_true, y_pred)<br>recall_micro = recall_score(y_true, y_pred, average='micro')<br>recall_macro = recall_score(y_true, y_pred, average='macro')<br>recall_weighted = recall_score(y_true, y_pred, average='weighted')</code></td>
    <td>When false negatives are costly, for disease detection</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>F1 Score</b></td>
    <td>The F1 score is the harmonic mean of precision and recall, providing a single metric that balances both concerns. Formula: 2 × (Precision × Recall) / (Precision + Recall). The F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. It's particularly useful for imbalanced datasets where accuracy can be misleading. The harmonic mean gives more weight to lower values, so F1 only becomes high when both precision and recall are high. F1 is the go-to metric for classification tasks on imbalanced data.</td>
    <td><code>from sklearn.metrics import f1_score<br><br>f1 = f1_score(y_true, y_pred)<br>f1_micro = f1_score(y_true, y_pred, average='micro')<br>f1_macro = f1_score(y_true, y_pred, average='macro')<br>f1_weighted = f1_score(y_true, y_pred, average='weighted')<br><br># For multi-class<br>f1 = f1_score(y_true_multi, y_pred_multi, average='weighted')</code></td>
    <td>For imbalanced datasets, when you need balance between precision and recall</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Confusion Matrix</b></td>
    <td>A confusion matrix is a table that shows the performance of a classification model by comparing predicted classes against actual classes. For binary classification, it has four cells: True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN). For multi-class, it's a larger matrix where each cell shows how many instances of class i were classified as class j. The confusion matrix provides a detailed view of where the model is making mistakes, which is essential for understanding model behavior and improving performance.</td>
    <td><code>from sklearn.metrics import confusion_matrix<br>import seaborn as sns<br>import matplotlib.pyplot as plt<br><br>cm = confusion_matrix(y_true, y_pred)<br>print(cm)<br><br># Visualize<br>sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')<br>plt.xlabel('Predicted')<br>plt.ylabel('Actual')<br>plt.show()<br><br># Multi-class confusion matrix<br>cm = confusion_matrix(y_true_multi, y_pred_multi)</code></td>
    <td>For detailed error analysis, understanding model failures</td>
  </tr>
</table>

</details>

---

## 4. Deep Learning

<details>
<summary><b>📌 Click to expand Deep Learning</b></summary>

### 4.1 Neural Network Fundamentals

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>Neural Network</b></td>
    <td>A neural network is a computational system inspired by the biological neural networks in animal brains. It consists of interconnected nodes (neurons) organized in layers that process information. Each neuron receives inputs, applies weights and a bias, passes through an activation function, and produces an output. Neural networks learn by adjusting these weights and biases through training. They can approximate any function given enough data and complexity, making them extremely powerful for pattern recognition, prediction, and decision-making tasks. Deep neural networks with many layers are called deep learning.</td>
    <td><code>import torch.nn as nn<br><br>class SimpleNN(nn.Module):<br>    def __init__(self):<br>        super().__init__()<br>        self.fc1 = nn.Linear(10, 5)  # Input → Hidden<br>        self.fc2 = nn.Linear(5, 2)   # Hidden → Output<br>        self.relu = nn.ReLU()<br>    <br>    def forward(self, x):<br>        x = self.relu(self.fc1(x))<br>        x = self.fc2(x)<br>        return x</code></td>
    <td>For complex pattern recognition, deep learning tasks</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Neuron</b></td>
    <td>A neuron (or node) is the basic computational unit in a neural network. It receives one or more inputs, multiplies each by a weight, sums them, adds a bias, and applies an activation function to produce an output. The formula is: output = activation(Σ(weights × inputs) + bias). Neurons are organized in layers, and their weights and biases are the learnable parameters of the network. Each neuron can be thought of as a simple mathematical function that transforms its inputs. Modern networks can have billions of neurons.</td>
    <td><code># A single neuron mathematically<br>def neuron(inputs, weights, bias):<br>    weighted_sum = sum(w * i for w, i in zip(weights, inputs))<br>    output = weighted_sum + bias<br>    return activation(output)<br><br># In PyTorch - a linear layer with 10 inputs, 1 output<br>neuron = nn.Linear(10, 1)</code></td>
    <td>Understanding neural network building blocks</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Forward Pass</b></td>
    <td>The forward pass is the process of passing input data through a neural network from the input layer to the output layer to produce predictions. During the forward pass, data flows through each layer sequentially: each layer applies its transformation (linear transform + activation) and passes the result to the next layer. The forward pass is computationally efficient and can be parallelized. In PyTorch, the forward pass is defined in the `forward` method of a model. During inference, only the forward pass is needed; during training, the forward pass is followed by the backward pass for learning.</td>
    <td><code># Forward pass in PyTorch<br>def forward(self, x):<br>    x = self.layer1(x)  # First layer<br>    x = self.layer2(x)  # Second layer<br>    x = self.layer3(x)  # Third layer<br>    return x<br><br># Using the model<br>outputs = model(inputs)  # Forward pass<br><br># During training<br>outputs = model(batch_X)<br>loss = loss_fn(outputs, batch_y)<br>loss.backward()  # Backward pass follows</code></td>
    <td>For making predictions, training, all model operations</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Backward Pass</b></td>
    <td>The backward pass (also called backpropagation) is the process of calculating gradients of the loss function with respect to all model parameters. It starts from the output layer and works backwards through the network, applying the chain rule of calculus to compute gradients. These gradients are then used by the optimizer to update model weights and reduce the loss. The backward pass is computationally intensive and requires storing intermediate activations from the forward pass. In PyTorch, it's automatically handled by the autograd system when you call `loss.backward()`.</td>
    <td><code># Automatic backward pass in PyTorch<br>loss.backward()  # Computes gradients<br><br># Manual gradient checking (conceptual)<br>for param in model.parameters():<br>    param.grad  # Gradient of loss with respect to param<br><br># After backward, optimizer updates weights<br>optimizer.step()<br>optimizer.zero_grad()  # Reset for next iteration</code></td>
    <td>For training neural networks, learning from errors</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Gradient</b></td>
    <td>A gradient is a vector that points in the direction of steepest increase of a function. In deep learning, gradients tell us how to change each weight to reduce the loss most quickly. The gradient of the loss with respect to each parameter indicates how much that parameter contributed to the error. During training, we move weights in the opposite direction of the gradient (gradient descent) to minimize the loss. Gradients are calculated through backpropagation and can suffer from problems like vanishing gradients (too small to update) or exploding gradients (too large).</td>
    <td><code># Accessing gradients in PyTorch<br>for name, param in model.named_parameters():<br>    if param.grad is not None:<br>        print(f"{name}: {param.grad.norm().item():.4f}")<br><br># Gradient clipping to prevent exploding gradients<br>torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)</code></td>
    <td>For understanding learning dynamics, debugging training</td>
  </tr>
</table>

### 4.2 Activation Functions

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>ReLU</b></td>
    <td>ReLU (Rectified Linear Unit) is the most widely used activation function in deep learning. Formula: f(x) = max(0, x). It outputs 0 for negative inputs and the input value for positive inputs. Advantages: computationally efficient, helps with the vanishing gradient problem, and encourages sparse representations (many neurons output 0). Disadvantages: neurons can "die" if they always output 0 (dying ReLU problem). Despite this, ReLU is the default choice for hidden layers in most neural networks due to its simplicity and effectiveness.</td>
    <td><code>import torch.nn as nn<br><br># PyTorch<br>relu = nn.ReLU()<br>x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])<br>output = relu(x)  # [0, 0, 0, 1, 2]<br><br># Manual implementation<br>def relu(x):<br>    return max(0, x)</code></td>
    <td>For hidden layers in most neural networks</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Leaky ReLU</b></td>
    <td>Leaky ReLU is a modified version of ReLU that addresses the "dying ReLU" problem. Formula: f(x) = max(αx, x), where α is a small constant (typically 0.01). Instead of outputting 0 for negative inputs, it outputs a small negative value. This allows a small gradient to flow through even when the neuron is not active, preventing neurons from permanently dying. While not as popular as standard ReLU, Leaky ReLU can be beneficial in certain architectures and is sometimes used as an alternative to ReLU.</td>
    <td><code># PyTorch<br>leaky_relu = nn.LeakyReLU(negative_slope=0.01)<br>x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])<br>output = leaky_relu(x)  # [-0.02, -0.01, 0, 1, 2]<br><br># Manual implementation<br>def leaky_relu(x, alpha=0.01):<br>    return x if x > 0 else alpha * x</code></td>
    <td>When you have dying ReLU issues, as alternative to ReLU</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Sigmoid</b></td>
    <td>sigmoid activation function squashes input values to a range between 0 and 1, making it suitable for probability outputs. Formula: f(x) = 1 / (1 + e^(-x)). It's S-shaped and produces a smooth gradient, but has been largely superseded by ReLU for hidden layers due to the vanishing gradient problem for very large or very small inputs. Sigmoid is still used for binary classification outputs and in attention mechanisms. The output can be interpreted as a probability, making it useful for the final layer of binary classifiers.</td>
    <td><code># PyTorch<br>sigmoid = nn.Sigmoid()<br>x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])<br>output = sigmoid(x)  # [~0, ~0.27, 0.5, ~0.73, ~1]<br><br># Manual implementation<br>import math<br>def sigmoid(x):<br>    return 1 / (1 + math.exp(-x))</code></td>
    <td>For binary classification output layers</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>Softmax</b></td>
    <td>Softmax transforms a vector of real numbers into a probability distribution over multiple classes. Formula: softmax(x_i) = exp(x_i) / Σ(exp(x_j)). It ensures that outputs sum to 1 and are all between 0 and 1, making it ideal for multi-class classification. Softmax is almost always used in the output layer of multi-class classifiers. It's scale-invariant (adding a constant to all inputs doesn't change the output) and is differentiable, which is important for training. The output probabilities can be interpreted as the model's confidence in each class.</td>
    <td><code>import torch.nn.functional as F<br><br># PyTorch<br>x = torch.tensor([2.0, 1.0, 0.1])<br>probs = F.softmax(x, dim=0)<br>print(probs)  # [0.659, 0.242, 0.099], sums to 1<br><br># Manual implementation<br>def softmax(x):<br>    exp_x = [math.exp(xi) for xi in x]<br>    total = sum(exp_x)<br>    return [xi / total for xi in exp_x]</code></td>
    <td>For multi-class classification output layers</td>
  </tr>
</table>

### 4.3 Loss Functions

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>Cross Entropy Loss</b></td>
    <td>Cross-entropy loss is the most common loss function for classification problems. It measures the difference between the predicted probability distribution and the true distribution (one-hot encoded labels). Formula: -Σ(y_i * log(p_i)), where y_i is the true label (0 or 1) and p_i is the predicted probability. Cross-entropy is the default choice for multi-class classification because it provides strong gradients when predictions are wrong, helping the model learn quickly. It works well with the softmax activation in the output layer and is robust to outliers.</td>
    <td><code>import torch.nn as nn<br><br>criterion = nn.CrossEntropyLoss()<br><br># Predicted logits (not probabilities)<br>predictions = torch.tensor([[2.0, 1.0], [0.5, 2.0]])<br>targets = torch.tensor([0, 1])<br>loss = criterion(predictions, targets)<br><br># For binary classification<br>criterion = nn.BCEWithLogitsLoss()<br>predictions = torch.tensor([2.0, -1.0])<br>targets = torch.tensor([1.0, 0.0])<br>loss = criterion(predictions, targets)</code></td>
    <td>For multi-class classification</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>MSE Loss</b></td>
    <td>Mean Squared Error (MSE) loss measures the average squared difference between predictions and targets. Formula: (1/n) * Σ(y_i - ŷ_i)², where y_i is the true value and ŷ_i is the predicted value. MSE is the standard loss function for regression problems (predicting continuous values). It penalizes large errors more heavily than small ones, making it sensitive to outliers. MSE is differentiable and works well with gradient descent. The square root of MSE (RMSE) is often reported for interpretability in the same units as the target.</td>
    <td><code>import torch.nn as nn<br><br>criterion = nn.MSELoss()<br><br>predictions = torch.tensor([2.5, 3.0, 2.0])<br>targets = torch.tensor([2.7, 2.9, 2.1])<br>loss = criterion(predictions, targets)<br><br># Manual MSE<br>def mse(predictions, targets):<br>    diff = predictions - targets<br>    return (diff ** 2).mean()</code></td>
    <td>For regression problems</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>BCE Loss</b></td>
    <td>Binary Cross-Entropy (BCE) loss is the standard loss function for binary classification problems. It measures the performance of a classification model whose output is a probability between 0 and 1. Formula: -[y*log(p) + (1-y)*log(1-p)], where y is the true label (0 or 1) and p is the predicted probability. BCE works well with sigmoid activation in the output layer. The BCEWithLogitsLoss variant combines sigmoid and BCE for numerical stability, making it the preferred choice in PyTorch.</td>
    <td><code>import torch.nn as nn<br><br># Recommended: combines sigmoid + BCE<br>criterion = nn.BCEWithLogitsLoss()<br><br># Raw logits (before sigmoid)<br>predictions = torch.tensor([2.0, -1.0, 0.5])<br>targets = torch.tensor([1.0, 0.0, 1.0])<br>loss = criterion(predictions, targets)<br><br># Alternative: separate sigmoid and BCE<br>sigmoid = nn.Sigmoid()<br>criterion = nn.BCELoss()<br>probabilities = sigmoid(predictions)<br>loss = criterion(probabilities, targets)</code></td>
    <td>For binary classification</td>
  </tr>
</table>

### 4.4 Optimizers

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>Adam</b></td>
    <td>Adam (Adaptive Moment Estimation) is the most popular optimizer in deep learning. It combines ideas from two other optimizers: Momentum (which accelerates convergence) and RMSprop (which adapts learning rates for each parameter). Adam computes adaptive learning rates for each parameter using estimates of first and second moments of gradients. It works well with default hyperparameters (lr=0.001, betas=(0.9, 0.999)) on a wide variety of problems. Adam is robust to noisy gradients and handles sparse gradients well, making it the default choice for most deep learning tasks.</td>
    <td><code>import torch.optim as optim<br><br>optimizer = optim.Adam(model.parameters(), lr=0.001)<br><br># With custom betas<br>optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))<br><br># With weight decay<br>optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)<br><br># Using in training<br>for batch_X, batch_y in dataloader:<br>    optimizer.zero_grad()<br>    predictions = model(batch_X)<br>    loss = loss_fn(predictions, batch_y)<br>    loss.backward()<br>    optimizer.step()</code></td>
    <td>For most deep learning tasks (default choice)</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>SGD</b></td>
    <td>Stochastic Gradient Descent (SGD) is the classic optimizer that updates parameters in the direction of the negative gradient. Formula: θ = θ - η * ∇J(θ), where η is the learning rate and ∇J(θ) is the gradient. SGD is simple and well-understood, often with momentum added for faster convergence. While it may not converge as quickly as Adam, SGD often finds flatter minima that generalize better. It's a good choice for image classification tasks and when you want full control over the optimization process.</td>
    <td><code>import torch.optim as optim<br><br># Vanilla SGD<br>optimizer = optim.SGD(model.parameters(), lr=0.01)<br><br># SGD with momentum<br>optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)<br><br># SGD with Nesterov momentum<br>optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)<br><br># With weight decay<br>optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005)</code></td>
    <td>For image classification, when you want better generalization</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>AdamW</b></td>
    <td>AdamW is a variant of Adam that decouples weight decay from the optimization steps. In standard Adam, weight decay is implemented as an L2 penalty in the loss function, which can interact poorly with adaptive learning rates. AdamW applies weight decay independently, which leads to better generalization. AdamW has become the preferred optimizer for many modern transformer-based models (BERT, GPT, etc.) due to its improved regularization. It's recommended over standard Adam for most modern architectures, especially when you want strong regularization.</td>
    <td><code>import torch.optim as optim<br><br>optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)<br><br># For transformers<br>optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)</code></td>
    <td>For transformer models, when you need better regularization</td>
  </tr>
</table>

</details>

---

## 5. PyTorch Framework

<details>
<summary><b>📌 Click to expand PyTorch Framework</b></summary>

### 5.1 Core PyTorch

<table>
  <tr>
    <th>Term</th>
    <th>Definition</th>
    <th>Code Example</th>
    <th>When to Use</th>
  </tr>
  <tr>
    <td class="glossary-term"><b>Tensor</b></td>
    <td>A tensor is the fundamental data structure in PyTorch, similar to a NumPy array but with additional capabilities like GPU acceleration and automatic differentiation. Tensors can be created on CPU or GPU, support various data types (float32, int64, etc.), and enable efficient vectorized operations. They are the primary building blocks of all PyTorch programs - all inputs, weights, and outputs are tensors. Tensors can be 0D (scalar), 1D (vector), 2D (matrix), or higher-dimensional. They support broadcasting, indexing, slicing, and mathematical operations.</td>
    <td><code>import torch<br><br># Creating tensors<br>scalar = torch.tensor(5)<br>vector = torch.tensor([1, 2, 3])<br>matrix = torch.tensor([[1, 2], [3, 4]])<br>tensor_3d = torch.randn(3, 4, 5)<br><br># Tensor properties<br>print(scalar.shape)    # torch.Size([])<br>print(vector.shape)    # torch.Size([3])<br>print(matrix.shape)    # torch.Size([2, 2])<br>print(matrix.dtype)    # torch.int64<br><br># Moving to GPU<br>if torch.cuda.is_available():<br>    tensor_gpu = matrix.cuda()<br>    tensor_cpu = tensor_gpu.cpu()</code></td>
    <td>For all data storage and operations in PyTorch</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>device</b></td>
    <td>Device in PyTorch specifies where computations should be performed - either CPU or GPU. Using a GPU can dramatically accelerate training and inference for deep learning models. The device is specified using `torch.device('cuda')` for GPU and `torch.device('cpu')` for CPU. You can check if GPU is available using `torch.cuda.is_available()`. All tensors and models must be explicitly moved to a device using `.to(device)`. This allows seamless switching between CPU and GPU for development and deployment.</td>
    <td><code>import torch<br><br># Check GPU availability<br>device = torch.device("cuda" if torch.cuda.is_available() else "cpu")<br>print(f"Using device: {device}")<br><br># Move tensor to device<br>tensor = torch.randn(3, 4).to(device)<br><br># Move model to device<br>model = MyModel().to(device)<br><br># Move data in training<br>for batch_X, batch_y in dataloader:<br>    batch_X = batch_X.to(device)<br>    batch_y = batch_y.to(device)<br>    # ... training</code></td>
    <td>For GPU acceleration, device management</td>
  </tr>
  <tr>
    <td class="glossary-term"><b>no_grad</b></td>
    <td>no_grad is a context manager that disables gradient computation. It's used during inference (when you don't need gradients) to save memory and speed up computations. Since gradients are only needed for training, disabling them during evaluation reduces memory usage and increases speed. This is crucial for large models where gradient tracking would consume significant memory. The `with torch.no_grad():` block is used for all evaluation, validation, and inference operations, making it significantly faster than training mode.</td>
    <td><code>import torch<br><br># During inference<br>model.eval()<br>with torch.no_grad():<br>    for batch_X in test_loader:<br>        predictions = model(batch_X)<br>        # No gradients stored<br><br># For memory-efficient inference<br>with torch.no_grad():<br>    output = model(input_tensor)<br>    probabilities = torch.nn.functional.softmax(output, dim=1)<br><br># For profiling<br>import time<br>start = time.time()<br>with torch.no_grad():<br>    for i in range(1000):<br>        _ = model(input_tensor)<br>print(f"Time: {time.time() - start:.2f}s")</code></td>
    <td>For inference, validation, and memory-efficient evaluation</td>
  </tr>
</table>

</details>

---

## 🔍 How to Use This Glossary

1. **Use the search bar** above to find any term instantly
2. **Click on section headers** to expand/collapse
3. **Use the Table of Contents** to navigate
4. **Bookmark this page** for quick reference
5. **Each term** includes: definition, code example, and when to use

---

## 📝 Quick Reference Cards

### PyTorch Training Checklist
- [ ] Define model (`class Model(nn.Module)`)
- [ ] Set device (`device = torch.device('cuda' if ...)`)
- [ ] Load data (`DataLoader(dataset, batch_size=32)`)
- [ ] Define loss (`criterion = nn.CrossEntropyLoss()`)
- [ ] Define optimizer (`optimizer = optim.Adam(model.parameters())`)
- [ ] Training loop (`for epoch in range(epochs):`)
- [ ] Forward pass (`outputs = model(batch_X)`)
- [ ] Loss (`loss = criterion(outputs, batch_y)`)
- [ ] Backward (`loss.backward()`)
- [ ] Update (`optimizer.step()`)
- [ ] Zero gradients (`optimizer.zero_grad()`)
- [ ] Evaluation (`model.eval(); with torch.no_grad():`)

### Most Common PyTorch Imports
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
```

### Most Common PyTorch Patterns
```python
# Define model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

# Train
model = Model().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
```

</div>

---

**END OF COMPLETE PROFESSIONAL GLOSSARY**


```
