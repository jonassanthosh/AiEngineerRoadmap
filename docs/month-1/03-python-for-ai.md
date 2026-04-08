---
sidebar_position: 4
title: "Python for AI Engineering"
slug: python-for-ai
---


# Python for AI Engineering

:::info[What You'll Learn]
- NumPy arrays and vectorized operations
- Data loading and manipulation with Pandas
- Matplotlib for visualization
- Writing clean, efficient ML experiment code
:::

:::note[Prerequisites]
[Python Essentials](/curriculum/month-1/python-essentials) — you should be comfortable with Python syntax, data structures, functions, and classes before starting this lesson.
:::

**Estimated time:** Reading: ~40 min | Exercises: ~3 hours

Python is the lingua franca of AI and machine learning. Not because it's the fastest language—it isn't—but because its ecosystem is unmatched: NumPy for numerical computing, Pandas for data manipulation, Matplotlib for visualization, PyTorch and TensorFlow for deep learning, and Hugging Face for working with pre-trained models.

This chapter is a **crash course** in the three libraries you'll use daily: NumPy, Pandas, and Matplotlib. If you already know these well, skim through and try the exercises. If not, work through every code example—type them out, modify them, break them.

## NumPy: The Foundation of Numerical Python

NumPy is the bedrock of Python's scientific computing stack. Every ML framework—PyTorch, TensorFlow, scikit-learn—is built on top of NumPy or uses its conventions. NumPy's core contribution is the **ndarray**: an efficient, fixed-type, multidimensional array that enables vectorized operations.

### Creating Arrays

```python title="Creating NumPy Arrays"
import numpy as np

# From Python lists
a = np.array([1, 2, 3, 4, 5])
print(f"1D array: {a}, dtype: {a.dtype}")

# 2D array (matrix)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"2D array shape: {matrix.shape}")  # (2, 3)

# Common constructors
zeros = np.zeros((3, 4))        # 3x4 matrix of zeros
ones = np.ones((2, 2))          # 2x2 matrix of ones
identity = np.eye(3)            # 3x3 identity matrix
sequence = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5) # 5 evenly spaced values from 0 to 1

print(f"zeros:\\n{zeros}")
print(f"linspace: {linspace}")

# Random arrays (critical for initializing neural network weights)
np.random.seed(42)
uniform = np.random.rand(3, 3)         # uniform [0, 1)
normal = np.random.randn(3, 3)         # standard normal
integers = np.random.randint(0, 10, (2, 3))  # random ints

print(f"\\nRandom normal:\\n{normal}")
```

:::tip[Line-by-Line Walkthrough]
- **`import numpy as np`** — Imports the NumPy library and gives it the short nickname `np`, which is the universal convention.
- **`a = np.array([1, 2, 3, 4, 5])`** — Creates a 1D NumPy array from a regular Python list. Think of it as upgrading a shopping list into a spreadsheet column.
- **`matrix = np.array([[1, 2, 3], [4, 5, 6]])`** — Creates a 2D array (a table with 2 rows and 3 columns).
- **`np.zeros((3, 4))`** — Builds a 3-row, 4-column table filled entirely with zeros.
- **`np.ones((2, 2))`** — Same idea, but every cell is 1.
- **`np.eye(3)`** — Creates a 3×3 identity matrix (1s on the diagonal, 0s everywhere else), like the "do nothing" matrix in linear algebra.
- **`np.arange(0, 10, 2)`** — Generates numbers from 0 to 10 (exclusive) stepping by 2: `[0, 2, 4, 6, 8]`.
- **`np.linspace(0, 1, 5)`** — Creates 5 evenly spaced numbers between 0 and 1, inclusive of both endpoints.
- **`np.random.seed(42)`** — Locks the random number generator so you get the same "random" numbers every time you run the code. Great for reproducibility.
- **`np.random.rand(3, 3)`** — Generates a 3×3 grid of random numbers between 0 and 1.
- **`np.random.randn(3, 3)`** — Generates a 3×3 grid of random numbers drawn from a bell curve (mean 0, spread 1).
- **`np.random.randint(0, 10, (2, 3))`** — Generates a 2×3 grid of random whole numbers from 0 to 9.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Install NumPy:
```bash
pip install numpy
```

**Steps:**
1. Save the code to a file, e.g. `numpy_arrays.py`
2. Open a terminal and run: `python numpy_arrays.py`

**Expected output:**
```
1D array: [1 2 3 4 5], dtype: int64
2D array shape: (2, 3)
zeros:
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
linspace: [0.   0.25 0.5  0.75 1.  ]

Random normal:
[[ 0.49671415 -0.1382643   0.64768854]
 ...
```
(Exact random numbers depend on the seed but will be consistent across runs.)

</details>

### Array Properties and Reshaping

```python title="Array Properties and Shape Manipulation"
import numpy as np

a = np.random.randn(2, 3, 4)  # a 3D tensor

print(f"Shape: {a.shape}")      # (2, 3, 4)
print(f"Dimensions: {a.ndim}")  # 3
print(f"Total elements: {a.size}")  # 24
print(f"Data type: {a.dtype}")  # float64

# Reshaping — the total number of elements must stay the same
b = np.arange(12)
print(f"\\nOriginal: {b}")
print(f"Reshaped to (3,4):\\n{b.reshape(3, 4)}")
print(f"Reshaped to (2,2,3):\\n{b.reshape(2, 2, 3)}")

# -1 means "infer this dimension"
print(f"Reshape with -1: {b.reshape(3, -1).shape}")  # (3, 4)

# Flatten to 1D
c = np.array([[1, 2], [3, 4]])
print(f"Flattened: {c.flatten()}")  # [1, 2, 3, 4]

# Transpose
print(f"\\nTranspose:\\n{c.T}")
```

:::tip[Line-by-Line Walkthrough]
- **`a = np.random.randn(2, 3, 4)`** — Creates a 3D block of random numbers shaped like 2 layers, each with 3 rows and 4 columns. Imagine a small cube of data.
- **`a.shape`** — Reports the dimensions: `(2, 3, 4)`.
- **`a.ndim`** — Tells you how many dimensions the array has (3 in this case).
- **`a.size`** — Total number of elements: 2 × 3 × 4 = 24.
- **`a.dtype`** — The data type of each element (e.g., `float64`).
- **`b = np.arange(12)`** — Creates a 1D array `[0, 1, 2, ..., 11]`.
- **`b.reshape(3, 4)`** — Rearranges the 12 elements into 3 rows and 4 columns. The total number of elements must stay the same.
- **`b.reshape(3, -1)`** — The `-1` tells NumPy to figure out the missing dimension automatically. Since we have 12 elements and 3 rows, NumPy infers 4 columns.
- **`c.flatten()`** — Squashes a 2D array back into a single flat list.
- **`c.T`** — Transposes the array: rows become columns and columns become rows.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** Install NumPy: `pip install numpy`

**Steps:**
1. Save the code to a file, e.g. `array_shapes.py`
2. Run: `python array_shapes.py`

**Expected output:**
```
Shape: (2, 3, 4)
Dimensions: 3
Total elements: 24
Data type: float64

Original: [ 0  1  2  3  4  5  6  7  8  9 10 11]
Reshaped to (3,4):
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
...
```

</details>

### Indexing and Slicing

```python title="Indexing and Slicing"
import numpy as np

a = np.array([[10, 20, 30, 40],
              [50, 60, 70, 80],
              [90, 100, 110, 120]])

# Basic indexing: a[row, col]
print(f"Element (0,0): {a[0, 0]}")   # 10
print(f"Element (1,2): {a[1, 2]}")   # 70

# Slicing: a[start:end, start:end]
print(f"First row: {a[0, :]}")         # [10, 20, 30, 40]
print(f"First column: {a[:, 0]}")      # [10, 50, 90]
print(f"Submatrix:\\n{a[0:2, 1:3]}")   # [[20,30],[60,70]]

# Boolean indexing — extremely useful for filtering data
print(f"Elements > 50: {a[a > 50]}")

# Fancy indexing
rows = [0, 2]
cols = [1, 3]
print(f"Fancy indexing: {a[rows, cols]}")  # [20, 120]
```

:::tip[Line-by-Line Walkthrough]
- **`a = np.array([[10, 20, 30, 40], ...])`** — Creates a 3×4 table of numbers.
- **`a[0, 0]`** — Grabs the element in row 0, column 0 (top-left corner): 10.
- **`a[1, 2]`** — Row 1, column 2: 70.
- **`a[0, :]`** — The `:` means "everything." So this grabs all columns from row 0: the entire first row.
- **`a[:, 0]`** — All rows from column 0: the entire first column.
- **`a[0:2, 1:3]`** — A rectangular slice: rows 0–1, columns 1–2. Like cutting a rectangle out of the table.
- **`a[a > 50]`** — Boolean indexing. Creates a True/False mask for every element, then keeps only the ones where the condition is True. Like a filter in a spreadsheet.
- **`a[rows, cols]`** — Fancy indexing: picks specific elements by providing lists of row and column indices. `a[[0, 2], [1, 3]]` grabs element (0,1) and element (2,3).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `indexing.py` and run: `python indexing.py`

**Expected output:**
```
Element (0,0): 10
Element (1,2): 70
First row: [10 20 30 40]
First column: [10 50 90]
Submatrix:
[[20 30]
 [60 70]]
Elements > 50: [ 60  70  80  90 100 110 120]
Fancy indexing: [ 20 120]
```

</details>

### Vectorized Operations and Broadcasting

This is where NumPy's power really shines. **Vectorized operations** apply element-wise without explicit Python loops, running in optimized C code behind the scenes.

```python title="Vectorized Operations"
import numpy as np
import time

# Element-wise operations
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(f"a + b = {a + b}")       # [11, 22, 33, 44]
print(f"a * b = {a * b}")       # [10, 40, 90, 160]
print(f"a ** 2 = {a ** 2}")     # [1, 4, 9, 16]
print(f"np.sqrt(a) = {np.sqrt(a)}")

# Speed comparison: vectorized vs. Python loop
size = 1_000_000
x = np.random.randn(size)

start = time.time()
result_loop = [xi ** 2 + 2 * xi + 1 for xi in x]
loop_time = time.time() - start

start = time.time()
result_vec = x ** 2 + 2 * x + 1
vec_time = time.time() - start

print(f"\\nPython loop: {loop_time:.4f}s")
print(f"NumPy vectorized: {vec_time:.4f}s")
print(f"Speedup: {loop_time / vec_time:.0f}x")
```

:::tip[Line-by-Line Walkthrough]
- **`a + b`, `a * b`, `a ** 2`** — Element-wise operations. Each element in `a` is paired with the corresponding element in `b` and the operation is applied. Like adding two columns in a spreadsheet cell by cell.
- **`np.sqrt(a)`** — Takes the square root of every element, all at once.
- **`size = 1_000_000`** — Creates one million random numbers for a speed test. The underscores are just for readability (Python ignores them).
- **`[xi ** 2 + 2 * xi + 1 for xi in x]`** — A Python list comprehension that loops through each number one at a time. This is the slow way.
- **`x ** 2 + 2 * x + 1`** — The NumPy vectorized version does the same math but on the entire array at once, behind the scenes in optimized C code. This is the fast way.
- **`loop_time / vec_time`** — Compares the two speeds. NumPy is typically 10–100× faster.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `vectorized.py` and run: `python vectorized.py`

**Expected output:**
```
a + b = [11 22 33 44]
a * b = [ 10  40  90 160]
a ** 2 = [ 1  4  9 16]
np.sqrt(a) = [1.  1.41421356  1.73205081  2.]

Python loop: 0.3200s
NumPy vectorized: 0.0030s
Speedup: 107x
```
(Exact timings will vary by machine, but the speedup should be dramatic.)

</details>

:::warning[Avoid Python Loops Over NumPy Arrays]
If you find yourself writing a `for` loop that iterates over elements of a NumPy array, stop. There is almost certainly a vectorized way to do it that will be 10–100x faster. This habit is essential for ML code, where operations over millions of parameters must be fast.
:::

**Broadcasting** allows NumPy to perform operations on arrays of different shapes by automatically expanding the smaller array:

```python title="Broadcasting"
import numpy as np

# Scalar broadcast: adds 10 to every element
a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"a + 10:\\n{a + 10}")

# Vector broadcast: adds a different value to each column
col_means = np.array([10, 20, 30])  # shape (3,)
print(f"\\na + col_means:\\n{a + col_means}")

# Broadcasting rules:
# 1. Dimensions are compared from the trailing (rightmost) end
# 2. Dimensions are compatible if they're equal or one of them is 1
# 3. Missing dimensions are treated as 1

# Example: normalizing data (subtract mean, divide by std per column)
data = np.random.randn(100, 4)  # 100 samples, 4 features
means = data.mean(axis=0)       # shape (4,) — mean of each column
stds = data.std(axis=0)         # shape (4,)
normalized = (data - means) / stds  # broadcasting handles shape mismatch

print(f"\\nBefore normalization — means: {data.mean(axis=0).round(2)}")
print(f"After normalization — means: {normalized.mean(axis=0).round(2)}")
print(f"After normalization — stds: {normalized.std(axis=0).round(2)}")
```

:::tip[Line-by-Line Walkthrough]
- **`a + 10`** — Scalar broadcast: NumPy adds 10 to every single element in the array. It "stretches" the single number to match the array's shape.
- **`a + col_means`** — Vector broadcast: `col_means` has shape (3,) and `a` has shape (2, 3). NumPy repeats the row `[10, 20, 30]` for each of the 2 rows, adding different values to each column.
- **`data = np.random.randn(100, 4)`** — 100 samples, each with 4 features. Like a spreadsheet with 100 rows and 4 columns.
- **`data.mean(axis=0)`** — Computes the average of each column (down the rows). `axis=0` means "collapse the rows."
- **`data.std(axis=0)`** — Standard deviation per column — a measure of how spread out each column's values are.
- **`(data - means) / stds`** — Normalization: for each column, subtract its average and divide by its spread. After this, every column has mean ≈ 0 and spread ≈ 1. Broadcasting handles the fact that `data` is 100×4 but `means` and `stds` are just (4,).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `broadcasting.py` and run: `python broadcasting.py`

**Expected output:**
```
a + 10:
[[11 12 13]
 [14 15 16]]

a + col_means:
[[11 22 33]
 [14 25 36]]

Before normalization — means: [...]
After normalization — means: [ 0.  0. -0.  0.]
After normalization — stds: [1. 1. 1. 1.]
```

</details>

### Aggregation and Linear Algebra

```python title="Aggregation and Linear Algebra"
import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])

# Aggregation along axes
print(f"Sum all: {a.sum()}")           # 21
print(f"Sum cols (axis=0): {a.sum(axis=0)}")  # [5, 7, 9]
print(f"Sum rows (axis=1): {a.sum(axis=1)}")  # [6, 15]
print(f"Mean: {a.mean()}")
print(f"Max per row: {a.max(axis=1)}")
print(f"Argmax per row: {a.argmax(axis=1)}")  # index of max

# Linear algebra
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"\\nMatrix multiply (A @ B):\\n{A @ B}")
print(f"Determinant: {np.linalg.det(A):.1f}")
print(f"Inverse:\\n{np.linalg.inv(A)}")

eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
```

:::tip[Line-by-Line Walkthrough]
- **`a.sum()`** — Adds up every element in the entire array (21).
- **`a.sum(axis=0)`** — Sums down each column: `[1+4, 2+5, 3+6]` = `[5, 7, 9]`.
- **`a.sum(axis=1)`** — Sums across each row: `[1+2+3, 4+5+6]` = `[6, 15]`.
- **`a.argmax(axis=1)`** — Finds the *position* (index) of the largest value in each row, not the value itself.
- **`A @ B`** — Matrix multiplication. Each element in the result is a dot product of a row from A with a column from B. This is the core operation in neural networks.
- **`np.linalg.det(A)`** — Computes the determinant: a single number that tells you whether a matrix is invertible (non-zero) or not.
- **`np.linalg.inv(A)`** — Finds the inverse matrix: the matrix that "undoes" A, so A × A⁻¹ = identity.
- **`np.linalg.eig(A)`** — Computes eigenvalues and eigenvectors: the special directions where multiplying by A just stretches (no rotation).
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `linalg.py` and run: `python linalg.py`

**Expected output:**
```
Sum all: 21
Sum cols (axis=0): [5 7 9]
Sum rows (axis=1): [ 6 15]
Mean: 3.5
Max per row: [3 6]
Argmax per row: [2 2]

Matrix multiply (A @ B):
[[19 22]
 [43 50]]
Determinant: -2.0
...
```

</details>

## Pandas: Data Manipulation

Pandas builds on NumPy to provide labeled, tabular data structures. It's indispensable for loading, cleaning, and exploring datasets before feeding them to models.

### DataFrames

```python title="Creating and Exploring DataFrames"
import pandas as pd
import numpy as np

# Create from a dictionary
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'score': [92.5, 85.0, 78.3, 95.1, 88.7],
    'passed': [True, True, False, True, True],
})
print(df)
print(f"\\nShape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\\n{df.dtypes}")

# Quick statistics
print(f"\\nDescribe:\\n{df.describe()}")
print(f"\\nInfo:")
df.info()
```

:::tip[Line-by-Line Walkthrough]
- **`import pandas as pd`** — Imports the Pandas library with the standard short name `pd`.
- **`pd.DataFrame({...})`** — Creates a table (DataFrame) from a dictionary. Each key becomes a column name, and each value list becomes the column data. Think of it as building a spreadsheet from scratch.
- **`df.shape`** — Reports the table size as (rows, columns).
- **`df.columns`** — Lists all column names.
- **`df.dtypes`** — Shows the data type of each column (int, float, bool, object/string).
- **`df.describe()`** — Generates summary statistics (count, mean, std, min, max, quartiles) for all numeric columns. A one-line health check on your data.
- **`df.info()`** — Prints column types, non-null counts, and memory usage. Useful for spotting missing data.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install pandas numpy`

**Steps:**
1. Save to `dataframes.py` and run: `python dataframes.py`

**Expected output:**
```
      name  age  score  passed
0    Alice   25   92.5    True
1      Bob   30   85.0    True
2  Charlie   35   78.3   False
3    Diana   28   95.1    True
4      Eve   32   88.7    True

Shape: (5, 4)
...
```

</details>

### Filtering, Selection, and Sorting

```python title="Filtering and Selection"
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'student': [f'Student_{i}' for i in range(10)],
    'math': np.random.randint(50, 100, 10),
    'science': np.random.randint(50, 100, 10),
    'english': np.random.randint(50, 100, 10),
})

# Select columns
print("Math scores:", df['math'].values)

# Boolean filtering
high_math = df[df['math'] > 80]
print(f"\\nHigh math students:\\n{high_math}")

# Multiple conditions: use & (and), | (or), ~ (not)
# Parentheses are required!
strong = df[(df['math'] > 70) & (df['science'] > 70)]
print(f"\\nStrong in both:\\n{strong}")

# Sorting
sorted_df = df.sort_values('math', ascending=False)
print(f"\\nSorted by math (desc):\\n{sorted_df.head()}")

# Adding computed columns
df['average'] = df[['math', 'science', 'english']].mean(axis=1)
df['grade'] = pd.cut(df['average'], bins=[0, 60, 70, 80, 90, 100],
                     labels=['F', 'D', 'C', 'B', 'A'])
print(f"\\nWith computed columns:\\n{df}")
```

:::tip[Line-by-Line Walkthrough]
- **`df['math'].values`** — Selects the `math` column and returns it as a raw NumPy array.
- **`df[df['math'] > 80]`** — Boolean filtering: keeps only rows where the math score is above 80. Like applying a filter in Excel.
- **`df[(df['math'] > 70) & (df['science'] > 70)]`** — Combines two conditions with `&` (AND). Parentheses are required because of Python's operator precedence.
- **`df.sort_values('math', ascending=False)`** — Sorts all rows by the math column, highest first.
- **`.head()`** — Shows only the first 5 rows (useful for large datasets).
- **`df[['math', 'science', 'english']].mean(axis=1)`** — Computes the average across columns for each row (each student's average).
- **`pd.cut(...)`** — Bins a continuous variable into categories. Here it turns numeric averages into letter grades (F through A) based on the specified boundaries.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install pandas numpy`

**Steps:**
1. Save to `filtering.py` and run: `python filtering.py`

**Expected output:**
```
Math scores: [87 79 ... ]

High math students:
   student  math  science  english
...
```

</details>

### GroupBy and Aggregation

```python title="GroupBy Operations"
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'department': np.random.choice(['Engineering', 'Marketing', 'Sales'], 100),
    'experience': np.random.choice(['Junior', 'Mid', 'Senior'], 100),
    'salary': np.random.normal(80000, 20000, 100).astype(int),
    'satisfaction': np.random.uniform(1, 10, 100).round(1),
})

# Basic groupby
dept_stats = df.groupby('department').agg({
    'salary': ['mean', 'median', 'std'],
    'satisfaction': 'mean',
})
print("Department statistics:")
print(dept_stats)

# Multi-level groupby
cross = df.groupby(['department', 'experience'])['salary'].mean().unstack()
print(f"\\nAverage salary by department × experience:\\n{cross}")

# Value counts
print(f"\\nDepartment distribution:\\n{df['department'].value_counts()}")
```

:::tip[Line-by-Line Walkthrough]
- **`np.random.choice(['Engineering', 'Marketing', 'Sales'], 100)`** — Randomly picks one of the three departments for each of the 100 rows.
- **`np.random.normal(80000, 20000, 100)`** — Generates 100 salaries centered around $80,000 with a spread of $20,000 (bell curve).
- **`df.groupby('department')`** — Groups all rows by department. Think of it as sorting papers into piles by department, then doing calculations on each pile.
- **`.agg({'salary': ['mean', 'median', 'std'], 'satisfaction': 'mean'})`** — Applies different summary functions to different columns within each group.
- **`df.groupby(['department', 'experience'])['salary'].mean().unstack()`** — Groups by two columns and pivots the result into a cross-tabulation table, showing average salary for every department × experience combination.
- **`df['department'].value_counts()`** — Counts how many rows belong to each department. Like tallying a survey.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install pandas numpy`

**Steps:**
1. Save to `groupby.py` and run: `python groupby.py`

**Expected output:**
```
Department statistics:
                salary                  satisfaction
                  mean  median       std         mean
department
Engineering      ...
Marketing        ...
Sales            ...
...
```

</details>

### Handling Missing Data

```python title="Handling Missing Data"
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, 3, 4, 5],
})

print(f"Original:\\n{df}")
print(f"\\nMissing values per column:\\n{df.isnull().sum()}")

# Drop rows with any missing values
print(f"\\nDrop NaN rows:\\n{df.dropna()}")

# Fill missing values
print(f"\\nFill with 0:\\n{df.fillna(0)}")
print(f"\\nFill with column mean:\\n{df.fillna(df.mean())}")

# Forward fill (useful for time series)
print(f"\\nForward fill:\\n{df.fillna(method='ffill')}")
```

:::tip[Line-by-Line Walkthrough]
- **`np.nan`** — Represents a missing value. Like an empty cell in a spreadsheet.
- **`df.isnull().sum()`** — Counts how many missing values exist in each column. Your first diagnostic step with any dataset.
- **`df.dropna()`** — Removes every row that has at least one missing value. Simple but can lose a lot of data.
- **`df.fillna(0)`** — Replaces all missing values with 0. Quick but may not make sense for every column.
- **`df.fillna(df.mean())`** — Replaces each missing value with the average of its column. A smarter default because it preserves the column's overall distribution.
- **`df.fillna(method='ffill')`** — Forward fill: copies the last known value downward into the gap. Especially useful for time series data where yesterday's value is a reasonable guess for today.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install pandas numpy`

**Steps:**
1. Save to `missing_data.py` and run: `python missing_data.py`

**Expected output:**
```
Original:
     A    B  C
0  1.0  NaN  1
1  2.0  2.0  2
2  NaN  3.0  3
3  4.0  NaN  4
4  5.0  5.0  5

Missing values per column:
A    1
B    2
C    0
...
```

</details>

:::tip[Pandas for ML Preprocessing]
In a typical ML workflow, you'll use Pandas to: load CSV/parquet files, explore the data, handle missing values, encode categorical variables, split into train/test sets, and then convert to NumPy arrays or PyTorch tensors for model training.
:::

## Matplotlib: Visualization

Visualization is how you *understand* your data and *diagnose* your models. Matplotlib is Python's foundational plotting library. While libraries like Seaborn and Plotly offer higher-level APIs, understanding Matplotlib gives you full control.

### Line Plots

```python title="Line Plots — Training Curves"
import numpy as np
# import matplotlib.pyplot as plt

# Simulating a training curve
epochs = np.arange(1, 51)
train_loss = 2.0 * np.exp(-0.05 * epochs) + 0.1 + np.random.normal(0, 0.02, 50)
val_loss = 2.0 * np.exp(-0.04 * epochs) + 0.2 + np.random.normal(0, 0.03, 50)

# In a real notebook, you'd use:
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, train_loss, label='Training Loss', color='blue')
# plt.plot(epochs, val_loss, label='Validation Loss', color='red', linestyle='--')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training vs Validation Loss')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

print("Training loss (first 5):", train_loss[:5].round(3))
print("Val loss (first 5):", val_loss[:5].round(3))
print("Final train loss:", train_loss[-1].round(4))
print("Final val loss:", val_loss[-1].round(4))
```

:::tip[Line-by-Line Walkthrough]
- **`epochs = np.arange(1, 51)`** — Creates an array `[1, 2, 3, ..., 50]` representing 50 training epochs (passes through the data).
- **`2.0 * np.exp(-0.05 * epochs) + 0.1 + ...`** — Simulates a training loss curve: starts high and exponentially decays toward 0.1, with small random noise added to look realistic.
- **`val_loss = 2.0 * np.exp(-0.04 * epochs) + 0.2 + ...`** — Simulates validation loss. It decays slower and settles higher (0.2 vs 0.1), mimicking the typical pattern where validation loss is slightly worse than training loss.
- **The commented `plt` lines** — If you uncomment these in a Jupyter notebook, they produce a line plot comparing training vs. validation loss over time.
- **`train_loss[:5].round(3)`** — Shows the first 5 loss values, rounded to 3 decimal places.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy` (add `matplotlib` if you uncomment the plotting lines)

**Steps:**
1. Save to `training_curve.py` and run: `python training_curve.py`

**Expected output:**
```
Training loss (first 5): [1.896 1.803 1.761 1.629 1.573]
Val loss (first 5): [1.937 1.87  1.737 1.686 1.694]
Final train loss: 0.1729
Final val loss: 0.3262
```
(Exact values will vary slightly due to random noise.)

</details>

### Scatter Plots

```python title="Scatter Plots — Feature Relationships"
import numpy as np
# import matplotlib.pyplot as plt

np.random.seed(42)

# Generate two-class data
class_0_x = np.random.normal(2, 0.8, 50)
class_0_y = np.random.normal(2, 0.8, 50)
class_1_x = np.random.normal(5, 0.8, 50)
class_1_y = np.random.normal(5, 0.8, 50)

# In a real notebook:
# plt.figure(figsize=(8, 8))
# plt.scatter(class_0_x, class_0_y, c='blue', label='Class 0', alpha=0.6)
# plt.scatter(class_1_x, class_1_y, c='red', label='Class 1', alpha=0.6)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Two-Class Classification Data')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

print(f"Class 0 centroid: ({class_0_x.mean():.2f}, {class_0_y.mean():.2f})")
print(f"Class 1 centroid: ({class_1_x.mean():.2f}, {class_1_y.mean():.2f})")
```

:::tip[Line-by-Line Walkthrough]
- **`np.random.normal(2, 0.8, 50)`** — Generates 50 random numbers centered at 2 with a spread of 0.8. Creates one coordinate (x or y) for 50 points in Class 0.
- **`np.random.normal(5, 0.8, 50)`** — Same thing, but centered at 5 for Class 1. The two clusters will be separated in space.
- **The commented `plt` lines** — Would create a scatter plot with blue dots for Class 0 and red dots for Class 1, showing two distinct clusters.
- **`class_0_x.mean()`** — Computes the centroid (average position) of each cluster. If the centroids are far apart, a classifier should be able to separate them easily.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `scatter.py` and run: `python scatter.py`

**Expected output:**
```
Class 0 centroid: (2.03, 1.97)
Class 1 centroid: (4.99, 5.11)
```

</details>

### Histograms and Subplots

```python title="Histograms and Subplots"
import numpy as np
# import matplotlib.pyplot as plt

np.random.seed(42)

# Different distributions
uniform = np.random.uniform(0, 1, 10000)
normal = np.random.normal(0, 1, 10000)
exponential = np.random.exponential(1, 10000)

# In a real notebook:
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#
# axes[0].hist(uniform, bins=50, color='steelblue', alpha=0.7)
# axes[0].set_title('Uniform Distribution')
#
# axes[1].hist(normal, bins=50, color='coral', alpha=0.7)
# axes[1].set_title('Normal Distribution')
#
# axes[2].hist(exponential, bins=50, color='seagreen', alpha=0.7)
# axes[2].set_title('Exponential Distribution')
#
# plt.tight_layout()
# plt.show()

for name, data in [('Uniform', uniform), ('Normal', normal), ('Exponential', exponential)]:
    print(f"{name}: mean={data.mean():.3f}, std={data.std():.3f}, "
          f"min={data.min():.3f}, max={data.max():.3f}")
```

:::tip[Line-by-Line Walkthrough]
- **`np.random.uniform(0, 1, 10000)`** — Generates 10,000 random numbers spread evenly between 0 and 1. Like rolling a perfectly fair die with infinite sides.
- **`np.random.normal(0, 1, 10000)`** — Generates 10,000 numbers from a bell curve centered at 0. Most values cluster near the center.
- **`np.random.exponential(1, 10000)`** — Generates 10,000 numbers from an exponential distribution. Lots of small values, a few very large ones. Models things like wait times.
- **The `for` loop at the end** — Prints summary statistics (mean, std, min, max) for each distribution so you can compare their characteristics without a plot.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install numpy`

**Steps:**
1. Save to `histograms.py` and run: `python histograms.py`

**Expected output:**
```
Uniform: mean=0.500, std=0.289, min=0.000, max=1.000
Normal: mean=0.002, std=1.001, min=-3.845, max=3.764
Exponential: mean=0.993, std=0.985, min=0.001, max=8.987
```

</details>

:::tip[Plotting in Practice]
In Jupyter notebooks and Google Colab, `matplotlib` plots render inline automatically. The code examples here include the matplotlib code as comments so you can uncomment them in your own notebook. In production code, you'll often use **Weights & Biases** or **TensorBoard** for experiment tracking instead of raw matplotlib.
:::

## Putting It Together: A Mini Data Analysis

Let's combine all three libraries in a realistic workflow:

```python title="End-to-End Data Analysis Pipeline"
import numpy as np
import pandas as pd

np.random.seed(42)

# Generate synthetic ML experiment data
n_experiments = 200
data = pd.DataFrame({
    'learning_rate': np.random.choice([0.001, 0.01, 0.1], n_experiments),
    'batch_size': np.random.choice([16, 32, 64, 128], n_experiments),
    'hidden_dim': np.random.choice([64, 128, 256, 512], n_experiments),
    'dropout': np.random.uniform(0, 0.5, n_experiments).round(2),
    'epochs': np.random.randint(10, 100, n_experiments),
})

# Simulate accuracy based on hyperparameters (with some noise)
data['accuracy'] = (
    0.7
    + 0.1 * (data['learning_rate'] == 0.01).astype(float)
    - 0.05 * (data['learning_rate'] == 0.1).astype(float)
    + 0.02 * np.log2(data['hidden_dim'] / 64)
    - 0.1 * data['dropout']
    + 0.001 * data['epochs']
    + np.random.normal(0, 0.03, n_experiments)
).clip(0.5, 0.99).round(4)

print("Dataset shape:", data.shape)
print(data.head(10))

# Analysis: best hyperparameters
print("\\n--- Best Configurations ---")
top5 = data.nlargest(5, 'accuracy')
print(top5[['learning_rate', 'batch_size', 'hidden_dim', 'accuracy']])

# GroupBy analysis
print("\\n--- Accuracy by Learning Rate ---")
lr_stats = data.groupby('learning_rate')['accuracy'].agg(['mean', 'std', 'max'])
print(lr_stats)

print("\\n--- Accuracy by Hidden Dim ---")
dim_stats = data.groupby('hidden_dim')['accuracy'].agg(['mean', 'std', 'max'])
print(dim_stats)

# Correlation matrix
print("\\n--- Correlations with Accuracy ---")
numeric_cols = ['learning_rate', 'batch_size', 'hidden_dim', 'dropout', 'epochs', 'accuracy']
correlations = data[numeric_cols].corr()['accuracy'].sort_values(ascending=False)
print(correlations)
```

:::tip[Line-by-Line Walkthrough]
- **`n_experiments = 200`** — We simulate 200 ML experiments, each with different hyperparameter settings.
- **`np.random.choice([0.001, 0.01, 0.1], n_experiments)`** — Randomly picks a learning rate for each experiment from three options.
- **`data['accuracy'] = (...).clip(0.5, 0.99)`** — Simulates accuracy as a function of the hyperparameters with some noise added. `.clip()` keeps values between 0.5 and 0.99 (realistic range).
- **`data.nlargest(5, 'accuracy')`** — Finds the 5 experiments with the highest accuracy. Like sorting by accuracy and taking the top 5.
- **`data.groupby('learning_rate')['accuracy'].agg(['mean', 'std', 'max'])`** — Groups experiments by learning rate and computes the average, spread, and best accuracy for each group.
- **`data[numeric_cols].corr()['accuracy']`** — Computes the correlation of each feature with accuracy. Values near +1 mean "as this goes up, accuracy goes up"; near −1 means the opposite; near 0 means no relationship.
:::

<details>
<summary><b>How to Run This Code</b></summary>

**Prerequisites:** `pip install pandas numpy`

**Steps:**
1. Save to `analysis_pipeline.py` and run: `python analysis_pipeline.py`

**Expected output:**
```
Dataset shape: (200, 6)
   learning_rate  batch_size  hidden_dim  dropout  epochs  accuracy
0          0.001          64         256     0.39      96    0.8417
...

--- Best Configurations ---
...

--- Accuracy by Learning Rate ---
               mean       std    max
learning_rate
0.001         ...
0.010         ...
0.100         ...
...
```

</details>

## Exercises

:::tip[Exercise 1: NumPy Array Gymnastics — beginner]

1. Create a 4x5 matrix containing integers from 1 to 20
2. Compute the mean of each row and the sum of each column
3. Find the maximum element and its position (row, column)
4. Normalize the matrix so it has mean 0 and standard deviation 1

<details>
<summary>Hints</summary>

1. Use np.arange() to create sequences.
2. Use .reshape() to change dimensions.
3. axis=0 operates along columns, axis=1 along rows.

</details>

:::

:::tip[Exercise 2: Broadcasting Challenge — intermediate]

Without using any Python loops, compute the **pairwise Euclidean distance matrix** between two sets of points:
- Set A: 5 points in 3D (shape: 5×3)
- Set B: 4 points in 3D (shape: 4×3)
- Result should be a 5×4 matrix where element (i,j) is the distance between point A[i] and B[j]

Use broadcasting to do this in a single expression.

<details>
<summary>Hints</summary>

1. Euclidean distance: sqrt(sum((a - b)^2))
2. Use broadcasting to avoid nested loops.
3. Reshape A to (n, 1, d) and B to (1, m, d) to compute all pairwise differences.

</details>

:::

:::tip[Exercise 3: Pandas Data Cleaning — intermediate]

Create a messy dataset with 1000 rows and the following issues, then clean it:

1. Some missing values in numeric columns (fill with column median)
2. Duplicate rows (remove them)
3. An 'age' column with some negative values (replace with NaN, then fill)
4. A 'category' column with inconsistent casing ('Cat', 'cat', 'CAT' should all be 'cat')

Write the cleaning pipeline and verify each step.

<details>
<summary>Hints</summary>

1. Use pd.read_csv() or create synthetic data.
2. Check missing values with df.isnull().sum().
3. Use fillna(), drop_duplicates(), and astype() for cleaning.

</details>

:::

:::tip[Exercise 4: Softmax Implementation — intermediate]

Implement the **softmax function** from scratch using NumPy:

:::info[Plain English: What Is Softmax?]
Imagine you have a handful of scores—like confidence ratings for each possible answer. Softmax converts those raw scores into percentages that all add up to 100%. A high score gets a big share, and a low score gets a tiny share. It's like dividing up a pie where the slices are proportional to each score's "enthusiasm."
:::

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

**Reading the formula:** $x_i$ is the raw score for option $i$. $e^{x_i}$ raises the mathematical constant $e$ (≈ 2.718) to the power of that score, which makes bigger scores *much* bigger (exaggerating differences). The bottom part, $\sum_j e^{x_j}$, adds up those exaggerated values for *all* options. Dividing gives each option its share of the total, producing a number between 0 and 1. All the results sum to exactly 1, so you can treat them as probabilities.

Requirements:
1. Must be numerically stable (handle large values without overflow)
2. Must work on both 1D arrays and 2D arrays (batches, applying softmax per row)
3. Verify that the outputs sum to 1

<details>
<summary>Hints</summary>

1. Softmax: exp(x_i) / sum(exp(x_j))
2. Subtract the max for numerical stability.
3. Your implementation should work on batches (2D arrays) along axis=1.

</details>

:::

:::tip[Exercise 5: Data Visualization Dashboard — advanced]

Using the Iris dataset (available via `sklearn.datasets.load_iris()`), create a 2×2 grid of plots:

1. **Top-left**: Scatter plot of sepal length vs. sepal width, colored by species
2. **Top-right**: Histogram of petal lengths for each species (overlaid)
3. **Bottom-left**: Box plot of all four features
4. **Bottom-right**: Bar chart of mean feature values per species

Add proper titles, labels, legends, and a consistent color scheme.

<details>
<summary>Hints</summary>

1. Use plt.subplots() to create a grid of plots.
2. Try fig, axes = plt.subplots(2, 2, figsize=(12, 10)).
3. Use different plot types: scatter, histogram, line, bar.

</details>

:::

## Resources

- **[NumPy User Guide](https://numpy.org/doc/stable/user/)** _(tutorial)_ — The official NumPy documentation. Comprehensive reference with excellent examples.

- **[Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)** _(book)_ by Jake VanderPlas — Free online book covering NumPy, Pandas, Matplotlib, and scikit-learn in depth.

- **[Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html)** _(tutorial)_ — Official Pandas tutorials. Start with '10 minutes to pandas' for a quick overview.

- **[Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)** _(tutorial)_ — Official Matplotlib tutorials covering basic to advanced plotting techniques.

- **[From Python to NumPy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)** _(book)_ by Nicolas Rougier — Deep dive into NumPy's internals and performance optimization. Read after mastering the basics.

---

**Next up**: With our math and Python foundations in place, we're ready to understand what machine learning actually *is* and how it works at a conceptual level.
