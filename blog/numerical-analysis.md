# Identifying yield using numpy analysis

Data Analysis refers to analysis of numerical data to make effective decisions in a more scientific way.. The [Numpy](https://numpy.org) library supports large multi-dimensional arrays and matrices operations and it also contains a large number of hight-level mathematical functions for working with numbers.


> If you have climate data like the temperature, rainfall, and humidity and you want to determine if
  a region is well suited for growing a certain fruit, a simple approach would be to use a linear equation and formulate
  the relationship between the annual yield of the fruit (tons per hectare) and the climatic conditions
  like the average temperature (in degrees Fahrenheit), rainfall (in  millimeters) & average relative
  humidity (in percentage).

```
yield_of_fruits = w1 * temperature + w2 * rainfall + w3 * humidity
```

We can express the yield of fruits as the weighted sum of the temperature, rainfall, and humidity. This equation is only an approximation since the actual relationship may not necessarily be linear, and there may be other factors involved.  A simple linear model like this often works well when you need a rough estimate.

Based on some statical analysis of historical data, we might come up with reasonable values for the weights `w1`, `w2`, and `w3`. some sample data:

```
w1, w2, w3 = 0.3, 0.2, 0.5
```

Given some climate data for a region, we can now predict the yield of fruits. Here's some sample data:

<img src="https://github.com/samueldsingh/numpy-analysis/blob/master/numpy_analysis.png?raw=true" style="width:360px;">

To begin, we can define some variables to record climate data for a region.

```
region1_temp = 73
region2_rainfall = 67
region3_humidity = 43
```

The variables can now be substituted into the linear equation to predict the yield of fruits.

```
crop_yield(region1, weights)
print("The expected yield of apples in region1 is {} tons per hectare.".format(region1_yield_apples))
```

```
Output:
The expected yield of apples in Kanto region is 56.8 tons per hectare.
```

## Dot product (multiplication) of two vectors

Convert the lists into numpy arrays in order to compute the dot product of the two vectors using the Numpy library.

Next, let's import the `numpy` module with the alias `np`.

```
import numpy as np
```

We can now use the `np.array` function to create Numpy arrays:

```
kanto = np.array([73, 67, 43])
kanto
```

```
array([73, 67, 43])
```

Compute the dot product of the two vectors using the `np.dot` function.

```
np.dot(region1, weights)
```

The output is:
```
56.8
```

## Multi-dimensional Numpy arrays 

Represent the climate data for all the regions using a single 2-dimensional Numpy array.

```
climate_data = np.array([[73, 67, 43],
                         [91, 88, 64],
                         [87, 134, 58],
                         [102, 43, 37],
                         [69, 96, 70]])
climate_data
```

```
The result is: array([[ 73,  67,  43],
                      [ 91,  88,  64],
                      [ 87, 134,  58],
                      [102,  43,  37],
                      [ 69,  96,  70]])
```

The above `2-d array` is a matrix with five rows and three columns. Each row represents one region, and the columns represent temperature, rainfall, and humidity, respectively.

```
# 2D array (matrix)
climate_data.shape
```

```
(5, 3)
```

```
weights
```

```
array([0.3, 0.2, 0.5])
```

All the elements in a numpy array have the same data type. You can check the data type of an array using the `.dtype` property.

```
weights.dtype
```

The output is:
```
dtype('int64')
```

```
climate_data.dtype
```

```
dtype('int64')
```

## Matrix multiplication
We can use the ```np.matmul``` function or the ```@``` operator to perform matrix multiplication.

```
np.matmul(climate_data, weights)
array([56.8, 76.9, 81.9, 57.7, 74.9])
```

Using `@` operator:
```
climate_data @ weights
array([56.8, 76.9, 81.9, 57.7, 74.9])
```

## Working with CSV data files

Numpy also provides helper functions reading from & writing to files. Let's download a file `climate.txt`, which contains 10,000 climate measurements (temperature, rainfall & humidity) for the five regions in the following format:


```
temperature,rainfall,humidity
25.00,76.00,99.00
39.00,65.00,70.00
59.00,45.00,77.00
84.00,63.00,38.00
66.00,50.00,52.00
41.00,94.00,77.00
91.00,57.00,96.00
49.00,96.00,99.00
67.00,20.00,28.00
...
```

This format of storing data is known as *comma-separated values* or CSV. 

> **CSVs**: A comma-separated values (CSV) file is a delimited text file that uses a comma to separate values. Each line of the file is a data record. Each record consists of one or more fields, separated by commas. A CSV file typically stores tabular data (numbers and text) in plain text, in which case each line will have the same number of fields. (Wikipedia)


To read this file into a numpy array, we can use the `genfromtxt` function.

```
import urllib.request

urllib.request.urlretrieve(
    'https://gist.github.com/BirajCoder/a4ffcb76fd6fb221d76ac2ee2b8584e9/raw/4054f90adfd361b7aa4255e99c2e874664094cea/climate.csv', 
    'climate.txt')
    
('climate.txt', <http.client.HTTPMessage at 0x7f5ca0e5ab90>)
```

```
climate_data = np.genfromtxt('climate.txt', delimiter=',', skip_header=1)
```

```
climate_data
```

```
array([[25., 76., 99.],
       [39., 65., 70.],
       [59., 45., 77.],
       ...,
       [99., 62., 58.],
       [70., 71., 91.],
       [92., 39., 76.]])
```

```
climate_data.shape
(10000, 3)
```

We can now perform a matrix multiplication using the `@` operator to predict the yield of apples for the entire dataset using a given set of weights.

```
weights = np.array([0.3, 0.2, 0.5])
```

```
yields = climate_data @ weights
yields
```

```
array([72.2, 59.7, 65.2, ..., 71.1, 80.7, 73.4])
```

Let's add the `yields` to `climate_data` as a fourth column using the [`np.concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html) function.

```
climate_results = np.concatenate((climate_data, yields.reshape(10000, 1)), axis=1)
climate_results

array([[25. , 76. , 99. , 72.2],
       [39. , 65. , 70. , 59.7],
       [59. , 45. , 77. , 65.2],
       ...,
       [99. , 62. , 58. , 71.1],
       [70. , 71. , 91. , 80.7],
       [92. , 39. , 76. , 73.4]])
```

There are a couple of subtleties here:

- Since we wish to add new columns, we pass the argument ```axis=1``` to np.concatenate. The axis argument specifies the dimension for concatenation.

- The arrays should have the same number of dimensions, and the same length along each except the dimension used for concatenation. We use the ```np.reshape``` function to change the shape of yields from ```(10000,)``` to ```(10000,1)```.

Let's write the final results from our computation above back to a file using the `np.savetxt` function.

```
climate_results

array([[25. , 76. , 99. , 72.2],
       [39. , 65. , 70. , 59.7],
       [59. , 45. , 77. , 65.2],
       ...,
       [99. , 62. , 58. , 71.1],
       [70. , 71. , 91. , 80.7],
       [92. , 39. , 76. , 73.4]])
```

```
np.savetxt('climate_results.txt', 
           climate_results, 
           fmt='%.2f', 
           delimiter=',',
           header ='temperature,rainfall,humidity,yeild_apples',
           comments='')
```
