''' 
 This script basically follows this tutorial: 
   http://scipy.github.io/old-wiki/pages/Tentative_NumPy_Tutorial.html
 
 Other useful links:
   http://mathesaurus.sourceforge.net/r-numpy.html
   http://scipy.github.io/old-wiki/pages/Numpy_Example_List.html
'''

import numpy as np

# Create a range array
a = np.arange(15)

# Change the array properties (reshape creates a shallow copy)
a = a.reshape(3, 5)

# Print some important information
print(a)
print(a.shape)
print(a.ndim)
print(a.dtype.name)
print(a.itemsize)
print(a.size)
print(type(a))

# Create another array from a python array
b = np.array([6, 7, 8])
print(b.data[0] == b[0])

# Specify the type
c = np.array([[1, 2, 3], [3, 4, 5]], dtype='float32')
print(c)
print(c.dtype)

# Other constructors
print(np.zeros((2, 3)))
print(np.ones((1, 3), dtype='float'))
print(np.empty((2, 4)))
print(np.arange(2, 7, 0.1))
print(np.linspace(1, 2, 9))
print(np.random.random((2, 3)))

# Basic operations
a = np.array([20, 30, 40, 50])
b = np.arange(4)
print(a + b)
print(a.sum())
print(a ** 2)
print(a < 35)
print(a * b)
print(np.dot(a, b))
print(a.sum())
print(dir(a))

# Working with axis
a = np.arange(12).reshape(3, 4)
print(a)
print(a.sum(axis=0))
print(a.min(axis=1))
print(a.cumsum(axis=1))

# Accessing the data
a = np.arange(20)
print(a[1:10:2])
print(a[-1])
print(a[::-1])
print([x ** 2 for x in a if x > 3])

# Initialize using a function
def f(x, y):
    return x + y
a = np.fromfunction(f, (4, 5))
print(a)
print(a[1, 2:4])
print(a[1, ...])

# Modifying the shape of an array
print(a.ravel())
a.shape = (10, 2)
print(a)
print(a.reshape(5, -1))

# Slicing returns a shallow copy!!
a = np.arange(20).reshape(4, 5)
print(a)
b = a[:, 1:3]
b[:] = 10
print(a)

# Deep copies
b = a.copy()
b[:] = 20
print(b[0, 0] != a[0, 0])

# Use of argmin
a = np.array([1, 2, 0, 2, 5, 3]).reshape(2, 3)
print(a[1].argmin())

