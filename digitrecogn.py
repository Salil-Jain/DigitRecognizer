from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import glob

def dist(x,y):
	return np.sqrt(np.sum((x-y)**2))


digits = datasets.load_digits()
# The dataset has images of size 8 X 8

# fig = plt.figure()
# plt.subplot(221)
# plt.imshow(digits.images[0],cmap = plt.cm.gray_r)
# plt.subplot(222)
# plt.imshow(digits.images[41],cmap = plt.cm.gray_r)
# plt.subplot(223)
# plt.imshow(digits.images[157],cmap = plt.cm.gray_r)
# plt.subplot(224)
# plt.imshow(digits.images[289],cmap = plt.cm.gray_r)
# txt = "We have numbers %d, %d, %d, %d at 10, 2, 8, and 4 o'clock"%(digits.target[0],digits.target[41],digits.target[157],digits.target[289],)
# fig.text(0,0,txt)
# plt.show()

#Preditction part

# print(len(digits.images)) 
# Total length is 1797

x = 100

X_train = digits.data[0:x]
Y_train = digits.target[0:x]

# print len(X_train)
# # shows 10

# print len(Y_train)
# # shows 10

# Choose a test image
# pred = 813
# X_test = digits.data[pred]
# print "X_test's real value is %d"%digits.target[pred]


# # # Uncomment below snippet to plot the number
# # plt.figure()
# # plt.imshow(digits.images[pred],cmap = plt.cm.gray_r)
# # plt.show()


# # Running Nearest Neighbour Classifier
# l = len(X_train)
# distance = np.zeros(l) #This will store the distance of test from every training value
# for i in range(l):
# 	distance[i] = dist(X_train[i],X_test)
# min_index = np.argmin(distance)
# print "Preditcted value"
# print(Y_train[min_index])

# # At this point you can see how increasing number of test data increases accuracy. Predict
# # for pred = 345 and you will get wring result if i = 10 and right result if i = 100

# # Now lets find number of wrong results

# l = len(X_train)
# no_err = 0
# distance = np.zeros(l)
# for j in range(1697,1797):
# 	X_test = digits.data[j]
# 	for i in range(l):
# 		distance[i] = dist(X_train[i],X_test)
# 	min_index = np.argmin(distance)
# 	if Y_train[min_index] != digits.target[j]:
# 		no_err+=1
# print "Total errors for train length = %d is %d"%(x,no_err)
# # As you increase the value of x you will see that the error decreases

# print X_train[1]
for i in xrange(len(X_train)):
	maximum = np.amax(X_train[i])
	X_train[i]/=maximum

# Custom images
image = misc.imread("test.png",flatten = 1)
image1 = 255 - image
image2 = np.reshape(image1,64)
# print image2

X_test = image2
maximum = np.amax(X_test)
X_test/=maximum
# Running Nearest Neighbour Classifier
l = len(X_train)
distance = np.zeros(l) #This will store the distance of test from every training value
for i in range(l):
	distance[i] = dist(X_train[i],X_test)
min_index = np.argmin(distance)
# print X_test
print "Preditcted value"
print(Y_train[min_index])