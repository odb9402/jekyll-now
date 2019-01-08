---
layout: post
title: What if neural network does not have activation function
---

## Activation function
 Activation function is a function that would adapted for neurons in neural network after matrix multiplication with layer weights. Well, everyone who has a focus about neural networks might already know what activation function is and what kind of activation function is widely used. But why our network essentially need it? Or even is it possible my network could work without these activation functions?
 
{:refdef: style="text-align: center;"}
![ActivationFunc](https://github.com/odb9402/odb9402.github.io/blob/master/images/activation_func.JPG?raw=true)
{: refdef}

## What happen in my network?
{:refdef: style="text-align: center;"}
![NetworkEx](https://github.com/odb9402/odb9402.github.io/blob/master/images/network_example.JPG?raw=true)
{: refdef}

Let assume a neural network has 1 hidden layer ( layer 1 ). Each edge has a weight value which is just a real number scalar, and each neuron also has a bias value. The sizes of our network are $i,j,k$ for each layer including an input layer.

Let X is an input data, Y is a output and W, W` are layers in our neural network, our notation is :

$$X = \langle x_0,...,x_i \rangle\\
W = \begin{bmatrix}
w_{00} & \cdots & w_{0j}\\
\vdots & \ddots & \vdots\\
w_{i0} & \cdots & w_{ij}\\
\end{bmatrix}
\\
B = \langle b_0, \cdots, b_j \rangle
\\
W' = \begin{bmatrix}
w'_{00} & \cdots & w'_{0i}\\
\vdots & \ddots & \vdots\\
w'_{k0} & \cdots & w'_{kj}\\
\end{bmatrix}
\\
B' = \langle b'_0, \cdots, b'_j \rangle
\\
Y=(W'(WX+B)+B')$$

It make sense and we might already have seen for a long time since we met a concept of neural network or similar feed forward networks. . . Except for activation function. If use activation function $f(X)$ after multiplication, the output $Y$ is : $Y=f(W'(f(WX) +B)) +B'$.  Note that each neuron in neural network without activation function $f$, it is just repetition of multiplications and additions.

### It Is The Affine Transformation.
I`m a noob and novice in linear algebra, but I know the **matrix-vector multiplication is a linear transform** of a vector. And linear transform is, moving my vector with some other linear space that has basis with row vectors in the matrix. For instance, we have an image in 2-Dimensional space, the transformation is just scaling up for some directions.  An Image of IU, a most adorable Korean singer, will be linearly transformed:
![BeautifulIU](https://github.com/odb9402/odb9402.github.io/blob/master/images/linear_transformation_IU.JPG?raw=true)
Moreover, our activation-less neural network model has one another operation, adding bias. But it is just a movement of the origin. Yep. So matrix multiplication for a vector is a scaling and adding bias is a changing origin with this 2-D space. And, as like we consider matrix multiplication as linear transform, **matrix multiplication and adding bias is an affine transformation**.


## Can never go deeper with only affine transformations
The interest thing is, whatever how much time it is transformed, the transformations can be done with merely single affine transformation. In terms of adding bias, we can recognize it from that no matter how much time we add  $(a_i,b_i)$ for 2 dimensional vector $(x,y)$ -> $(x+a_0+...+a_n, y+b_0+...+b_n)$, it can be represented $(x+a',y+b')$ where $a_i,b_i,a',b'\in\mathbb{R}$. In the case of transformation is described similarly, a number of multiplications with $\alpha_j,\beta_j$ also can be represented by single multiplication with $\alpha',\beta'$.  What dose it tell us? **Without activation function, our deep deep network is always same with just a neural network with two layer including input and output no matter how deep the network is.**

![Network](https://github.com/odb9402/odb9402.github.io/blob/master/images/nn_without_activation.JPG?raw=true)

Even if our network converts a dimension of our input data by different neuron numbers for each layer, the problem does not change. The transformation with a matrix that is not square, is also linear transformation. For example, when above the IU picture is linearly or affinely transformed with 2\*3 matrix, it will be converted 3-dimensional vector but its shape ( flat picture ) does not change in 3 dimensional space. Conversely, with 2\*1 matrix, it will be converted 1-dimensional vector but it is just a concept of shade (projection). The essence, lots of transformation can be replaced by just one transformation does not change.


### Linear? Nonlinear?
Therefore, we need something else that **Nonlinerity** to escape this limitation. In simple, any system that is not a linear is nonlinear. Even ReLU is a non-linear activation function although it looks absolutely linear for positive domain. As explained earlier, neural network without an activation function is a linear. And most of our complex real world problems are not linear. If our network does not have any activation function or even has an activation function that just a linear function such as $f(X)=\alpha X + \beta$, **the network never make nonlinear function** because $f$ is still linear.  **In conclusion, if we want to imitate some real world function with non-linearity, we must use a non-linear activation function to give our network non-linearity.** 


 
