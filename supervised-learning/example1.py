#!/usr/bin/env python3

"""
Suppose we have a 3-dimensional input x = (x1,x2,x3) and weights w=(w1,w2,w3) where:

  x1 = 2                    w1 = 1
  x2 = -1                   w2 = -0.5
  x3 = 1                    w3 = 0

  and the bias is b = 0.5. Furthermore, suppose the target for x is t = 1

 If our output is given by y = x(Transpose) times w + b (linear), and we are using the squared loss (defined as 1 / 2 (y - t)^2 ), what is the value of the loss incurred by this example?
"""

import numpy as np

def main():
    bias = 0.5
    inputs = np.array([ 2,-1,1 ])
    weights = np.array([ 1,-0.5,0 ])
    target = 1

    output = sum(inputs.transpose() * weights) + bias

    loss = 1/2 * ((output - target) ** 2)

    print("Calculated squared loss is {}".format(loss))

if __name__ == '__main__':
    main()
