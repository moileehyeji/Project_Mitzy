#중앙차분 구현 
f = 0
x = 0

import numpy as np
def gradient(f, x):
    h = 0.0001  
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]

        # f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

lr = 0.01
momentum = 0.9
v = 0 

def update(lr, momentum, v):
    grads = gradient(f, x)
    v = momentum *v - lr * grads
    x += v