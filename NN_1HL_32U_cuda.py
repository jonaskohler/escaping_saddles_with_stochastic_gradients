
# coding: utf-8

# In[240]:

import torch
import numpy as np
from torch.autograd import Variable
import sys

n_steps=int(sys.argv[1])

learning_rate = float(sys.argv[2]) 


#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

#get_ipython().magic('matplotlib inline')

def get_hessian(grad_params,w1,w2):
       Hessian=torch.zeros((d,d))
       row=0

       # 2nd derivative of first layer
       for i in range(D_in):
           for j in range (H1):
               if H1>1:
                   der_w1=torch.autograd.grad(grad_params[0][i,j], w1, create_graph=True)
                   der_w2=torch.autograd.grad(grad_params[0][i,j], w2, create_graph=True)
                   
               else:
                   der_w1=torch.autograd.grad(grad_params[0][i], w1, create_graph=True)
                   der_w2=torch.autograd.grad(grad_params[0][i], w2, create_graph=True)

                 
               der_w1=der_w1[0].view(1,D_in*H1)
               der_w2=der_w2[0].view(1,H1*D_out)
               Hessian[row,:]=torch.cat((der_w1,der_w2),1).data
               row=row+1

       # 2nd derivative of second layer 
       for i in range(H1):
           for j in range(D_out):
               if D_out>1:
                   der_w1=torch.autograd.grad(grad_params[1][i,j], w1, create_graph=True)
                   der_w2=torch.autograd.grad(grad_params[1][i,j], w2, create_graph=True)
                   
               else:
                   der_w1=torch.autograd.grad(grad_params[1][i], w1, create_graph=True)
                   der_w2=torch.autograd.grad(grad_params[1][i], w2, create_graph=True)

               der_w1=der_w1[0].view(1,D_in*H1)
               der_w2=der_w2[0].view(1,H1*D_out)
               Hessian[row,:]=torch.cat((der_w1,der_w2),1).data
               row=row+1
                   
       return Hessian




# Real data

H1 = 32

X=np.load('mnist_10by10_X.npy')
Y=np.load('mnist_10by10_y.npy')

x=Variable(torch.from_numpy(X).type(dtype), requires_grad=False)
y=Variable(torch.from_numpy(Y).type(dtype), requires_grad=False)
#y=y.view(70000,1) <- this for softmax loss
y=y.long() #this for cross entropy loss

N=X.shape[0]
D_in=X.shape[1]
D_out=len(np.unique(Y))

d=D_in*H1+H1*D_out


#set initial weights
#w1 = Variable(torch.randn(D_in, H1).type(dtype), requires_grad=True)
#w2 = Variable(torch.randn(H1, D_out).type(dtype), requires_grad=True)

w1 = Variable(torch.zeros(D_in, H1).type(dtype), requires_grad=True)
w2 = Variable(torch.zeros(H1, D_out).type(dtype), requires_grad=True)

#clone weights
w1_0=w1.data.clone()
w2_0=w2.data.clone()

### GRADIENT DESCENT


# Create random Tensors for weights, and wrap them in Variables.

loss_collector=[]
grad_collector=[]
EV_collector=[]

#reset w_0 to restart from the same point!!!
w1.data=w1_0.clone()
w2.data=w2_0.clone()

loss_fn = torch.nn.CrossEntropyLoss(size_average=True)


for t in range(n_steps):
    # Forward pass: 
    #y_pred = x.mm(w1).clamp(min=0).mm(w2).clamp(min=0).mm(w3)  #RELU activation
    y_pred=torch.sigmoid(x.mm(w1)).mm(w2) #SIGMOID activation

    # Compute and print loss using operations on Variables.
    loss = loss_fn(y_pred,y)
    #loss = (y_pred - y).pow(2).sum()/(N)

    loss_collector.append(loss.data[0])

    ### compute gradients   
    loss.backward(retain_graph=True)
    # safe the gradient and graph
    grad_params=torch.autograd.grad(loss, (w1,w2), create_graph=True)
    
    if t%25==0:
        print(t, loss.data[0])
        Hessian=get_hessian(grad_params,w1,w2)

        numpy_hessian= Hessian.numpy()

        ev_min=min(np.linalg.eigvals(numpy_hessian))
        ev_max=max(np.linalg.eigvals(numpy_hessian))
        EV_collector.append((ev_min,ev_max))


    # Update weights using gradient descent;
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data


    w_1_grad_flat=w1.grad.data.view(1,D_in*H1)
    w_2_grad_flat=w2.grad.data.view(1,H1*D_out)
    grad_flat=torch.cat((w_1_grad_flat,w_2_grad_flat),1)
    grad_collector.append(torch.norm(grad_flat))
    
    # Manually zero the gradients after updating weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()
EV_collector=np.array(EV_collector)

np.save('loss_1HL_32U',loss_collector)
np.save('grads_1HL_32U',grad_collector)
np.save('EVs_1HL_32U', EV_collector)
