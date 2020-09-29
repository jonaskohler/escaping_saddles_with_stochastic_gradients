
# coding: utf-8

# In[240]:

import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
matplotlib.use('Agg')


dtype = torch.FloatTensor

#get_ipython().magic('matplotlib inline')

def get_hessian(grad_params,w1,w2,w3):
       Hessian=torch.zeros((d,d))
       row=0

       # 2nd derivative of first layer
       for i in range(D_in):
           for j in range (H1):
               if H1>1:
                   der_w1=torch.autograd.grad(grad_params[0][i,j], w1, create_graph=True)
                   der_w2=torch.autograd.grad(grad_params[0][i,j], w2, create_graph=True)
                   der_w3=torch.autograd.grad(grad_params[0][i,j], w3, create_graph=True)
                   
               else:
                   der_w1=torch.autograd.grad(grad_params[0][i], w1, create_graph=True)
                   der_w2=torch.autograd.grad(grad_params[0][i], w2, create_graph=True)
                   der_w3=torch.autograd.grad(grad_params[0][i], w3, create_graph=True)

                 
               der_w1=der_w1[0].view(1,D_in*H1)
               der_w2=der_w2[0].view(1,H1*H2)
               der_w3=der_w3[0].view(1,H2*D_out)
               Hessian[row,:]=torch.cat((der_w1,der_w2,der_w3),1).data
               row=row+1

       # 2nd derivative of second layer 
       for i in range(H1):
           for j in range(H2):
               if H2>1:
                   der_w1=torch.autograd.grad(grad_params[1][i,j], w1, create_graph=True)
                   der_w2=torch.autograd.grad(grad_params[1][i,j], w2, create_graph=True)
                   der_w3=torch.autograd.grad(grad_params[1][i,j], w3, create_graph=True)
                   
               else:
                   der_w1=torch.autograd.grad(grad_params[1][i], w1, create_graph=True)
                   der_w2=torch.autograd.grad(grad_params[1][i], w2, create_graph=True)
                   der_w3=torch.autograd.grad(grad_params[0][i], w3, create_graph=True)

                   
               der_w1=der_w1[0].view(1,D_in*H1)
               der_w2=der_w2[0].view(1,H1*H2)
               der_w3=der_w3[0].view(1,H2*D_out)
               Hessian[row,:]=torch.cat((der_w1,der_w2,der_w3),1).data
               row=row+1
                   
       # 2nd derivative of third layer 
       for i in range(H2):
           for j in range(D_out):
               if D_out>1:
                   der_w1=torch.autograd.grad(grad_params[2][i,j], w1, create_graph=True)
                   der_w2=torch.autograd.grad(grad_params[2][i,j], w2, create_graph=True)
                   der_w3=torch.autograd.grad(grad_params[2][i,j], w3, create_graph=True)

               else:
                   der_w1=torch.autograd.grad(grad_params[2][i], w1, create_graph=True)
                   der_w2=torch.autograd.grad(grad_params[2][i], w2, create_graph=True)
                   der_w3=torch.autograd.grad(grad_params[2][i], w3, create_graph=True)
                
               der_w1=der_w1[0].view(1,D_in*H1)
               der_w2=der_w2[0].view(1,H1*H2)
               der_w3=der_w3[0].view(1,H2*D_out)
               Hessian[row,:]=torch.cat((der_w1,der_w2,der_w3),1).data
               row=row+1
       return Hessian



# Real data

H1,H2, D_out = 10, 5, 1

X=np.load('mnist_10by10_X.npy')
Y=np.load('mnist_10by10_y.npy')

x=Variable(torch.from_numpy(X).type(dtype), requires_grad=False)
y=Variable(torch.from_numpy(Y).type(dtype), requires_grad=False)
y=y.view(70000,1)

D_in=X.shape[1]
N=X.shape[0]

d=D_in*H1+H1*H2+H2*D_out

#set initial weights
w1 = Variable(torch.randn(D_in, H1).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H1, H2).type(dtype), requires_grad=True)
w3 = Variable(torch.randn(H2, D_out).type(dtype), requires_grad=True)
#clone weights
w1_0=w1.data.clone()
w2_0=w2.data.clone()
w3_0=w3.data.clone()

### GRADIENT DESCENT

learning_rate = 1e-3
n_steps=100

# Create random Tensors for weights, and wrap them in Variables.

loss_collector=[]
grad_collector=[]
EV_collector=[]

#reset w_0 to restart from the same point!!!
w1.data=w1_0.clone()
w2.data=w2_0.clone()
w3.data=w3_0.clone()

for t in range(n_steps):
    # Forward pass: 
    #y_pred = x.mm(w1).clamp(min=0).mm(w2).clamp(min=0).mm(w3)  #RELU activation
    y_pred=torch.sigmoid(torch.sigmoid(x.mm(w1)).mm(w2)).mm(w3) #SIGMOID activation

    # Compute and print loss using operations on Variables.
    #loss = loss_fn(y_pred,y)

    #import IPython ; IPython.embed() ; exit(1)
   
    loss = (y_pred - y).pow(2).sum()/(N)
    loss= np.dot(y_pred-y,y_pred-y)/N


    loss_collector.append(loss.data[0])

    ### compute gradients   
    loss.backward(retain_graph=True)
    # safe the gradient and graph
    grad_params=torch.autograd.grad(loss, (w1,w2,w3), create_graph=True)
    
    if t%10==0:
        print(t, loss.data[0])
        Hessian=get_hessian(grad_params,w1,w2,w3)

        numpy_hessian= Hessian.numpy()

        ev_min=min(np.linalg.eigvals(numpy_hessian))
        ev_max=max(np.linalg.eigvals(numpy_hessian))
        EV_collector.append((ev_min,ev_max))


    # Update weights using gradient descent;
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data
    w3.data -= learning_rate * w3.grad.data


    w_1_grad_flat=w1.grad.data.view(1,D_in*H1)
    w_2_grad_flat=w2.grad.data.view(1,H1*H2)
    w_3_grad_flat=w3.grad.data.view(1,H2*D_out)
    grad_flat=torch.cat((w_1_grad_flat,w_2_grad_flat,w_3_grad_flat),1)
    grad_collector.append(torch.norm(grad_flat))
    
    # Manually zero the gradients after updating weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    w3.grad.data.zero_()
EV_collector=np.array(EV_collector)

fig = plt.figure()
plt.subplot(1, 3, 1)
plt.plot(np.arange(len(loss_collector)),loss_collector)
plt.title('LOSS')
plt.subplot(1, 3, 2)
plt.plot(np.arange(len(grad_collector)),grad_collector)
plt.title('|GRAD|')

plt.subplot(1, 3, 3)

plt.plot(np.arange(len(EV_collector)), EV_collector[:,0])
plt.plot(np.arange(len(EV_collector)), EV_collector[:,1],':')
plt.title('|EVs|')


plt.subplots_adjust(left=4, right=6)

savefig('NN.pdf')

# ### 1) Newton

# In[145]:

### Compute hessian

