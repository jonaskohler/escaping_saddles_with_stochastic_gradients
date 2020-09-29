
import torch
import numpy as np
from torch.autograd import Variable
import sys
import torch.nn.functional as func
import pickle

n_points=int(sys.argv[1])   #first arg is the number of points (parameters w) to be assessed.
no_of_hl=int(sys.argv[2])   #second arg is the number of hidden layers
HUs=2

hidden_layers=np.ones(no_of_hl,dtype=int)*HUs  


dtype = torch.FloatTensor # Uncomment this to run on CPU
#dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU



def forward_pass(n,batch): # this is a recursive way to implement the foward pass .. 
        if n==0:
            return torch.sigmoid(batch.mm(weight_list[n]))
        else:
            return torch.sigmoid(forward_pass(n-1,batch).mm(weight_list[n]))


def phi_prime(t):
    return torch.sigmoid(t)*(1-torch.sigmoid(t))
def phi_prime_prime(t):
    return torch.sigmoid(t)*(1-torch.sigmoid(t))*(1-2*torch.sigmoid(t))
def get_hessian(grad_params,weight_list,no_of_hl,layers,d ):
        Hessian=torch.zeros((d,d)).type(dtype)
        row=0
        col=0

        for h in range(no_of_hl+1): #for all weight matrices (layers)
            for i in range(layers[h]): #for all rows (input dim)
                for j in range (layers[h+1]): # for all columns (output dim)
                    for k in range(no_of_hl+1): #for all weight matrices (other + self)
                        ###compute derivative,i.e. 
                        der_k=torch.autograd.grad(grad_params[h][i,j], weight_list[k], create_graph=True) #take first derivative of layer h at i,j and derive w.r.t. layer k

                        der_k=der_k[0].view(1,(layers[k]*layers[k+1]).item()) #flatten

                        assert ((layers[k]*layers[k+1]).item()==der_k.size()[1]), 'dimension mismatch'

                        Hessian[row,col:col+layers[k]*layers[k+1]]=der_k[0].view(1,(layers[k]*layers[k+1]).item()).data #write the flattened second derivate into row of hessian matrix
                        col=col+layers[k]*layers[k+1]
                    row=row+1
                    col=0 
        w_1=weight_list[0][:,0]
        sigma_1=weight_list[1][0]
        r=(y-y_pred).sum()
        first_quarter=torch.zeros((D_in,D_in)).type(dtype)
        for i in range(N):
            a=0
            b=0
            a=-2*sigma_1*phi_prime(torch.dot(w_1,x[i]))*sigma_1*phi_prime(torch.dot(w_1,x[i]))
            b=-2*r*sigma_1*phi_prime_prime(torch.dot(w_1,x[i]))
            first_quarter=first_quarter+ (a.data+b.data)[0]*torch.ger(x[i],x[i]).data
        first_quarter=first_quarter/N
        import IPython ; IPython.embed()


        return Hessian 

## 1. Load data
MNIST=False

if MNIST:

    X=np.load('mnist_10by10_X.npy') 
    Y=np.load('mnist_10by10_y.npy') 

    x=Variable(torch.from_numpy(X).type(dtype), requires_grad=False)
    y=Variable(torch.from_numpy(Y).type(dtype), requires_grad=False)
    loss_fn = torch.nn.CrossEntropyLoss(size_average=True)


    y=y.long() #this for cross entropy loss
    N=X.shape[0]
    D_in=X.shape[1]
    D_out=len(np.unique(Y))


else:
    # Create random input and output data
    N=10
    x = Variable(torch.randn(N, 3).type(dtype), requires_grad=False)
    y = Variable(torch.randn(N, 1).type(dtype), requires_grad=False)
    D_in=3
    D_out=1
    #y=y.long() #this for cross entropy loss  






_layers=np.append(hidden_layers,[D_out])
layers=np.append([D_in], _layers) #this variable contains the Network arcitecture in terms of units in each layer,e.g. 5,10,10,1 (D_in,hidden 1, hidden 2, D_out)

print('Network architecture (no of units in layer): ',layers)

#compute total number of trainable weights
d=0
for l in range(no_of_hl+1):
    d=d+layers[l]*layers[l+1]
d=d.item()

#some lists to save results
SGD_positive_covariance_collector=[]
SGD_negative_covariance_collector=[]

uniform_positive_covariance_collector=[]
uniform_negative_covariance_collector=[]

EV_collector=[]


counter=0
weight_list=[None]*(no_of_hl+1) #list of trainable weight by layer: [w1,w2,w3,w4], where,e.g., w1 is the first weight matrix(!) connecting input and first HL



for t in range(n_points): #for all random points (number of experiments)

    ### 2. Initialize weight matrices randomly 
    for m in range(no_of_hl+1):
        weight_list[m]= Variable(torch.randn(layers[m].item(), layers[m+1].item()).type(dtype), requires_grad=True)

    ### 3. Forward pass:
    #specify loss
    
    # predict y (for all N) recursively by multiplying the matrices and appling sigmoid activations (function call). Append last linear layer (explicitly in next line)
    y_pred=forward_pass(no_of_hl-1,x).mm(weight_list[no_of_hl])

    #Compute loss (for all N)
    if MNIST:
        loss = loss_fn(y_pred,y)
    else:
        loss = (y_pred - y).pow(2).sum()

    ### 4. Backward pass: 
    loss.backward(retain_graph=True)

    # Safe the gradient and graph
    grad_params=torch.autograd.grad(loss, weight_list, create_graph=True)

    print(t, loss.data[0])

    ### 5. Compute the Hessian and SVD
    Hessian=get_hessian(grad_params,weight_list,no_of_hl,layers,d)

    SVD=torch.symeig(Hessian,eigenvectors=True)
    eigVals=SVD[0]
    eigVecs=SVD[1]

    #the list of Eigvals is sorted in increasing order. find the last negative and first positive eigenvalue index
    if (eigVals>0).any():
      index_first_positive_EV=(eigVals > 0).nonzero()[0]
    else:
      index_first_positive_EV=d

    if (eigVals==0).any():
      index_first_zero_EV=(eigVals == 0).nonzero()[0] # this could be simplified by checking >= instead
    else:
      index_first_zero_EV=index_first_positive_EV


    EV_collector.append(eigVals.cpu())
    
    #leftmost_eigenvec_torch=eigVecs[:,0]

    ### 6. Now, compute the stochastic gradients

    gradient_matrix_indiv=torch.zeros((N,d)).type(dtype)
    sum_loss=0
    for i in range(N): #for each datapoint
        x_indiv=x[i].view(1,D_in) #sample datapoint
        y_indiv=forward_pass(no_of_hl-1,x_indiv).mm(weight_list[no_of_hl]) #do an individual forward pass
        loss_indiv = loss_fn(y_indiv,y[i]) #compute individual loss
        sum_loss= sum_loss+loss_indiv #sum (this is just to check) 
        #get stochastic gradient nabla f_i
        grad_indiv = torch.autograd.grad(loss_indiv, weight_list, create_graph=True) 
        grad_flat_indiv=torch.zeros((1,d)).type(dtype)
        col=0
        #... and flatten it
        for h in range(no_of_hl+1): #for all weight matrices
            grad_flat_indiv[0][col:col+layers[h]*layers[h+1]]=grad_indiv[h].view(1,(layers[h]*layers[h+1]).item()).data
            col=col+layers[h]*layers[h+1]
        #Finally, store the individual gradient in nxd matrix. 
        gradient_matrix_indiv[i]=grad_flat_indiv
    
    ### 7. Compute the covariance of SGD
    #normalize the gradients and eigenvectors:
    gradient_matrix_indiv= func.normalize(gradient_matrix_indiv,dim=1)
    eigVecs=func.normalize(eigVecs,dim=0)

    # 7.1 compute covariances

    cov_matrix=gradient_matrix_indiv.mm(eigVecs)     
    covariances=torch.sum(cov_matrix*cov_matrix,dim=0)/N # this vector constains in each column_j: 1/N sum_i (nabla f_i^T v_j)^2
    del cov_matrix #remove from memory
    del gradient_matrix_indiv

    #split the covariances depending on negative or positive eigenvalues
    negative_covariances=covariances[0:int(index_first_zero_EV[0])]
    positive_covariances=covariances[int(index_first_positive_EV[0]):d]
    SGD_negative_covariance_collector.append(negative_covariances.cpu())  #save results (cpu readable)
    SGD_positive_covariance_collector.append(positive_covariances.cpu())  


    ### 8. Compute uniform noise from Ball
    uniform_noise_matrix=torch.zeros((N,d)).type(dtype)
    for j in range(N):
        u=torch.Tensor(1,d).uniform_(0,1).type(dtype)
        z=torch.randn(d).type(dtype)
        z_norm=torch.norm(z)
        uniform_ball_vec=u**(1/d)*z/z_norm
        uniform_noise_matrix[j]=uniform_ball_vec
    # 8.1Compute the covariance of ball noise
    uniform_noise_matrix= func.normalize(uniform_noise_matrix,dim=1)

    cov_matrix_uniform_noise=uniform_noise_matrix.mm(eigVecs)
    covariances_uniform_noise=torch.sum(cov_matrix_uniform_noise*cov_matrix_uniform_noise,dim=0)/N
    del cov_matrix_uniform_noise
    del uniform_noise_matrix
    negative_covariances_uniform_noise=covariances_uniform_noise[0:int(index_first_positive_EV[0])]
    positive_covariances_uniform_noise=covariances_uniform_noise[int(index_first_positive_EV[0]):d]

    uniform_negative_covariance_collector.append(negative_covariances_uniform_noise.cpu())  #save results (cpu readable)
    uniform_positive_covariance_collector.append(positive_covariances_uniform_noise.cpu())  
   

    for i in range(no_of_hl+1):
        weight_list[i].grad.data.zero_() #zero gradient data for next round. (needed by pytroch. god know why...)

### 9. save all files 
file_name="results/EVs_"+str(no_of_hl)+"HL.txt"
with open(file_name, "wb") as fp:   #Pickling
   pickle.dump(EV_collector, fp)

torch.save(SGD_positive_covariance_collector,'results/SGD_cov_'+str(no_of_hl)+'HL_POS')
torch.save(SGD_negative_covariance_collector,'results/SGD_cov_'+str(no_of_hl)+'HL_NEG')
torch.save(uniform_positive_covariance_collector,'results/uniform_cov_'+str(no_of_hl)+'HL_POS')
torch.save(uniform_negative_covariance_collector,'results/uniform_cov_'+str(no_of_hl)+'HL_NEG')
