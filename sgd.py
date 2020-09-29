###################################
### Stochastic Gradient Descent ###
###################################

# Authors: Aurelien Lucchi and Jonas Kohler, 2017

from datetime import datetime
import numpy as np

def SGD(w, loss, gradient,hessian, X=None, Y=None, opt=None, **kwargs):
    print ('--- SGD ---')
    n = X.shape[0]
    d = X.shape[1]

    print_progress= opt.get('print_progress',True)    
    n_epochs = opt.get('n_epochs_sgd', 100)
    eta = opt.get('learning_rate_sgd',1e-1)
    batch_size =max(int(opt.get('batch_size_sgd',0.01*n)),1)

    print("- Batch size:" , batch_size)
    print("- Learning rate:" , eta)



    n_steps = int((n_epochs * n) / batch_size)
    n_samples_seen = 0  # number of samples processed so far

    loss_collector=[]
    timings_collector=[]
    samples_collector=[]
    steps_collector=[]
    grad_norm_collector=[]
    EV_collector=np.zeros((n_epochs+2,2))


    _loss = loss(w, X, Y, **kwargs) 
    loss_collector.append(_loss)
    timings_collector.append(0)
    samples_collector.append(0)
    steps_collector.append(w)

    start = datetime.now()
    timing=0
    k=0
     #Safe full grad norm
    grad_norm=np.linalg.norm(gradient(w, X, Y,**kwargs))
    grad_norm_collector.append(grad_norm)

    H= hessian(w, X, Y,**kwargs)
    ev_min=min(np.linalg.eigvals(H))
    ev_max=max(np.linalg.eigvals(H))

    EV_collector[0]=(ev_min,ev_max)


    for i in range(n_steps):

        # I: subsampling
        #int_idx=np.random.permutation(n)[0:batch_size]
        int_idx=np.random.randint(0, high=n, size=batch_size)        

        bool_idx = np.zeros(n,dtype=bool)
        bool_idx[int_idx]=True
        _X=np.zeros((batch_size,d))
        _X=np.compress(bool_idx,X,axis=0)
        _Y=np.compress(bool_idx,Y,axis=0)



        # II: compute step
        grad = gradient(w, _X, _Y,**kwargs)  

        n_samples_seen += batch_size
        w = w - eta * grad

        if (n_samples_seen >= n*k)  == True:
            _timing=timing
            timing=(datetime.now() - start).total_seconds()
            
            _loss = loss(w, X, Y, **kwargs)
             #Safe full grad norm
            grad_norm=np.linalg.norm(gradient(w, X, Y,**kwargs))
            grad_norm_collector.append(grad_norm)

            H= hessian(w, X, Y,**kwargs)
            ev_min=min(np.linalg.eigvals(H))
            ev_max=max(np.linalg.eigvals(H))
            EV_collector[k+1]=(ev_min,ev_max)
    
            if print_progress:
                print ('Epoch ' + str(k) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(np.linalg.norm(grad)), 'time=',round(timing-_timing,3))
            k+=1

            timings_collector.append(timing)
            samples_collector.append((i+1)*batch_size)
            loss_collector.append(_loss)
            steps_collector.append(w)

    return w, timings_collector, loss_collector, samples_collector,np.array(steps_collector),grad_norm_collector,EV_collector