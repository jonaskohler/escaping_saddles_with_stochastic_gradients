###################################
### Stochastic Gradient Descent ###
###################################

# Authors: Aurelien Lucchi and Jonas Kohler, 2017

from datetime import datetime
import numpy as np

def GD(w, loss, gradient,hessian, X=None, Y=None, opt=None, **kwargs):
    print ('--- GD ---')
    n = X.shape[0]
    d = X.shape[1]

    print_progress= opt.get('print_progress',True)    
    n_epochs = opt.get('n_epochs_gd', 100)
    eta = opt.get('learning_rate_gd',1e-1)
    batch_size =n

    print("- Batch size:" , batch_size)
    print("- Learning rate:" , eta)

    n_steps = n_epochs

    loss_collector=[]
    timings_collector=[]
    samples_collector=[]
    steps_collector=[]
    grad_norm_collector=[]
    EV_collector=np.zeros((n_epochs+1,2))


    _loss = loss(w, X, Y, **kwargs) 
    loss_collector.append(_loss)
    timings_collector.append(0)
    samples_collector.append(0)
    steps_collector.append(w)

    start = datetime.now()
    timing=0
     #Safe full grad norm
    #grad_norm=np.linalg.norm(gradient(w, X, Y,**kwargs))
    #grad_norm_collector.append(grad_norm)

    #H= hessian(w, X, Y,**kwargs)
    #ev_min=min(np.linalg.eigvals(H))
    #ev_max=max(np.linalg.eigvals(H))

    #EV_collector[0]=(ev_min,ev_max)

    for i in range(n_steps):


        # II: compute step
        grad = gradient(w, X, Y,**kwargs)  
        grad_norm=np.linalg.norm(grad)
        grad_norm_collector.append(grad_norm)

        H= hessian(w, X, Y,**kwargs)
        ev_min=min(np.linalg.eigvals(H))
        ev_max=max(np.linalg.eigvals(H))
        EV_collector[i]=(ev_min,ev_max)
        
        w = w - eta * grad
   
        _timing=timing
        timing=(datetime.now() - start).total_seconds()
        
        _loss = loss(w, X, Y, **kwargs)

        if print_progress:
            print ('Epoch ' + str(i) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(np.linalg.norm(grad)), 'time=',round(timing-_timing,3))

        timings_collector.append(timing)
        samples_collector.append((i+1)*batch_size)
        loss_collector.append(_loss)
        steps_collector.append(w)

    #append final gradient norm

    grad = gradient(w, X, Y,**kwargs)  
    grad_norm=np.linalg.norm(grad)
    grad_norm_collector.append(grad_norm)
    H= hessian(w, X, Y,**kwargs)
    ev_min=min(np.linalg.eigvals(H))
    ev_max=max(np.linalg.eigvals(H))
    EV_collector[n_epochs]=(ev_min,ev_max)
    

    return w, timings_collector, loss_collector, samples_collector,np.array(steps_collector),grad_norm_collector,EV_collector