###################################
### Perturbed Gradient Descent ###
###################################

# Authors: Aurelien Lucchi and Jonas Kohler, 2017

from datetime import datetime
import numpy as np

def PGD_2(w, loss, gradient,hessian, X=None, Y=None, opt=None, **kwargs):
    print ('--- PGD ---')
    n = X.shape[0]
    d = X.shape[1]

    print_progress= opt.get('print_progress',True)    
    n_epochs = opt.get('n_epochs_pgd', 100)
    eta = opt.get('learning_rate_pgd',1e-1)
    r=opt.get('radius_pgd',1e-1)
    g_thres=opt.get('g_thres_pgd',1e-1)

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
    perturbations=0
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

        if grad_norm < g_thres and ev_min<-0.005:
            int_idx=np.random.randint(0, high=n, size=1)        
            bool_idx = np.zeros(n,dtype=bool)
            bool_idx[int_idx]=True
            _X=np.zeros((batch_size,d))
            _X=np.compress(bool_idx,X,axis=0)
            _Y=np.compress(bool_idx,Y,axis=0)
            # II: compute step
            grad_i = gradient(w, _X, _Y,**kwargs) 
            w = w - eta*2*grad_i
            perturbations+=1
        else:
            w= w- eta * grad 
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
    print('pertubs',perturbations)

    return w, timings_collector, loss_collector, samples_collector,np.array(steps_collector),grad_norm_collector,EV_collector