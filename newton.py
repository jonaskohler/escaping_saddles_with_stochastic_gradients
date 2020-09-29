
###################################
### Newtons Method ###
###################################

# Authors: Aurelien Lucchi and Jonas Kohler, 2017

from datetime import datetime
import numpy as np

def NEWTON(w, loss, gradient,hessian ,X=None, Y=None, opt=None, **kwargs):
    print ('--- NEWTON ---')
    n = X.shape[0]
    d = X.shape[1]

    print_progress= opt.get('print_progress',True)    
    n_epochs = opt.get('n_epochs_newton', 100)
    eta = opt.get('learning_rate_newton',1)
    batch_size =max(int(opt.get('batch_size_newton',n)),1)

    print("- Batch size:" , batch_size)
    print("- Learning rate:" , eta)



    n_steps = int((n_epochs * n) / batch_size)
    n_samples_seen = 0  # number of samples processed so far

    loss_collector=[]
    timings_collector=[]
    samples_collector=[]
    steps_collector=[]
    grad_norm_collector=[]
    EV_collector=np.zeros((n_steps+1,2))

    _loss = loss(w, X, Y, **kwargs) 
    loss_collector.append(_loss)
    timings_collector.append(0)
    samples_collector.append(0)
    steps_collector.append(w)

    start = datetime.now()
    timing=0
    k=0

    for i in range(n_steps):

        # II: compute step
        grad = gradient(w, X, Y,**kwargs) 
        grad_norm_collector.append(np.linalg.norm(grad))

        H= hessian(w, X, Y,**kwargs)
        ev_min=min(np.linalg.eigvals(H))
        ev_max=max(np.linalg.eigvals(H))

        EV_collector[i]=(ev_min,ev_max)

        n_samples_seen += batch_size

        w = w - eta *np.dot(np.linalg.inv(H), grad)

        if (n_samples_seen >= n*k)  == True:
            _timing=timing
            timing=(datetime.now() - start).total_seconds()
            
            _loss = loss(w, X, Y, **kwargs)
            if print_progress:
                print ('Epoch ' + str(k) + ': loss = ' + str(_loss) + ' norm_grad = ' + str(np.linalg.norm(grad)), 'time=',round(timing-_timing,3))
            k+=1

            timings_collector.append(timing)
            samples_collector.append((i+1)*batch_size)
            loss_collector.append(_loss)
            steps_collector.append(w)

    #add final gradient
    grad = gradient(w, X, Y,**kwargs) 
    grad_norm_collector.append(np.linalg.norm(grad))
    H= hessian(w, X, Y,**kwargs)
    ev_min=min(np.linalg.eigvals(H))
    ev_max=max(np.linalg.eigvals(H))

    EV_collector[n_steps]=(ev_min,ev_max)
    return w, timings_collector, loss_collector, samples_collector,np.array(steps_collector), grad_norm_collector, EV_collector