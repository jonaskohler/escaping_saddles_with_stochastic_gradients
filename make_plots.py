# imports + functions
# %matplotlib qt
import matplotlib
import matplotlib.pyplot as plt
import simplejson
import sys


def preprocess(list_loss):
    # find overall min value
    min_value = 1000
    for k in range(len(list_loss)):
        min_value = min(list_loss[k]) if (min(list_loss[k]) <= min_value) else min_value

    # subtract min value and add epsilon
    eps = min_value * 1e-6
    for k in range(len(list_loss)):
        list_loss[k] = [i - min_value + eps for i in list_loss[k]]
    return list_loss


def two_d_plot_time(list_loss, list_x, list_params,list_grads, dataset_name, n, d, log_scale, x_limits=None):
    list_loss = preprocess(list_loss)
    colors = ['#1B2631', '#C0392B', '#9B59B6', '#2980B9', '#1E8449', '#27AE60', '#E67E22', '#95A5A6', '#FF97F2',
              '#34495E']
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    fig = plt.figure()
    plt.subplot(1, 2, 1)

    for i in range(len(list_loss)):
        plt.plot(list_x[i], list_loss[i], linestyles[i % 10], color=colors[i % 10], linewidth=3.0)
    plt.legend(list_params, fontsize=12, loc=1)

    if log_scale == True:
        plt.yscale('log')
        plt.ylabel('$\log(f-f^*)$', fontsize=12)
    else:
        plt.yscale('linear')
        plt.ylabel('$(f-f^*)$')
    plt.xlabel('time in seconds', fontsize=12)
    if not x_limits == None:
        plt.xlim(x_limits)
    plt.title(str(dataset_name) + ' (n=' + str(n) + ', d=' + str(d) + ')', fontsize=13)  # + ', lambda=1e-3)')

    plt.subplot(1, 2, 2)


    for i in range(len(list_loss)):
        plt.plot(list_x[i], list_grads[i], linestyles[i % 10], color=colors[i % 10], linewidth=3.0)
    plt.legend(list_params, fontsize=12, loc=1)

   
    plt.yscale('linear')
    plt.ylabel('$\|  grad \|$')
    plt.xlabel('time in seconds', fontsize=12)
    if not x_limits == None:
        plt.xlim(x_limits)
    plt.title(str(dataset_name) + ' (n=' + str(n) + ', d=' + str(d) + ')', fontsize=13)  # + ', lambda=1e-3)')

    plt.show


def two_d_plot_iterations(list_loss, list_x, list_params,list_grads, dataset_name, n, d, log_scale, x_limits=None):
    colors = ['#1B2631', '#C0392B', '#9B59B6', '#2980B9', '#1E8449', '#27AE60', '#E67E22', '#95A5A6', '#FF97F2',
              '#34495E']
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']

    fig = plt.figure()

    plt.subplot(1, 2, 1)

    for i in range(len(list_loss)):
        _x = []
        for k in range(len(list_x[i])):
            _x.append(k)
        list_x[i] = _x

        plt.plot(list_x[i], list_loss[i], linestyles[i % 10], color=colors[i % 10], linewidth=3.0)

    plt.legend(list_params, fontsize=12, loc=1)

    if log_scale == True:
        plt.yscale('log')
        plt.ylabel('$\log(f-f^*)$', fontsize=12)
    else:
        plt.yscale('linear')
        plt.ylabel('$(f-f^*)$')

    plt.xlabel('iteration', fontsize=12)
    if not x_limits == None:
        plt.xlim(x_limits)
    plt.title(str(dataset_name) + ' (n=' + str(n) + ', d=' + str(d) + ')', fontsize=13)  # + ', lambda=1e-3)')


    plt.subplot(1, 2, 2)


    for i in range(len(list_loss)):
        plt.plot(list_x[i], list_grads[i], linestyles[i % 10], color=colors[i % 10], linewidth=3.0)
    plt.legend(list_params, fontsize=12, loc=1)

   
    plt.yscale('linear')
    plt.ylabel('$\|  grad \|$')
    plt.xlabel('iteration', fontsize=12)
    if not x_limits == None:
        plt.xlim(x_limits)
    plt.title(str(dataset_name) + ' (n=' + str(n) + ', d=' + str(d) + ')', fontsize=13)  # + ', lambda=1e-3)')



    plt.show


def two_d_plot_epochs(list_loss, list_samples, list_params,list_grads,list_EVs, dataset_name, n, d, log_scale, x_limits=None):
    colors = ['#1B2631', '#C0392B', '#9B59B6', '#2980B9', '#1E8449', '#27AE60', '#E67E22', '#95A5A6', '#FF97F2',
              '#34495E']
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']

    for i in range(len(list_loss)):
        fig = plt.figure(i)
        plt.subplot(1, 3, 1)


        list_x = [[j / n for j in i] for i in list_samples]
        plt.plot(list_x[i], list_loss[i], linestyles[i % 10], color=colors[i % 10], linewidth=3.0)

        plt.legend(list_params[i], fontsize=12, loc=1)
        if not x_limits == None:
            plt.xlim(x_limits)

        if log_scale == True:
            plt.yscale('log')
            plt.ylabel('$\log(f-f^*)$', fontsize=12)
        else:
            plt.yscale('linear')
            plt.ylabel('$f-f^*$', fontsize=12)

        plt.xlabel('epochs', fontsize=12)
        plt.title(str(dataset_name) + ' (n=' + str(n) + ', d=' + str(d) + ')', fontsize=13)  # + ', lambda=1e-3)')

        plt.subplot(1, 3, 2)


        plt.plot(list_x[i], list_grads[i], linestyles[i % 10], color=colors[i % 10], linewidth=3.0)
        #plt.legend(list_params[i], fontsize=12, loc=1)

       
        plt.yscale('linear')
        plt.ylabel('$\|  grad \|$')
        plt.xlabel('epochs', fontsize=12)
        if not x_limits == None:
            plt.xlim(x_limits)
        plt.title(str(dataset_name) + ' (n=' + str(n) + ', d=' + str(d) + ')', fontsize=13)  # + ', lambda=1e-3)')

        plt.subplot(1, 3, 3)

        #show EVs
        plt.plot(list_x[i], list_EVs[i][:,0], linestyles[i % 10], color=colors[i % 10], linewidth=3.0)
        plt.plot(list_x[i], list_EVs[i][:,1], ':' , color=colors[i % 10], linewidth=3.0)

        #plt.legend(list_params[i], fontsize=12, loc=1)

       
        plt.yscale('linear')
        plt.ylabel('$\| \lambda_{\max} \|$ and $\| \lambda_{\min} \|$')
        plt.xlabel('epochs', fontsize=12)
        if not x_limits == None:
            plt.xlim(x_limits)
        plt.title(str(dataset_name) + ' (n=' + str(n) + ', d=' + str(d) + ')', fontsize=13)  # + ', lambda=1e-3)')


        plt.subplots_adjust(left=4, right=6)
        plt.show

def two_d_plot_epochs_all_in_one(list_loss, list_samples, list_params,list_grads,list_EVs, dataset_name, n, d, log_scale, x_limits=None):
    colors = ['#42A5F5','#8BC34A', '#37474F', '#9B59B6', '#2980B9', '#1E8449', '#27AE60', '#E67E22', '#95A5A6','#1B2631' ]
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']
    import numpy as np

    fig = plt.figure(1)

    list_x = [[j / n for j in i] for i in list_samples]
    for i in range(len(list_loss)):
        print(len(list_loss[i]))
        plt.plot(np.arange(len(list_loss[i])), list_loss[i], linestyles[i % 10], color=colors[i % 10], linewidth=3.0)

    plt.legend(list_params, fontsize=12, loc=1)
    if not x_limits == None:
        plt.xlim(x_limits)

    if log_scale == True:
        plt.yscale('log')
        plt.ylabel('$\log(f-f^*)$', fontsize=12)
    else:
        plt.yscale('linear')
        plt.ylabel('$f-f^*$', fontsize=12)

    plt.xlabel('epochs', fontsize=12)
    plt.title(str(dataset_name) + ' (n=' + str(n) + ', d=' + str(d) + ')', fontsize=13)  # + ', lambda=1e-3)')

    fig.savefig('loss.pdf')

    fig=plt.figure(2)


    for i in range(len(list_grads)):
    	plt.plot(np.arange(len(list_grads[i])), list_grads[i], linestyles[i % 10], color=colors[i % 10], linewidth=3.0)
    #plt.legend(list_params[i], fontsize=12, loc=1)

   
    plt.yscale('linear')
    plt.ylabel('$\|  grad \|$')
    plt.xlabel('epochs', fontsize=12)
    if not x_limits == None:
        plt.xlim(x_limits)
    plt.title(str(dataset_name) + ' (n=' + str(n) + ', d=' + str(d) + ')', fontsize=13)  # + ', lambda=1e-3)')
    fig.savefig('grads.pdf')

    fig=figure(3)
    #show EVs
    for i in range(len(list_loss)):
	    plt.plot(np.arange(len(list_EVs[i][:,0])), list_EVs[i][:,0], linestyles[i % 10], color=colors[i % 10], linewidth=3.0)
	    plt.plot(np.arange(len(list_EVs[i][:,1])), list_EVs[i][:,1], ':' , color=colors[i % 10], linewidth=3.0)

    #plt.legend(list_params[i], fontsize=12, loc=1)

   
    plt.yscale('linear')
    plt.ylabel('$\| \lambda_{\max} \|$ and $\| \lambda_{\min} \|$')
    plt.xlabel('epochs', fontsize=12)
    if not x_limits == None:
        plt.xlim(x_limits)
    plt.title(str(dataset_name) + ' (n=' + str(n) + ', d=' + str(d) + ')', fontsize=13)  # + ', lambda=1e-3)')


    plt.subplots_adjust(left=4, right=6)
    fig.savefig('EVs.pdf')

    plt.show
