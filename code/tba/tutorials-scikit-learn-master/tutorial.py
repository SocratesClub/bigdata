from matplotlib import pyplot as plt
import numpy as np

def plot_surface(clf, X, y, 
                 xlim=(-10, 10), ylim=(-10, 10), n_steps=250, 
                 subplot=None, show=True):
    if subplot is None:
        fig = plt.figure()
    else:
        plt.subplot(*subplot)
        
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], n_steps), 
                         np.linspace(ylim[0], ylim[1], n_steps))
    
    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, alpha=0.8, cmap=plt.cm.RdBu_r)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    
    if show:
        plt.show()

def plot_histogram(clf, X, y, subplot=None, show=True):
    if subplot is None:
        fig = plt.figure()
    else:
        plt.subplot(*subplot)
    
    if hasattr(clf, "decision_function"):
        d = clf.decision_function(X)
    else:
        d = clf.predict_proba(X)[:, 1]
    
    plt.hist(d[y == "b"], bins=50, normed=True, color="b", alpha=0.5)
    plt.hist(d[y == "r"], bins=50, normed=True, color="r", alpha=0.5)
    
    if show:
        plt.show()

def plot_clf(clf, X, y):
    plt.figure(figsize=(16, 8))
    plot_surface(clf, X, y, subplot=(1, 2, 1), show=False)
    plot_histogram(clf, X, y, subplot=(1, 2, 2), show=True)
