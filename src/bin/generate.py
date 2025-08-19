#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import argparse
import time
from datetime import timedelta

from scipy.stats import ortho_group

import os

def main(args):

  if args.actf == "tanh":
    actf = lambda x: np.tanh(x)
    actdf = lambda x: 1 - np.tanh(x)**2
  elif args.actf == "relu":
    actf = lambda x: x * (x > 0)
    actdf = lambda x: 1. * (x > 0)
  elif args.actf == "identity":
    actf = lambda x: x
    actdf = lambda x: np.ones_like(x)
  else:
    raise ValueError("Wrong actf type")
  print("actf :", args.actf)

  # Input layer index: i
  # First layer index: l
  # Second layer index: k
  # Output layer index: o
  def MSE(x, y, M, W, Z, actf):
    hi = actf(np.einsum('li,ni->nl', M, x))
    h = actf(np.einsum('kl,nl->nk', W, hi))
    yp = np.einsum('ok,nk->no', Z, h)
    return 0.5*np.mean(np.sum((y - yp)**2, axis=1))

  def dL_dW(x, y, M, W, Z, actf, actdf):
    hi = actf(np.einsum('li,ni->nl', M, x))
    h = actf(np.einsum('kl,nl->nk', W, hi))
    dh = actdf(np.einsum('kl,nl->nk', W, hi))
    yp = np.einsum('ok,nk->no', Z, h)
    Zty = np.einsum('ko,no->nk', Z.T, y)
    ZtZh = np.einsum('ko,no->nk', Z.T, yp)
    return - np.mean(np.einsum('nk,nl->nkl', (Zty - ZtZh) * dh, hi), axis=0)

  def forward(x, M, W, Z, actf):
    hi = actf(np.einsum('li,ni->nl', M, x))
    h = actf(np.einsum('kl,nl->nk', W, hi))
    yp = np.einsum('ok,nk->no', Z, h)
    return yp

  # Network structure
  # np.random.seed(42)

  r = args.r # ratio r = K/L
  K = args.K
  L = K * r
  dim_i = 3
  dim_o = 1
  
  # Set teacher matrix M
  M_mu = args.M_mu
  M_sig = args.M_sig

  # Set teacher matrix W
  W_mu = args.W_mu
  W_sig = args.W_sig

  # Set shared matrix Z
  Z_mu = args.Z_mu
  Z_sig = args.Z_sig
      
  # Training settings
  bs = args.bs
  lr = args.lr

  print("T: ", lr/bs)

  n_batchs = 1000

  n_seed = args.n_seed
  print("n_seed: ", n_seed)

  # model
  eps = args.Wp_sig

  epochs = args.epochs
  save_points = min(20, epochs)
  index_to_save = np.linspace(0, epochs, save_points+1)
  index_interval = epochs // save_points

  history = {'W': np.zeros(shape=(n_seed, save_points+1, K, L)), 
             'dW': np.zeros(shape=(n_seed, save_points+1, K, L)),
             'loss': np.ones(shape=(n_seed, save_points+1))*100.,
             'Wt': np.empty(shape=(n_seed, K, L)),
             'Z': np.empty(shape=(n_seed, dim_o, K)),
             'M': np.empty(shape=(n_seed, L, dim_i))}

  project = args.project # 2 non degenerate, 2 degenerate pairs
  project_name = "r%dK%d_"%(r,K) + args.actf + "_lr%.5g_bs%.5g_eps%.5g"%(lr, bs, eps)
  print("Project: ", project_name)
  model_dir = args.model_dir+"/"+project+"/r"+str(r)+"K"+str(K)+"/"+project_name+"/"

  os.system("mkdir -p "+model_dir)
  
  # Train
  start_time = time.time()
  for s in range(n_seed):

    M = np.random.normal(M_mu, M_sig/np.sqrt(dim_i), size=(L, dim_i))
    W = np.random.normal(W_mu, W_sig/np.sqrt(L), size=(K, L))
    Z = np.random.normal(Z_mu, Z_sig/np.sqrt(K), size=(dim_o, K)) 

    history['M'][s] = M.copy()
    v, m, u = np.linalg.svd(M)
    print("Shared matrix M: ", M)
    print("Shared singular values: ", m)

    history['Wt'][s] = W.copy()
    v, w, u = np.linalg.svd(W)
    
    history['Z'][s] = Z.copy()
    v, z, u = np.linalg.svd(Z)
    print("Shared matrix Z: ", Z)
    print("Shared singular values: ", z)
    
    # Initial data
    x_data = np.random.normal(0., 1., size=(bs * n_batchs, dim_i))
    y_data = forward(x_data, M, W, Z, actf)

    print('===== seed %d ====='%(s))

    Wp = np.random.normal(loc=0, scale=eps/np.sqrt(L), size=(K, L))
    
    loss = MSE(x_data, y_data, M, Wp, Z, actf)
    grad = dL_dW(x_data, y_data, M, Wp, Z, actf, actdf)
    
    history['W'][s][0] = Wp.copy()
    history['dW'][s][0] = grad.copy()
    history['loss'][s][0] = loss.copy()
    
    isnan = False
    for e in range(epochs):
      loss = 0.
      grad = np.zeros_like(Wp)
      for b in range(n_batchs):
        x_batch = x_data[b*bs:(b+1)*bs]
        y_batch = forward(x_batch, M, W, Z, actf)

        batch_loss = MSE(x_batch, y_batch, M, Wp, Z, actf)
        batch_grad = dL_dW(x_batch, y_batch, M, Wp, Z, actf, actdf)
        
        if batch_loss > 100.:
          print("loss larger than 100. Training terminated")
          end_W = Wp.copy()
          end_dW = batch_grad
          isnan = True
          break

        Wp -= lr * batch_grad

        loss += batch_loss
        grad += batch_grad

      idx = e // index_interval + 1
      if isnan:
        history['W'][s][idx:] = np.tile(end_W, (save_points - idx + 1, 1, 1))
        history['dW'][s][idx:] = np.tile(end_dW, (save_points - idx + 1, 1, 1))
        break

      if e in index_to_save:

        loss /= n_batchs
        
        history['W'][s][idx] = Wp.copy()
        history['dW'][s][idx] = grad / n_batchs
        history['loss'][s][idx] = loss
        
        print(e + 1, loss)

    # Fail safe:
    try:
      _, wp, _ = np.linalg.svd(Wp)
      print("Final singular values: ", wp)
    except:
      print("Final singular values did not converge")
    print("Target singular values: ",w)
    print("Final W: ", history['W'][s][-1])
    print("Target matrix: ",W)

  runtime = time.time() - start_time
  print("Training time: "+str(timedelta(seconds=runtime)))
  print("avg loss: ", np.mean(history['loss'][:,-1]))
            
  # Save model
  np.savez(model_dir+'data.npz', history)
  print("Saved model ", model_dir+'data.npz', ".")

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="NNRM test")
  parser.add_argument("--actf", type=str, default="identity", help="Activation function")
  parser.add_argument("--r", type=int, default=2, help="Ratio K/L")
  parser.add_argument("--K", type=int, default=4, help="Second layer width")
  parser.add_argument("--M_mu", type=float, default=0., help="Mean of teacher M matrix")
  parser.add_argument("--M_sig", type=float, default=1., help="Width of teacher M matrix")
  parser.add_argument("--W_mu", type=float, default=0., help="Mean of teacher W matrix")
  parser.add_argument("--W_sig", type=float, default=1., help="Width of teacher W matrix")
  parser.add_argument("--Z_mu", type=float, default=0., help="Mean of shared Z matrix")
  parser.add_argument("--Z_sig", type=float, default=1., help="Width of shared Z matrix")
  parser.add_argument("--bs", type=int, default=16, help="Batch size")
  parser.add_argument("--lr", type=float, default=0.1, help="Step size")
  parser.add_argument("--n_seed", type=int, default=200, help="Ensemble size")
  parser.add_argument("--Wp_sig", type=float, default=1., help="Initialization width")
  parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
  parser.add_argument("--project", type=str, default="NN_RM", help="Project name")
  parser.add_argument("--model_dir", type=str, default="./", help="Model directory")
  args = parser.parse_args()

  main(args)
