#!/bin/bash
for linear in 'True' 'False'
do
  for variational in 'False' 'True'
  do
      for epochs in 300
      do
          for lr in 0.005
          do
              for cutoff in 0.5
              do
                  for outputchannels in 208
                  do
                      for projection in 'TSNE' 'UMAP'
                      do
                              python pytorch_linearVAE.py --linear=${linear} --variational=${variational} --epochs=${epochs} --lr=${lr} --outputchannels=${outputchannels} --cutoff=${cutoff} --projection=${projection}
                      done
                  done
              done
          done
      done
  done
done