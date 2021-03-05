#!/bin/bash
for linear in 'True'
do
  for variational in 'False'
  do
      for epochs in 300
      do
          for lr in 0.005
          do
              for cutoff in 0.5
              do
                  for outputchannels in 208 100 16 6
                  do
                      for projection in 'UMAP'
                      do
                              python pytorch_linearVAE.py --linear=${linear} --variational=${variational} --epochs=${epochs} --lr=${lr} --outputchannels=${outputchannels} --cutoff=${cutoff} --projection=${projection}
                      done
                  done
              done
          done
      done
  done
done