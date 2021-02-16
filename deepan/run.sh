#!/bin/bash
for linear in 'True'
do
  for variational in 'True'
  do
      for epochs in 200
      do
          for lr in 0.01 0.008
          do
              for cutoff in 0.33 0.37 0.4 0.43
              do
                  for outputchannels in 16
                  do
                              python pytorch_linearVAE.py --linear=${linear} --variational=${variational} --epochs=${epochs} --lr=${lr} --outputchannels=${outputchannels} --cutoff=${cutoff}
                      done
                  done
              done
          done
      done
  done