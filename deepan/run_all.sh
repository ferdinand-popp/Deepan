#!/bin/bash
for linear in 'True' 'False'
do
  for variational in 'True' 'False'
  do
      for epochs in 200 400 600
      do
          for lr in 0.008 0.01 0.012
          do
              for cutoff in 0.2 0.3 0.35 0.4 0.5
              do
                  for outputchannels in 2 8 16 24 52
                  do
                              python pytorch_linearVAE.py --linear=${linear} --variational=${variational} --epochs=${epochs} --lr=${lr} --outputchannels=${outputchannels} --cutoff=${cutoff}
                      done
                  done
              done
          done
      done
  done