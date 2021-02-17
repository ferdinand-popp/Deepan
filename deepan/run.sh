#!/bin/bash
for linear in 'True'
do
  for variational in 'False'
  do
      for epochs in 300
      do
          for lr in 0.006 0.008 0.01
          do
              for cutoff in 0.33
              do
                  for outputchannels in 208
                  do
                              python pytorch_linearVAE.py --linear=${linear} --variational=${variational} --epochs=${epochs} --lr=${lr} --outputchannels=${outputchannels} --cutoff=${cutoff}
                      done
                  done
              done
          done
      done
  done