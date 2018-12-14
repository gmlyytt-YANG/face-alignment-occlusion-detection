#!/bin/bash

sh ./shell/train.sh -s cmd -c $1 -e 100 -bs 64 -lr 1e-4 -p rough -f face -le .pts -mt resnet_rgr -ln landmark_loss
sh ./shell/train.sh -s cmd -c $1 -e 100 -bs 64 -lr 3e-4 -p rough -f face -le .pts -mt resnet_rgr -ln landmark_loss
sh ./shell/train.sh -s cmd -c $1 -e 100 -bs 64 -lr 5e-4 -p rough -f face -le .pts -mt resnet_rgr -ln landmark_loss
sh ./shell/train.sh -s cmd -c $1 -e 100 -bs 64 -lr 1e-5 -p rough -f face -le .pts -mt resnet_rgr -ln landmark_loss

sh ./shell/train.sh -s cmd -c $1 -e 100 -bs 128 -lr 1e-4 -p rough -f face -le .pts -mt resnet_rgr -ln landmark_loss
sh ./shell/train.sh -s cmd -c $1 -e 100 -bs 128 -lr 3e-4 -p rough -f face -le .pts -mt resnet_rgr -ln landmark_loss
sh ./shell/train.sh -s cmd -c $1 -e 100 -bs 128 -lr 5e-4 -p rough -f face -le .pts -mt resnet_rgr -ln landmark_loss
sh ./shell/train.sh -s cmd -c $1 -e 100 -bs 128 -lr 1e-5 -p rough -f face -le .pts -mt resnet_rgr -ln landmark_loss

