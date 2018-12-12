#!/bin/bash

sh ./shell/adaptor.sh -s cmd -le .pts -f face -e1 100 -bs1 64 -lr1 1e-4 -mt1 vgg16_rgr -ln1 landmark_loss -c1 vgg16_rgr\
 -e2 100 -bs2 64 -lr2 1e-4 -mt2 vgg16_clf -ln2 no -c2 vgg16_clf