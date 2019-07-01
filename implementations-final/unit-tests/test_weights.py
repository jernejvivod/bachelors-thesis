import scipy.io as sio
import numpy as np

data1 = sio.loadmat('data.mat')['data']
target1 = np.ravel(sio.loadmat('target.mat')['target'])

from algorithms.irelief import IRelief
from algorithms.reliefseq import ReliefSeq
from algorithms.SURFStar import SURFStar


## IRELIEF ###

print("IRelief")

irelief = IRelief(n_features_to_select=2)
irelief.fit(data1, target1)
print("weights: {0}".format(irelief.weights))

####################




## MULTISURFSTAR ###

print("MultiSURFStar")


####################




## RELIEFSEQ ###

print("ReliefSeq")

reliefseq = ReliefSeq(n_features_to_select=2)
reliefseq.fit(data1, target1)
print("weights: {0}".format(reliefseq.weights))

################



# TODO!!! ASK MENTOR
## SURFSTAR ####

print("SURFStar")
surfstar = SURFStar(n_features_to_select=2)
surfstar.fit(data1, target1)
print("weights: {0}".format(surfstar.weights))

################






## TURF ####
print("TuRf")


############




## BOOSTEDSURF ###
print("BoostedSURF")
##################





## ITERATIVE-RELIEF ####
print("Iterative-Relief")
########################





## MULTISURF ####
print("MultiSURF")
#################






## RELIEFF #####
print("Relieff")
################





## RELIEF ######
print("Relief")
################




## SURF ########
print("SURF")
################



## SURFSTAR ####
print("SURFStar")
################



# VLSRELIEF ####
print("VlsRelief")
################
