import scipy.io as sio
import numpy as np

data1 = sio.loadmat('data.mat')['data']
target1 = np.ravel(sio.loadmat('target.mat')['target'])

data2 = sio.loadmat('data2.mat')['data']
target2 = np.ravel(sio.loadmat('target2.mat')['target'])

data3 = sio.loadmat('data3.mat')['data']
target3 = np.ravel(sio.loadmat('target3.mat')['target'])

from algorithms.irelief import IRelief
from algorithms.reliefseq import ReliefSeq
from algorithms.SURFStar import SURFStar
from algorithms.turf import TURF
from algorithms.relieff import Relieff
from algorithms.relief import Relief
from algorithms.iterative_relief import IterativeRelief
from algorithms.multiSURF import MultiSURF
from algorithms.multiSURFStar import MultiSURFStar
from algorithms.SURF import SURF
from algorithms.vlsrelief import VLSRelief
from algorithms.boostedSURF import BoostedSURF
from algorithms.swrfStar import SWRFStar
from algorithms.reliefmms import ReliefMMS
from algorithms.ecrelieff import ECRelieff


## IRELIEF ###

print("IRelief")

irelief = IRelief(n_features_to_select=2)
irelief = irelief.fit(data1, target1)
print("weights: {0}".format(irelief.weights))

####################




## MULTISURFSTAR ###

print("MultiSURFStar")
multisurfstar = MultiSURFStar(n_features_to_select=2)
multisurfstar = multisurfstar.fit(data2, target2)
print("weights: {0}".format(multisurfstar.weights))

####################




## RELIEFSEQ ###

print("ReliefSeq")

reliefseq = ReliefSeq(n_features_to_select=2)
reliefseq = reliefseq.fit(data1, target1)
print("weights: {0}".format(reliefseq.weights))

################



# TODO!!! ASK MENTOR
## SURFSTAR ####

print("SURFStar")
surfstar = SURFStar(n_features_to_select=2)
surfstar = surfstar.fit(data3, target3)
print("weights: {0}".format(surfstar.weights))

################




## TURF ####

print("TuRf")
relieff = Relieff(n_features_to_select=2)
turf = TURF(n_features_to_select=2, num_it=1, rba=relieff)
turf = turf.fit(data1, target1)
print("weights: {0}".format(turf.weights))

############




## BOOSTEDSURF ###

print("BoostedSURF")
boostedsurf = BoostedSURF(n_features_to_select=2)
boostedsurf = boostedsurf.fit(data2, target2)
print("weights: {0}".format(boostedsurf.weights))

##################



## ITERATIVE-RELIEF ####

print("Iterative-Relief")

iterative_relief = IterativeRelief(n_features_to_select=2)
iterative_relief = iterative_relief.fit(data1, target1)
print("weights: {0}".format(iterative_relief.weights))

########################



## MULTISURF ####
# TODO: check

print("MultiSURF")
multisurf = MultiSURF(n_features_to_select=2)
multisurf = multisurf.fit(data3, target3)
print("weights: {0}".format(multisurf.weights))

#################



## RELIEFF #####

print("Relieff")
relieff = Relieff(n_features_to_select=2)
relieff = relieff.fit(data1, target1)
print("weights: {0}".format(relieff.weights))

################




## RELIEF ######

print("Relief")
relief = Relief(n_features_to_select=2)
relief = relief.fit(data1, target1)
print("weights: {0}".format(relief.weights))

################




## SURF ########

print("SURF")
surf = SURF(n_features_to_select=2)
surf = surf.fit(data1, target1)
print("weights: {0}".format(surf.weights))

################




# VLSRELIEF ####

print("VlsRelief")
vlsrelief = VLSRelief(n_features_to_select=2, num_partitions_to_select=2, num_subsets=10, partition_size=1)
vlsrelief = vlsrelief.fit(data1, target1)
print("weights: {0}".format(vlsrelief.weights))

################




# SWRFSTAR #####

print("SwrfStar")
swrfstar = SWRFStar(n_features_to_select=2)
swrfstar = swrfstar.fit(data3, target3)
print("weights: {0}".format(swrfstar.weights))

################




# RELIEFMMS ####

print("ReliefMMS")
reliefmms = ReliefMMS(n_features_to_select=2)
reliefmms = reliefmms.fit(data3, target3)
print("weights: {0}".format(reliefmms.weights))

################


# EVAPORATIVE COOLING RELIEFF ####

print("Evaporative Cooling ReliefF")
ecrelieff = ECRelieff(n_features_to_select=2)
ecrelieff = ecrelieff.fit(data3, target3)
print("rank: {0}".format(ECRelieff.rank))

##################################
