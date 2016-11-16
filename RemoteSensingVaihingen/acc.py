

from PIL import Image
import numpy as np
from numpy import count_nonzero as nnz

from sklearn import metrics
from sklearn.metrics import confusion_matrix


def acc(prediction,ref):

	#compute overall accuracy
	valid = (ref!=6)
	correct = valid & (prediction==ref)
	acc = nnz(correct) / float(nnz(valid))

	#compute f-1 per class
	ref_flattened = ref.flatten()
	pred_flattened = prediction.flatten()
	mask = ~np.equal(ref_flattened,6)
#	mask = ~np.equal(ref_flattened,6) & ~np.equal(ref_flattened,5)
	ref_flattened = ref_flattened[mask]
	pred_flattened = pred_flattened[mask]
	f1 = metrics.f1_score(ref_flattened, pred_flattened, labels=[0,1,2,3,4,5], average=None)
#	f1 = metrics.f1_score(ref_flattened, pred_flattened, labels=[0,1,2,3,4], average=None)



	#compute mean f-1
	mean_f1 = sum(f1[0:5])/float(len(f1)-1)
#	mean_f1 = sum(f1)/float(len(f1))

	#separate the reference in a vector of 
	#logical arrays for every class
	ref_classes = []
	pred_classes = []
	for i in range(0,5):
		ref_classes.append(np.equal(ref,i))
		pred_classes.append(np.equal(prediction,i))

	#compute IoU per class
	ious = []	
	for i in range(0,5):
		inters=ref_classes[i] & pred_classes[i]
		union=ref_classes[i] | pred_classes[i]
		iou = nnz(inters) / float(nnz(union))
		ious.append(iou)

	#compute mean IoU
	meanIou = sum(ious)/float(len(ious))

	#output

	out = {}
	out['acc']=acc
	out['meanIou']=meanIou
	out['ious']=ious
	out['f1']=f1
	out['mean_f1']=mean_f1



	return out
