from os import path

from PIL import Image
import numpy as np

from numpy import count_nonzero as nnz


pred_folder = 'predictions'

ref_folder = 'benchmark/test/gt'
test_set = True;

#or...
#ref_folder = 'benchmark_val/val/gt'
#test_set = False;


if (test_set):
	prefixes = ['bellingham','bloomington','innsbruck','sfo','tyrol-e']
	indices = range(1,37)
else:
	prefixes = ['austin','chicago','kitsap','tyrol-w','vienna']
	indices = range(1,6)



inters_acum = 0
union_acum = 0
correct_acum = 0;
total_acum = 0;

for prefix in prefixes:

	inters_count = 0
	union_count = 0
	correct_count = 0;
	total_count = 0;
	
	for index in indices:
		ref_path = path.join(ref_folder,prefix+str(index)+'.tif')
		pred_path = path.join(pred_folder,prefix+str(index)+'.tif')

		ref = np.array(Image.open(ref_path))/255
		pred = np.array(Image.open(pred_path))/255

		inters = ref & pred
		union = ref | pred
		correct = ref == pred

		inters_count += nnz(inters)
		union_count += nnz(union)
		correct_count += nnz(correct)
		total_count += ref.size

	inters_acum+=inters_count
	union_acum+=union_count
	correct_acum+=correct_count
	total_acum+=total_count

	iou = inters_count/float(union_count)
	acc = correct_count/float(total_count)

	print prefix, iou, acc

overall_iou = inters_acum/float(union_acum)
overall_acc = correct_acum/float(total_acum)

print 'Overall', overall_iou, overall_acc

