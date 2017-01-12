import os
import shutil
import sys

# DATA_PATH="/Users/bhavishyamittal/Documents/Reseach/validation/"
# RESULT_IMG_PATH="/Users/bhavishyamittal/Documents/Reseach/validation Res/human_bb/bb_img/"
# RESULT_REC_PATH="/Users/bhavishyamittal/Documents/Reseach/validation Res/human_bb/bb_rec/"
DATA_PATH="/Users/bhavishyamittal/Documents/Research/Dance Project/validation/"
RESULT_IMG_PATH="/Users/bhavishyamittal/Documents/Research/Dance Project/Darknet/result/bb_img/"
RESULT_REC_PATH="/Users/bhavishyamittal/Documents/Research/Dance Project/Darknet/result/bb_rec/"
thresh = 0.3

if os.path.isdir(RESULT_IMG_PATH):
	confirm = raw_input('Delete human_bb/bb_{img/rec} [Y/n]: ')
	if confirm.lower()=='y' or confirm == '':
		shutil.rmtree(RESULT_IMG_PATH, ignore_errors=True)
		shutil.rmtree(RESULT_REC_PATH, ignore_errors=True)
		os.mkdir(RESULT_IMG_PATH);
		os.mkdir(RESULT_REC_PATH);
	else:
		sys.exit(1)
else:
	print "Creating directory - ", RESULT_IMG_PATH
	os.mkdir(RESULT_IMG_PATH);
	os.mkdir(RESULT_REC_PATH);

print os.getcwd()
os.chdir("darknet")
for dance in os.listdir(DATA_PATH):
	print dance
	if os.path.isdir(os.path.join(DATA_PATH, dance)):
		dancePath = DATA_PATH+dance+"/"
		bbImgPath = RESULT_IMG_PATH + dance + "/"
		bbRecPath = RESULT_REC_PATH + dance + "/"
		os.mkdir(bbImgPath);
		os.mkdir(bbRecPath);
		# ./darknet yolo test_custom cfg/yolo.cfg yolo.weights ~/Documents/Reseach/dance1 ~/Documents/Reseach/dance1_bb_csv -thresh 0.3
		runcmd = './darknet yolo test_custom cfg/yolo.cfg yolo.weights "'\
			+dancePath+'" "'+bbRecPath+'" "'+bbImgPath+'" -thresh '+str(thresh)
		print runcmd
		os.system(runcmd)

		
