活体检测数据集采集要求：
	1、人脸需保持在画面指定区域内，只有在区域内检测到人脸才会采集；
	2、采集过程中，需完成以下指定动作：
		1) 左右上下摇头，保持3-5秒；
		2) 眨眼，张嘴，吐舌头等反复3-5次；
		3) 用手遮挡额头、下巴、左右小半边脸，反复3-5次；
		4) 人脸拉远和靠近摄像头，持续3-5秒；
		5) 带了眼镜等配饰的，需要取下后，重复上述过程，
		   同时取下和佩戴过程也需要进行采集；




1、采集完光流数据集后，执行
    python data_preprocess.py 将ori_dir目录中的数据集分为训练集和测试集，并存放到new_dir新文件夹中；
2、将tools文件夹中的im2rec.py文件拷贝到new_dir文件夹中，然后依次执行：
	python im2rec.py --recursive --exts=.png --list train train
	python im2rec.py --recursive --exts=.png --list test test
	python im2rec.py ./test.lst test
	python im2rec.py ./train.lst train
    注：由于windows版本的mxnet问题，执行程序后，进程并不会退出，需要手动结束python.exe进程；
        最终生成的train.rec和test.rec就是处理完后可以直接用于训练的数据集。