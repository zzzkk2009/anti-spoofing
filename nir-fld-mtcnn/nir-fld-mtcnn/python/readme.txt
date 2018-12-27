1、采集完光流数据集后，执行
    python data_preprocess.py 将ori_dir目录中的数据集分为训练集和测试集，并存放到new_dir新文件夹中；
2、将tools文件夹中的im2rec.py文件拷贝到new_dir文件夹中，然后依次执行：
	python im2rec.py --recursive --exts=.png --list train train
	python im2rec.py --recursive --exts=.png --list test test
	python im2rec.py ./test.lst test
	python im2rec.py ./train.lst train
    注：由于windows版本的mxnet问题，执行程序后，进程并不会退出，需要手动结束python.exe进程；
        最终生成的train.rec和test.rec就是处理完后可以直接用于训练的数据集。