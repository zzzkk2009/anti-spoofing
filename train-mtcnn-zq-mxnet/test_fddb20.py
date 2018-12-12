import numpy as np
import mxnet as mx
import argparse
from core.symbol import P_Net20, R_Net, O_Net
from core.imdb import IMDB
from config import config
from core.loader import TestLoader
from core.detector import Detector
from core.fcn_detector import FcnDetector
from tools.load_model import load_param
from core.MtcnnDetector20 import MtcnnDetector


def test_net(root_path, dataset_path, prefix, epoch,
             batch_size, ctx, test_mode="onet",  thresh=[0.6, 0.6, 0.7], min_face_size=24):

    detectors = [None, None, None]

    # load pnet model
    args, auxs = load_param(prefix[0], epoch[0], convert=True, ctx=ctx)
    PNet = FcnDetector(P_Net20("test"), ctx, args, auxs)
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["rnet", "onet"]:
        args, auxs = load_param(prefix[1], epoch[0], convert=True, ctx=ctx)
        RNet = Detector(R_Net("test"), 24, batch_size[1], ctx, args, auxs)
        detectors[1] = RNet

    # load onet model
    if test_mode == "onet":
        args, auxs = load_param(prefix[2], epoch[2], convert=True, ctx=ctx)
        ONet = Detector(O_Net("test",False), 48, batch_size[2], ctx, args, auxs)
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, ctx=ctx, min_face_size=min_face_size,
                                   stride=4, threshold=thresh, slide_window=False)

    for i in range(1,11):
        image_set = "fold-" + str(i).zfill(2)
        imdb = IMDB("fddb", image_set, root_path, dataset_path, 'test')
        gt_imdb = imdb.get_annotations()

        test_data = TestLoader(gt_imdb)
        all_boxes = mtcnn_detector.detect_face(imdb, test_data, vis=False)
        imdb.write_results(all_boxes)



def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', dest='root_path', help='output data folder',
                        default='data', type=str)
    parser.add_argument('--dataset_path', dest='dataset_path', help='dataset folder',
                        default='data/fddb', type=str)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='onet', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name',
                        default='%s/model/pnet20'%config.root+',%s/model/rnet'%config.root+',%s/model/onet'%config.root, type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', 
                        default='13,20,16', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', 
                        default='2048,256,16', type=str)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet',
                        default='0.3,0.3,0.3', type=str)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=24, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window', help='use sliding window in pnet', action='store_true')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',
                        default=0, type=int)
    parser.add_argument('--shuffle', dest='shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--vis', dest='vis', help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print 'Called with argument:'
    print args
    ctx = mx.gpu(args.gpu_id)
    if args.gpu_id == -1:
        ctx = mx.cpu(0)
    prefix = args.prefix.split(',')
    epoch = [int(i) for i in args.epoch.split(',')]
    batch_size = [int(i) for i in args.batch_size.split(',')]
    thresh = [float(i) for i in args.thresh.split(',')]
    test_net(args.root_path, args.dataset_path, prefix,
             epoch, batch_size, ctx, args.test_mode,
             thresh, args.min_face)
