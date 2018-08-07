import os
import argparse
import chainer.links as L
from chainer import cuda,serializers,training,optimizers
import chainer
from chainer.training import extensions

from pycocotools.coco import COCO
from coco_data_loader import CocoDataLoader
import Network2
from Constants import constants
from network_updator import Net_Updater

class GradientScaling(object):

    name = 'GradientScaling'

    def __init__(self, layer_names, scale):
        self.layer_names = layer_names
        self.scale = scale

    def __call__(self, opt):
        for layer_name in self.layer_names:
            for param in opt.target[layer_name].params(False):
                grad = param.grad
                with cuda.get_device_from_array(grad):
                    grad *= self.scale

def set_args():
    parser = argparse.ArgumentParser(description='Pose Estimator training')
    parser.add_argument('--batchsize', type=int, default=16, help='Training minibatch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--epoch', '-e', type=int, default=300000,
                       help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                 help='Learning rate for SGD')

    args = parser.parse_args()

    return args

def main():
    #引数の格納
    args = set_args()

    #modelの設定
    model = Network2.MyNet()

    #GPUの設定
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # optimizerのセットアップ
    #optimizer = chainer.optimizers.MomentumSGD(lr=args.learnrate).setup(model)
    optimizer = optimizers.Adam(alpha=1e-4, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    #optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))
    layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                       'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2',
                       'conv4_3_CPM', 'conv4_4_CPM']
    optimizer.add_hook(GradientScaling(layer_names, 1/4))

    layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                               'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2']
    for layer_name in layer_names:
        optimizer.target[layer_name].enable_update()

    #datasetの読み込み
    #CocoDataLoaderは既存ファイルの使用
    coco_train = COCO(os.path.join(constants['data_add'], 'annotations/person_keypoints_train2017.json'))
    coco_val = COCO(os.path.join(constants['data_add'], 'annotations/person_keypoints_val2017.json'))
    train = CocoDataLoader(coco_train, model.insize, mode='train')
    val = CocoDataLoader(coco_val, model.insize, mode='val', n_samples=100)

    #iteratorのセットアップ
    train_iter = chainer.iterators.SerialIterator(train,args.batchsize,shuffle=True)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize, repeat=False, shuffle=False)

    # UpdatorとTrainerのセットアップ
    updater = Net_Updater(train_iter,model,optimizer,device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    #trainerのExtend
    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

    #学習開始
    trainer.run()

    #学習結果の保存
    model.to_cpu()
    serializers.save_npz('trained_model',model)


if __name__ == '__main__':
    main()
