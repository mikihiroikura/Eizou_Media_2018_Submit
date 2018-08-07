import chainer
from chainer import training, cuda,reporter
from chainer import functions as F


def image_reshape(imgs):
    x_data = imgs.astype('f')
    x_data /= 255
    x_data -= 0.5
    x_data = x_data.transpose(0, 3, 1, 2)
    return x_data


def loss_func(heatmap_true,paf_true,heatmap_es,paf_es,ignore_mask):
    heatmap_loss_each = []
    paf_loss_each = []
    total_loss =0

    paf_masks = ignore_mask[:, None].repeat(paf_true.shape[1], axis=1)
    heatmap_masks = ignore_mask[:, None].repeat(heatmap_true.shape[1], axis=1)

    for i in range(6):
        stage_pafs_t = paf_true.copy()
        stage_heatmaps_t = heatmap_true.copy()
        stage_paf_masks = paf_masks.copy()
        stage_heatmap_masks = heatmap_masks.copy()
        if (paf_true.shape!=paf_es[i].shape):
            stage_pafs_t = F.resize_images(stage_pafs_t, paf_es[i].shape[2:]).data
            stage_heatmaps_t = F.resize_images(stage_heatmaps_t, paf_es[i].shape[2:]).data
            stage_paf_masks = F.resize_images(stage_paf_masks.astype('f'), paf_es[i].shape[2:]).data > 0
            stage_heatmap_masks = F.resize_images(stage_heatmap_masks.astype('f'), paf_es[i].shape[2:]).data > 0

        stage_pafs_t[stage_paf_masks == True] = paf_es[i].data[stage_paf_masks == True]
        stage_heatmaps_t[stage_heatmap_masks == True] = heatmap_es[i].data[stage_heatmap_masks == True]

        pafs_loss = F.mean_squared_error(paf_es[i], stage_pafs_t)
        heatmaps_loss = F.mean_squared_error(heatmap_es[i], stage_heatmaps_t)
        total_loss += pafs_loss+heatmaps_loss

        heatmap_loss_each.append(float(cuda.to_cpu(heatmaps_loss.data)))
        paf_loss_each.append(float(cuda.to_cpu(pafs_loss.data)))

    return total_loss,heatmap_loss_each,paf_loss_each

class Net_Updater(training.StandardUpdater):

    def __init__(self, iterator, model, optimizer, device=None):
        super(Net_Updater, self).__init__(iterator, optimizer, device=device)

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        
        batch = train_iter.next()
        #batchの中のChainer datasetの型をArray of tuples⇒tuples of arraysに変換する
        imgs_true, pafs_true, heatmaps_true, ignore_mask = self.converter(batch, self.device)

        img_res = image_reshape(imgs_true)
        heatmap_es, paf_es = optimizer.target(img_res)#Optimizerの中のNetworkのCallが呼ばれ，出力が計算される

        total_loss,heatmap_loss_each,paf_loss_each = loss_func(
                    heatmaps_true,pafs_true,heatmap_es,paf_es,ignore_mask)

        reporter.report({
            'main/loss':total_loss,
            'main/heat_loss':sum(heatmap_loss_each),
            'main/paf_loss':sum(paf_loss_each)})

        optimizer.target.cleargrads()
        total_loss.backward()
        optimizer.update()
