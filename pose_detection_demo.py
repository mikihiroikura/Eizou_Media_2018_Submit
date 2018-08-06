import cv2
import argparse
from chainer import serializers
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

import Network2
from Constants import constants,JointType
import time

#pose_detect用クラス
class pose_detect(object):
    def __init__(self,model,device=-1):
        self.model = model
        self.device = device
        #GPUの読み出し
        if device >= 0:
            cuda.get_device_from_id(device).use()
            self.model.to_gpu()

            #GPUモデル用ガウシアンフィルターの作成
            self.gaussian_kernel = self.create_gaussian_kernel(constants['gaussian_sigma'], constants['ksize'])[None, None]
            self.gaussian_kernel = cuda.to_gpu(self.gaussian_kernel)

    #ガウシアンカーネルの作成関数
    def create_gaussian_kernel(self, sigma=1, ksize=5):
        center = int(ksize / 2)
        grid_x = np.tile(np.arange(ksize), (ksize, 1))
        grid_y = grid_x.transpose().copy()
        grid_d2 = (grid_x - center) ** 2 + (grid_y - center) ** 2
        kernel = 1/(sigma**2 * 2 * np.pi) * np.exp(-0.5 * grid_d2 / sigma**2)
        return kernel.astype('f')

    #学習器へデータを入れる前の下準備関数
    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)[None]
        return x_data

    #画像の幅と高さがstrideの倍数になるように調節する関数
    def compute_optimal_size(self, orig_img, img_size, stride=8):
        orig_img_h, orig_img_w, _ = orig_img.shape
        aspect = orig_img_h / orig_img_w
        if orig_img_h < orig_img_w:
            img_h = img_size
            img_w = np.round(img_size / aspect).astype(int)
            surplus = img_w % stride
            if surplus != 0:
                img_w += stride - surplus
        else:
            img_w = img_size
            img_h = np.round(img_size * aspect).astype(int)
            surplus = img_h % stride
            if surplus != 0:
                img_h += stride - surplus
        return (img_w, img_h)

    #heatmapからピークを出力する関数，流用
    #https://github.com/DeNA/Chainer_Realtime_Multi-Person_Pose_Estimation/blob/master/pose_detector.py
    def compute_peaks_from_heatmaps(self, heatmaps):
        heatmaps = heatmaps[:-1]

        #cypyかnumpyと返す
        xp = cuda.get_array_module(heatmaps)
        #numpyならば
        if xp == np:
            all_peaks = []
            peak_counter = 0
            for i , heatmap in enumerate(heatmaps):
                #ガウシアンフィルターで滑らかにする
                heatmap = ndimage.filters.gaussian_filter(heatmap, sigma=constants['gaussian_sigma'])
                #上下左右に1ピクセルずらし，残りに０を挿入する
                map_left = xp.zeros(heatmap.shape)
                map_right = xp.zeros(heatmap.shape)
                map_top = xp.zeros(heatmap.shape)
                map_bottom = xp.zeros(heatmap.shape)
                map_left[1:, :] = heatmap[:-1, :]
                map_right[:-1, :] = heatmap[1:, :]
                map_top[:, 1:] = heatmap[:, :-1]
                map_bottom[:, :-1] = heatmap[:, 1:]
                #閾値，上下左右のピクセルと比較して，上回ればそのピクセルを１にする配列
                peaks_binary = xp.logical_and.reduce((
                    heatmap > constants['heatmap_peak_thresh'],
                    heatmap > map_left,
                    heatmap > map_right,
                    heatmap > map_top,
                    heatmap > map_bottom,
                ))

                peaks = zip(xp.nonzero(peaks_binary)[1], xp.nonzero(peaks_binary)[0])  # [(x, y), (x, y)...]のpeak座標配列
                peaks_with_score = [(i,) + peak_pos + (heatmap[peak_pos[1], peak_pos[0]],) for peak_pos in peaks]
                peaks_id = range(peak_counter, peak_counter + len(peaks_with_score))
                peaks_with_score_and_id = [peaks_with_score[i] + (peaks_id[i], ) for i in range(len(peaks_id))]
                peak_counter += len(peaks_with_score_and_id)
                all_peaks.append(peaks_with_score_and_id)
            #[]
            all_peaks = xp.array([peak for peaks_each_category in all_peaks for peak in peaks_each_category])
        else:#cupyならば
            heatmaps = F.convolution_2d(heatmaps[:, None], self.gaussian_kernel,
                                        stride=1, pad=int(constants['ksize']/2)).data.squeeze()
            left_heatmaps = xp.zeros(heatmaps.shape)
            right_heatmaps = xp.zeros(heatmaps.shape)
            top_heatmaps = xp.zeros(heatmaps.shape)
            bottom_heatmaps = xp.zeros(heatmaps.shape)
            left_heatmaps[:, 1:, :] = heatmaps[:, :-1, :]
            right_heatmaps[:, :-1, :] = heatmaps[:, 1:, :]
            top_heatmaps[:, :, 1:] = heatmaps[:, :, :-1]
            bottom_heatmaps[:, :, :-1] = heatmaps[:, :, 1:]

            peaks_binary = xp.logical_and(heatmaps > constants['heatmap_peak_thresh'], heatmaps >= right_heatmaps)
            peaks_binary = xp.logical_and(peaks_binary, heatmaps >= top_heatmaps)
            peaks_binary = xp.logical_and(peaks_binary, heatmaps >= left_heatmaps)
            peaks_binary = xp.logical_and(peaks_binary, heatmaps >= bottom_heatmaps)

            peak_c, peak_y, peak_x = xp.nonzero(peaks_binary)
            peak_score = heatmaps[peak_c, peak_y, peak_x]
            all_peaks = xp.vstack((peak_c, peak_x, peak_y, peak_score)).transpose()
            all_peaks = xp.hstack((all_peaks, xp.arange(len(all_peaks)).reshape(-1, 1)))
            all_peaks = all_peaks.get()
        return all_peaks

    #Hungarian Algorithmを用いた最適な接続セットの出力関数，オリジナル
    def compute_optimal_connections(self,paf,cand_a,cand_b,img_len,constants):
        cost_matrix = np.zeros((len(cand_a),len(cand_b)))
        connections = np.zeros((0,3))
        for a, joint_a in enumerate(cand_a):
            for b, joint_b in enumerate(cand_b):  # joint = [x,y,heatmap(x,y),peak_id]
                vector = joint_b[:2] - joint_a[:2]
                norm = np.linalg.norm(vector)
                if norm == 0:
                    continue

                ys = np.linspace(joint_a[1], joint_b[1], num=constants['n_integ_points'])
                xs = np.linspace(joint_a[0], joint_b[0], num=constants['n_integ_points'])#n_integ_points分だけ分割する
                integ_points = np.stack([ys, xs]).T.round().astype('i')  # joint_aとjoint_bの2点間を結ぶ線分上の座標点 [[x1, y1], [x2, y2]...]
                paf_in_edge = np.hstack([paf[0][np.hsplit(integ_points, 2)], paf[1][np.hsplit(integ_points, 2)]])
                unit_vector = vector / norm
                inner_products = np.dot(paf_in_edge, unit_vector)

                integ_value = inner_products.sum() / len(inner_products)#pafを線積分した時の値を
                # vectorの長さが基準値以上の時にペナルティを与える
                integ_value_with_dist_prior = integ_value + min(constants['limb_length_ratio'] * img_len / norm - constants['length_penalty_value'], 0)
                n_valid_points = sum(inner_products > constants['inner_product_thresh'])
                #cand_a,cand_bによるcost_matrixの作成
                if n_valid_points > constants['n_integ_points_thresh'] and integ_value_with_dist_prior > 0:
                    cost_matrix[a,b] = integ_value_with_dist_prior
        #cost_matrixの中身が存在するとき
        if (cost_matrix != np.zeros((len(cand_a),len(cand_b)))).any():
            #Hungarian algorithmで最適な組み合わせを選出
            a_ind, b_ind = linear_sum_assignment(-cost_matrix)
            for i in range(len(a_ind)):
                if cost_matrix[a_ind[i],b_ind[i]]!=0:
                    Z = np.array([int(cand_a[a_ind[i],3]), int(cand_b[b_ind[i],3]), cost_matrix[a_ind[i],b_ind[i]]])
                    connections = np.vstack([connections,Z])
            connections = sorted(connections, key=lambda x: x[2], reverse=True)
        return connections

    #pafと全peakから接続を計算する関数，オリジナル
    def compute_connections(self, pafs, all_peaks, img_len, params):
        all_connections = []
        for i in range(len(params['limbs_point'])):
            paf_index = [i*2, i*2 + 1]
            paf = pafs[paf_index]
            limb_point = params['limbs_point'][i]
            cand_a = all_peaks[all_peaks[:, 0] == limb_point[0]][:, 1:]
            cand_b = all_peaks[all_peaks[:, 0] == limb_point[1]][:, 1:]

            if len(cand_a) > 0 and len(cand_b) > 0:
                candidate_connections = self.compute_optimal_connections(paf, cand_a, cand_b, img_len, constants)
                all_connections.append(candidate_connections)
            else:
                all_connections.append(np.zeros((0, 3)))
        return all_connections

    #キーポイントのグルーピング関数，流用
    #https://github.com/DeNA/Chainer_Realtime_Multi-Person_Pose_Estimation/blob/master/pose_detector.py
    def grouping_key_points(self, all_connections, candidate_peaks, params):
        subsets = -1 * np.ones((0, 20))

        for l, connections in enumerate(all_connections):
            joint_a, joint_b = params['limbs_point'][l]

            for i in range(len(connections)):
                ind_a, ind_b, score = int(connections[i][0]), int(connections[i][1]),connections[i][2]

                joint_found_cnt = 0
                joint_found_subset_index = [-1, -1]
                for subset_ind, subset in enumerate(subsets):
                    # そのconnectionのjointをもってるsubsetがいる場合
                    if subset[joint_a] == ind_a or subset[joint_b] == ind_b:
                        joint_found_subset_index[joint_found_cnt] = subset_ind
                        joint_found_cnt += 1

                if joint_found_cnt == 1: # そのconnectionのどちらかのjointをsubsetが持っている場合
                    found_subset = subsets[joint_found_subset_index[0]]
                    # 肩->耳のconnectionの組合せを除いて、始点の一致しか起こり得ない。肩->耳の場合、終点が一致していた場合は、既に顔のbone検出済みなので処理不要。
                    if found_subset[joint_b] != ind_b:
                        found_subset[joint_b] = ind_b
                        found_subset[-1] += 1 # increment joint count
                        found_subset[-2] += candidate_peaks[ind_b, 3] + score  # joint bのscoreとconnectionの積分値を加算

                elif joint_found_cnt == 2: # subset1にjoint1が、subset2にjoint2がある場合(肩->耳のconnectionの組合せした起こり得ない)
                    # print('limb {}: 2 subsets have any joint'.format(l))
                    found_subset_1 = subsets[joint_found_subset_index[0]]
                    found_subset_2 = subsets[joint_found_subset_index[1]]

                    membership = ((found_subset_1 >= 0).astype(int) + (found_subset_2 >= 0).astype(int))[:-2]
                    if not np.any(membership == 2):  # merge two subsets when no duplication
                        found_subset_1[:-2] += found_subset_2[:-2] + 1 # default is -1
                        found_subset_1[-2:] += found_subset_2[-2:]
                        found_subset_1[-2:] += score  # connectionの積分値のみ加算(jointのscoreはmerge時に全て加算済み)
                        subsets = np.delete(subsets, joint_found_subset_index[1], axis=0)
                    else:
                        if found_subset_1[joint_a] == -1:
                            found_subset_1[joint_a] = ind_a
                            found_subset_1[-1] += 1
                            found_subset_1[-2] += candidate_peaks[ind_a, 3] + score
                        elif found_subset_1[joint_b] == -1:
                            found_subset_1[joint_b] = ind_b
                            found_subset_1[-1] += 1
                            found_subset_1[-2] += candidate_peaks[ind_b, 3] + score
                        if found_subset_2[joint_a] == -1:
                            found_subset_2[joint_a] = ind_a
                            found_subset_2[-1] += 1
                            found_subset_2[-2] += candidate_peaks[ind_a, 3] + score
                        elif found_subset_2[joint_b] == -1:
                            found_subset_2[joint_b] = ind_b
                            found_subset_2[-1] += 1
                            found_subset_2[-2] += candidate_peaks[ind_b, 3] + score

                elif joint_found_cnt == 0 and l != 9 and l != 13: # 新規subset作成, 肩耳のconnectionは新規group対象外
                    row = -1 * np.ones(20)
                    row[joint_a] = ind_a
                    row[joint_b] = ind_b
                    row[-1] = 2
                    row[-2] = sum(candidate_peaks[[ind_a, ind_b], 3]) + score
                    subsets = np.vstack([subsets, row])
                elif joint_found_cnt >= 3:
                    pass

        # delete low score subsets
        keep = np.logical_and(subsets[:, -1] >= params['n_subset_limbs_thresh'], subsets[:, -2]/subsets[:, -1] >= params['subset_score_thresh'])
        subsets = subsets[keep]
        return subsets

    def subsets_to_pose_array(self, subsets, all_peaks):
        person_pose_array = []
        for subset in subsets:
            joints = []
            for joint_index in subset[:18].astype('i'):
                if joint_index >= 0:
                    joint = all_peaks[joint_index][1:3].tolist()
                    joint.append(2)
                    joints.append(joint)
                else:
                    joints.append([0, 0, 0])
            person_pose_array.append(np.array(joints))
        person_pose_array = np.array(person_pose_array)
        return person_pose_array

    #クラスCall関数
    def __call__(self,img):
        edit_img = img.copy()
        img_h,img_w ,_ = edit_img.shape
        #画像とheatmapの大きさの最適化(stride=8の倍数にする)
        input_w,input_h = self.compute_optimal_size(edit_img,constants['img_size'])
        map_w,map_h = self.compute_optimal_size(edit_img,constants['heatmap_size'])
        #画像サイズの更新と学習器に入れるためにデータの編集
        resized_image = cv2.resize(edit_img, (input_w, input_h))
        x_data = self.preprocess(resized_image)
        #GPUへの適用
        if self.device >= 0:
            x_data = cuda.to_gpu(x_data)
        #学習器からの出力(全ステージから)
        Ss,Ls = self.model(x_data)
        #最終ステージの物のみ取り出す
        heatmaps = F.resize_images(Ss[-1], (map_h, map_w)).data[0]
        pafs = F.resize_images(Ls[-1], (map_h, map_w)).data[0]

        if self.device >= 0:
            pafs = pafs.get()
            cuda.get_device_from_id(self.device).synchronize()
        #heatmapからPeakを計算する
        all_peaks = self.compute_peaks_from_heatmaps(heatmaps)
        if len(all_peaks) == 0:
            return np.empty((0, len(JointType), 3)), np.empty(0)
        #peakとpafからConnectionを計算する
        all_connections = self.compute_connections(pafs, all_peaks, map_w, constants)
        #subsetの作成
        subsets = self.grouping_key_points(all_connections, all_peaks, constants)
        all_peaks[:, 1] *= img_w / map_w
        all_peaks[:, 2] *= img_h / map_h
        #poseの計算
        poses = self.subsets_to_pose_array(subsets, all_peaks)
        return poses

#入力引数の設定関数
def set_args():
    parser = argparse.ArgumentParser(description='Pose_detection_demonstration')
    parser.add_argument('--weight',default = './models/coco_posenet.npz')
    parser.add_argument('--img')
    parser.add_argument('--gpu','-g',type = int,default = -1)
    args = parser.parse_args()

    return args

#画像へのPoseの描画関数
def person_pose(img, poses):
    if len(poses) == 0:
        return img

    canvas = img.copy()

    # limbs
    for pose in poses.round().astype('i'):
        for i, (limb, color) in enumerate(zip(constants['limbs_point'], constants['limb_colors'])):
            if i != 9 and i != 13:  # don't show ear-shoulder connection
                limb_ind = np.array(limb)
                if np.all(pose[limb_ind][:, 2] != 0):
                    joint1, joint2 = pose[limb_ind][:, :2]
                    cv2.line(canvas, tuple(joint1), tuple(joint2), color, 2)

    # joints
    for pose in poses.round().astype('i'):
        for i, ((x, y, v), color) in enumerate(zip(pose, constants['joint_colors'])):
            if v != 0:
                cv2.circle(canvas, (x, y), 3, color, -1)
    return canvas

#文字列の挿入関数
def str_insert(repl, string, pos):
    front = string[:pos]
    rear = string[pos:]
    return front+repl+rear

#main関数
def main():
    t_start = time.time()
    args = set_args()

    #modelの設定
    model = Network2.MyNet()

    #npzファイルの読み出し
    serializers.load_npz(args.weight,model)
    t = time.time()-t_start
    print('モデル読み出し    ',f"経過時間[s]：{t}")

    #画像の読み出し
    img = cv2.imread(args.img)
    t = time.time()-t_start
    print('入力画像読み出し ',f"経過時間[s]：{t}")

    #pose_detectクラスの作成
    PoseDetect = pose_detect(model,device=args.gpu)

    #pose_detectクラス関数Callの呼び出し
    poses = PoseDetect(img)
    t = time.time()-t_start
    print('poseの計算  　　 ',f"経過時間[s]：{t}")

    #PoseのImgへの出力
    img = person_pose(img,poses)
    save_dir = str_insert('_result',args.img,-4)
    cv2.imwrite(save_dir,img)
    t = time.time()-t_start
    print('poseの画像出力  ',f"経過時間[s]：{t}")
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
