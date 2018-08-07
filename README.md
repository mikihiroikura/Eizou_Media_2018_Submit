# 2018年度映像メディア学　最終レポート
## コードの動かし方

1. 学習済みモデルのインポート  
`make_npz.bat`の実行  
**注意！**:　wgetをPCにインストール事前にしておかなければいけない．  
`./models/coco_posenet.npz`がインストールされていればOK.  

2. pose_detection_demo.pyの実行  
コマンドプロンプトに`python pose_detection_demo.py --img ./data/~~~`と入力する．  
`~~~`の部分は/dataディレクトリに保存しているPoseを出力したいPNGファイル名を入れる．  

3. 出力結果  
importされているOpenCVが出力画像をShowする．  
加えて，`/data/~~~_result.png`ファイルが作成されている．

## 学習に関して

今回は，以下の理由から学習済みモデルを使用する．  
1. HWの限界  
今回学習に使用する教師データは「COCO 2016 keypoints」であるが，作成したコード「train_pose.py」では，すべてのデータを学習器に入力すると，GPUのメモリがOverflowを起こし，動作が停止してしまった．目標の識別精度を達成するのに十分な学習を実現するためのHWの制約があった．  
  
2. 学習回数の時間的制約  
今回の学習済みモデルは440000回の学習を完了させているため十分な精度を達成できるが，そのための必要時間がHWの制約上実現が難しいと考えられる．  

  
  
以上より，以下の学習用コードもGithub上に挙げておくが上記の制約で動かなかった．  
1)gen_ignore_mask.py  
COCOデータセットからマスクを作成する関数．流用．(https://github.com/DeNA/Chainer_Realtime_Multi-Person_Pose_Estimation/blob/master/gen_ignore_mask.py)  
2)train_pose.py  
学習器．network_updator.pyを読み込み，学習を進める．  
3)network_updator.py  
オリジナルのLoss関数の作成と，Updatorの作成．
