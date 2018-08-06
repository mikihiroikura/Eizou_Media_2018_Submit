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
