cd models
wget http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel
python convert_model.py posenet pose_iter_440000.caffemodel coco_posenet.npz
cd ..
