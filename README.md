This is the repository for the trained model in the paper: Players and Teams Detection in a Soccer Video Using Deep Learning-based Models.

The 3 trained models can be found in the "Model" folder. Using Yolov5-tph apply the weights listed in the folder on the SoccerTrack dataset.

Please to refer to [Yolov5-tph](https://github.com/cv516Buaa/tph-yolov5) for original model repository and [SoccerTrack](https://github.com/AtomScott/SportsLabKit) for original dataset.

# Installation
Download and install Yolov5-tph and download the SoccerTrack dataset
Please note that you do not need to install SportsLabKit since it may cause conflicts with the module versions, you only need to download the datset from Kaggle.
# Training
Yolov5-tph
``` bash
python train.py --img 1536 --adam --batch 4 --epochs 32 --data /content/data.yaml --weights yolov5m.pt --hy data/hyps/hyp.scratch-med.yaml --cfg Yolov5m-xs-tph.pt --name v5l-xs-tph
```
Yolov5-tph-plus
``` bash
python train.py --img 1536 --adam --batch 4 --epochs 34 --data /content/data.yaml --weights yolov5m.pt --hy data/hyps/hyp.scratch-med.yaml --cfg Yolov5m-tph-plus.pt --name v5l-tph-plus
```
# Detect
``` bash
python detect.py --weights Yolov5m-xs-tph.pt --img 1536 --conf 0.6 --iou-thres 0.7 --source F_20220220_1_1140_1170_Team1Corner.mp4 --save-txt --save-conf
```
