
------3/6/2022 Ju Wang -----------------
newpcamd

train.py
	removed arch
	mkdir ckpt

	python train.py --model global_coord_latent_nl --num_frames 16 --logname experiment_name --batch_size 12 --coord_feature_dim 256 --root_frames ../20bn-something-something-v2-frames --json_data_train dataset_splits/compositional/train.json 
                   --json_data_val dataset_splits/compositional/validation.json 
                   --json_file_labels dataset_splits/compositional/labels.json
                   --tracked_boxes ../detected_annotations/combined_annonations_compositional.json
	
Computer Req:
	Mem >= 32 GB
	GPU > 4GB
	dataset > 500 GB

Time:
	epoch: 10 mins

resnet:
	model/pretrained_weights/kinetics-res50.pth
	automatic download if use certain model?
	the Globalxxx model use that.
	resnet3d_xl.py

model:
	default model is coord, VideoModelCoord
	try global_coord_latent_nl, VideoModelGlobalCoordLatentNL
	output classification is one-hot encode of video action class
	(b, num_classes)
------
run1: 16, 72, 512 model c_l_l experiment_name3
cd code
python train.py --model coord_latent_nl --num_frames 16 --logname experiment_name3 --resume ckpt/_experiment_name_latest.pth.tar --batch_size 72 --coord_feature_dim 512 --root_frames ../frames --json_data_train dataset_splits/compositional/train.json --json_data_val dataset_splits/compositional/validation.json --json_file_labels dataset_splits/compositional/labels.json --tracked_boxes ../detected_annotations/combined_annonations_compositional.json
	
batch time (avg time)
	Test: [400/401]	Time 2.235 (1.023)	Loss 3.8279 (3.7606)	
	Acc1 16.0 (21.7): 21.7 avg acc for top 1
	Acc5 41.7 (43.9): 43.9 avg acc for top 5

run2: 16, 72, 256 model c_l_l experiment_name4
--model coord_latent_nl --num_frames 16 --logname experiment_name4  --batch_size 72 --coord_feature_dim 256
about same as run1


run3: 8, 72, 256 model c_l_l lenova1 experiment_name7
	Time: 0.22 (0.29) Data 0.2 (0.19)
	Acc5 50.0 (53.4)
run4: 8, 72, 256 model c_l_l experiment_name7
	Epoch: [49][760/762]	Time 0.042 (0.265)	Data 0.001 (0.184)
	Loss 3.2759 (3.5507)	Acc1 20.8 (18.8)	Acc5 54.2 (45.1)

run4: 8, 72, 512 model c_l_l experiment_name6
	Loss 3.5118 (3.4050)	Acc1 25.0 (21.8)	Acc5 50.0 (48.4)

run5: 8, 36, 256 model c_l_l lenova1 experiment_name8, 6hr 5min
	Epoch: [49][1520/1525]	Time 0.057 (0.149)	Data 0.024 (0.086)
	Loss 3.7628 (3.3877)	Acc1 13.9 (21.6)	Acc5 36.1 (49.0)

run6: 8, 18, 256 model c_l_l lenova1 experiment_name9, 6hr 5min

---------------------
b: 72 batch size
3: image channels rgb
16: nr_frames, = num_frames?
4: nr_boxes

global_img_tensors, box_tensors, box_categories, video_label.shape torch.Size([72, 3, 16, 224, 224]) torch.Size([72, 8, 4, 4]) torch.Size([72, 8, 4]) torch.Size([72])

performance:
I reproduce the coord_latent in composition split by setting num_frames=8, batch_size=72, and coord_feature_dim=512, and get 50.1. It is still a bit lower than the reported performance, 51.3.

I have tried global_i3d with num_frames = 16, and the result is 50.2. Have you got a reasonable result with 8 frames? 
They trained the I3D baseline with 16 frames, which is not claimed in the paper. And I only obtained 40% on I3D with 8 frames.`

see performance.txt

-------------------------------------------------
# The Something-Else Annotations
This repository provides instructions regarding the annotations used in the paper: 'Something-Else: Compositional Action Recognition with Spatial-Temporal Interaction Networks' (https://arxiv.org/abs/1912.09930).
We collected annotations for 180049 videos from the Something-Something Dataset (https://20bn.com/datasets/something-something), that include per frame bounding box annotation for each object and hand in the human-object interaction in the video.

The file containing annotations can be downloaded from:

https://drive.google.com/open?id=1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ in four parts,
it containes a dictionary mapping each video id, the name of the video file to the list of per-frame annotations. The annotations assume that the frame rate of the videos is 12.
An example of per-frame annotation is shown below, the names and number of "something's" in the frame correspond to the fields
'gt_placeholders' and 'nr_instances', the frame path is given in the field 'name', 'labels' is a list of object's and hand's bounding boxes and names.

```
   [
    {'gt_placeholders': ['pillow'],
     'labels': [{'box2d': {'x1': 97.64950730138266,
                          'x2': 427,
                          'y1': 11.889318166856967,
                          'y2': 239.92858832368972},
                          'category': 'pillow',
                          'standard_category': '0000'}},
                {'box2d': {'x1': 210.1160330781122,
                          'x2': 345.4329005999551,
                          'y1': 78.65516045335991,
                          'y2': 209.68758889799403},
                          'category': 'hand',
                          'standard_category': 'hand'}}],
     'name': '2/0001.jpg',
     'nr_instances': 2}, 
     {...},
     ...
     {...},
     ]
```
The annotations for example videos are a small subset of the annotation file, and can be found in `annotations.json`.

# Citation
If you use our annotations in your research or wish to refer to the baseline results, please use the following BibTeX entry.
```
@inproceedings{CVPR2020_SomethingElse,
  title={Something-Else: Compositional Action Recognition with Spatial-Temporal Interaction Networks},
  author={Materzynska, Joanna and Xiao, Tete and Herzig, Roei and Xu, Huijuan and Wang, Xiaolong and Darrell, Trevor},
  booktitle = {CVPR},
  year={2020}
}

@inproceedings{goyal2017something,
  title={The" Something Something" Video Database for Learning and Evaluating Visual Common Sense.},
  author={Goyal, Raghav and Kahou, Samira Ebrahimi and Michalski, Vincent and Materzynska, Joanna and Westphal, Susanne and Kim, Heuna and Haenel, Valentin and Fruend, Ingo and Yianilos, Peter and Mueller-Freitag, Moritz and others},
  booktitle={ICCV},
  volume={1},
  number={4},
  pages={5},
  year={2017}
}
```

# Dataset splits
The compositional, compositional shuffle, one-class compositional and few-shot splits of the Something Something v2 Dataset are available in the folder `dataset_splits`. 

# Visualization of the ground-truth bounding boxes
The folder `videos` contains example videos from the dataset and selected annotations file (full file available on google drive). To visualize videos with annotated bounding boxes run:

```python annotate_videos.py```

The annotated videos will be saved in the `annotated_videos` folder.

# Visualization of the detected bounding boxes 

![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/10015.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/130153.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/154439.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/174270.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/17628.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/21037.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/24719.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/31061.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/31156.gif)
![Output sample](https://github.com/joaanna/something_else/blob/master/videos/tracking_annotations/35176.gif)

# Training
Dataset prepare:
	(0) 
	(1) 20bn-something-something-v2-frames point to a folder contain all video folders with extracted frames.
	(2) ln -sn 20bn-something-something-v2-frames frames
	(3) mkdir detected_annotations/
		download annoation files from google drive:
		detected_compositional_part1.json
		detected_compositional_part2.json
		detected_compositional_part3.json
		merge three json to a single file
		 pip install jsonmerge
		 python jsonmergy.py		

To train the models from our paper run:
	cd code
	mkdir ckpt
	python train.py --model coord_latent_nl --num_frames 16 --logname experiment_name --batch_size 12 --coord_feature_dim 256 --root_frames ../20bn-something-something-v2-frames --json_data_train dataset_splits/compositional/train.json 
                   --json_data_val dataset_splits/compositional/validation.json 
                   --json_file_labels dataset_splits/compositional/labels.json
                   --tracked_boxes ../detected_annotations/combined_annonations_compositional.json

	status: loading label and annotation, error at 52701, should we combine all the detected+compositional_part1/2/3 to 1 file?

Place the data in the folder /path/to/frames each video bursted into frames in a separate folder. The ground-truth box annotations 
can be found in the google drive in parts and have to be concatenated in a single json file.

The models that are using appearance features are initialized with I3D network pre-trained on Kinetics, the checkpoint can be found in the google drive and should be placed in 'model/pretrained_weights/kinetics-res50.pth'.

We also provide some checkpoints to the trained models. To evaluate a model use the same script as for training with a flag `--args.evaluate` and path to the checkpoint `--args.resume /path/to/checkpoint`'

# Acknowledgments
We used parts of code from following repositories:
https://github.com/facebookresearch/SlowFast
https://github.com/TwentyBN/something-something-v2-baseline
