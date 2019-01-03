### Training
```bash
$ cd code
$ python train.py --model beenburger --save_dir DIR_NAME --data_name coco 2>&1 | tee ../log/logs.txt
```

<br/>

### If files are not exist, download them by using below commands
### Download pretrained w2v embedding files
```bash
$ wget -P ./w2v/ http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
$ wget -P ./w2v/ http://www.cs.toronto.edu/~rkiros/models/utable.npy
$ wget -P ./w2v/ http://www.cs.toronto.edu/~rkiros/models/btable.npy
```

<br/>

### Make coco vocabulary
```bash
$ cd code/preprocess
$ python vocab.py
```

<br/>

### Make coco id list
```bash
$ cd code/preprocess
$ python make_npy.py
```

<br/>

### Download initial-pretrained resnet + 1x1 conv layers
* [init.pth](https://drive.google.com/open?id=1ldRO9LzTg2_1HPlqA1flpK7T0QEGBmHM)

