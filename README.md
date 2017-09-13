# TensorFlow-Slim image classification model library

Modiied TensorFlow-Slim image classfication repo for custom usage. The orginal repo is: 
[https://github.com/tensorflow/models/tree/master/slim](https://github.com/tensorflow/models/tree/master/slim )


## Usage

### Prepare for data set
put the images into DATASET\_DIR, which contains several sub-directories. For example: DATASET_DIR=sample\_data/flowers, and in DATASET\_DIR, it contains three folders:

```
daisy/
dandelion/
roses/
```


### To re-train the classficiation DNN:

```
DATASET_DIR=sample_data/flowers
TFRECORD_DIR=tf_records/flowersTRAIN_DIR=/tmp/train_logspython my_image_classifier.py  \
--train_dir=${TRAIN_DIR} \
--tfrecord_dir=${TFRECORD_DIR} \
--dataset_split_name=train \
--dataset_dir=${DATASET_DIR} \
--model_name=inception_v3
```

More options:

```
#run on cpu
--clone_on_cpu=1

# number of training steps
--max_number_of_steps=1000

```

The training and validaiton data and labels will be generated from the DATASET\_DIR and stored as TFRecords format in the TFRECORD\_DIR.  The training logs would be stored in the TRAIN\_DIR. 

### TensorBoard

To visualize the losses and other metrics during training, you can use TensorBoard by running the command below.

tensorboard --logdir=${TRAIN_DIR}
Once TensorBoard is running, navigate your web browser to http://localhost:6006





