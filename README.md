# TensorFlow-Slim image classification model library

Modiied TensorFlow-Slim image classfication repo for custom usage. The orginal repo is:
[https://github.com/tensorflow/models/tree/master/research/slim](https://github.com/tensorflow/models/tree/master/research/slim)


## Usage

```
./my_training.sh
```

Here is the basic description of the 'my_training.sh':

### Prepare for data set
put the images into DATASET\_DIR, which contains several sub-directories. For example: DATASET_DIR=sample\_data/flowers, and in DATASET\_DIR, it contains three folders:

```
daisy/
dandelion/
roses/
```


### To re-train the classficiation DNN:

```
# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=checkpoints

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=train_files

# Where the dataset is saved to.
DATASET_DIR=sample_data/flowers

# Where the TFrecords are saved to.
TFRECORD_DIR=tf_records/flowers


# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
  tar -xvf inception_v3_2016_08_28.tar.gz
  mv inception_v3.ckpt ${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt
  rm inception_v3_2016_08_28.tar.gz
fi


# Fine-tune only the new layers for 1000 steps.
python my_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --tfrecord_dir=${TFRECORD_DIR} \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

The training and validaiton data and labels will be generated from the DATASET\_DIR and stored as TFRecords format in the TFRECORD\_DIR.  The training logs would be stored in the TRAIN\_DIR.


### TensorBoard

To visualize the losses and other metrics during training, you can use TensorBoard by running the command below.

```
tensorboard --logdir=${TRAIN_DIR}
```

Once TensorBoard is running, navigate your web browser to http://localhost:6006



### Run the evaluation
```
VALID_DIR=valid_files
# Run evaluation.
python my_eval_classifier.py \
  --checkpoint_path=${VALID_DIR} \
  --eval_dir=${VALID_DIR} \
  --dataset_split_name=validation \
  --tfrecord_dir=${TFRECORD_DIR} \
  --model_name=inception_v3

```
