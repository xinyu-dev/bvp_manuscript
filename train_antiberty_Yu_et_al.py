"""
Yu et al
Example training script using Antiberty embedding
"""


import tensorflow as tf
import pickle, time
from tensorflow import keras
from tensorflow.data import Dataset
from keras.applications import EfficientNetV2B0
from keras import layers
import numpy as np
from loguru import logger


def get_dataset_partitions(ds, ds_size, train_split=0.8, val_split=0.1,
                           include_test = True, test_split=0.1, seed = 12345):
    """
    Generic function to shuffle and partition a tf.data.Dataset object into train, val, test
    :param ds: tf.data.Dataset
    :param ds_size: int, size of the dataset (i.e. # of data points)
    :param train_split: float, percentage of the training data.
    :param val_split: float, percentage of the validation data
    :param include_test: bool, whether to include the test partition.
    :param test_split: float, percentage of test data. Ignored when include_test if False
    return tuple of tf.data.Dataset objects: (train, val) or (train, val, test)
    """
    if include_test:
        sum_of_split = train_split + test_split + val_split
        assert np.isclose(1,sum_of_split, atol=0.001), f'train_split = {train_split}, val_split = {val_split}, test_split={test_split}, sum = {sum_of_split}, does not add up to 1'
    else:
        sum_of_split = train_split + val_split
        assert np.isclose(1,sum_of_split, atol= 0.001), f'train_split = {train_split}, val_split = {val_split}, sum = {sum_of_split}, does not add up to 1'

    ds = ds.shuffle(ds_size, seed=seed, reshuffle_each_iteration= False)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    # split train, val, test
    if include_test:
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)
        return train_ds, val_ds, test_ds

    # split train, val
    else:
        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size)
        return train_ds, val_ds

def load_data(input1_pkl_fp, input2_pkl_fp, input3_pkl_fp, input4_pkl_fp, labels_pkl_fp, batch_size = 128):
    """
    Function to load data
    :param input1_pkl_fp: str, file path to input1 (sequence embeddings)
    :param input2_pkl_fp: str, file path to input2 (integer encoded coating)
    :param input3_pkl_fp: str, file path to input3 (mAb concentration)
    :param input3_pkl_fp: str, file path to input4 (var loc)
    :param labels_pkl_fp: str, file path to labels
    :param batch_size: int, batch size
    :return: tuple
    """

    input1 = pickle.load(open(input1_pkl_fp, 'rb')) # seq
    input2 = pickle.load(open(input2_pkl_fp, 'rb')) # coating
    input3 = pickle.load(open(input3_pkl_fp, 'rb')) # mAb conc
    input4 = pickle.load(open(input4_pkl_fp, 'rb')) # var loc
    labels = pickle.load(open(labels_pkl_fp, 'rb'))
    
    # get number of unique coatings
    n_coatings = len(np.unique(input2))
    
    # get number of unique var loc categories
    n_var_locations = len(np.unique(input4))
    
    # construct dataset
    data = Dataset.from_tensor_slices((
        {
            'input1': input1, # seq embeddings
            'input2': input2, # coating,
            'input3': input3, # conc, 
            'input4': input4 # var loc
        },
        labels
    ))

    # get partition
    train_data, val_data, test_data = get_dataset_partitions(
        ds = data,
        ds_size = labels.shape[0],
        train_split = 0.8,
        val_split = 0.1,
        test_split = 0.1,
        include_test = True, 
        seed = 222222
    )
    # get number of data points in each partition, before batching. After batching, the len() returns the number of batches
    train_data_size = len(train_data)
    val_data_size = len(val_data)
    test_data_size = len(test_data)

    # shuffle the train data every epoch
    train_data = train_data.shuffle(len(train_data), reshuffle_each_iteration=True)

    # batch all data
    train_data = train_data.batch(batch_size = batch_size)
    val_data  = val_data.batch(batch_size = batch_size)
    test_data = test_data.batch(batch_size = batch_size)

    # check shape
    for input_dict, label in train_data:
        input1_shape = input_dict['input1'].shape
        input2_shape = input_dict['input2'].shape
        input3_shape = input_dict['input3'].shape
        input4_shape = input_dict['input4'].shape
        label_shape = label.shape
        break

    return train_data, val_data, test_data, input1_shape, input2_shape, input3_shape, input4_shape, label_shape, train_data_size, val_data_size, test_data_size, n_coatings, n_var_locations


def linear_warmup_scheduler(epoch, lr,
                            stage1_start_lr = 1e-5,
                            stage1_end_lr = 1e-3,
                            stage1_epochs = 10,
                            stage2_end_lr = 1e-3,
                            stage2_epochs = 10,
                            stage3_epochs = 10,
                            alpha = 0
                            ):
    # stage 1
    if epoch < stage1_epochs:
        delta = (stage1_end_lr - stage1_start_lr)/stage1_epochs
        return lr + delta

    # stage 2
    if epoch < stage2_epochs + stage1_epochs:
        delta = (stage2_end_lr - stage1_end_lr)/stage2_epochs
        return lr + delta

    # stage 3
    else:
        n = epoch - (stage1_epochs + stage2_epochs)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * (n) / stage3_epochs))
        delta = (1 - alpha) * cosine_decay + alpha
        return stage2_end_lr * delta


if __name__ == '__main__':
    
    logger.add("logs/logs.log", rotation = '1 week')
    logger.info("Starting")
    
    start_time = time.time()
    
    assert len(tf.config.list_physical_devices('GPU')) != 0 , 'GPU is not properly configured'
    logger.info("GPU recognized")
    
    input1_fp = 'inputs/input1.pkl'
    input2_fp = 'inputs/input2.pkl'
    input3_fp = 'inputs/input3.pkl'
    input4_fp = 'inputs/input4.pkl'
    labels_fp = 'inputs/labels.pkl'

    # load data
    logger.info("Loading data")
    train_data, val_data, test_data, input1_shape, input2_shape, input3_shape, input4_shape, label_shape, train_data_size, val_data_size, test_data_size, n_coatings, n_var_locations = load_data(
        input1_pkl_fp = input1_fp,
        input2_pkl_fp = input2_fp,
        input3_pkl_fp = input3_fp,
        input4_pkl_fp = input4_fp,
        labels_pkl_fp = labels_fp,
        batch_size = 128
    )
    logger.success(f"Data loaded. Train size: {train_data_size}, validation size: {val_data_size} ,test size: {test_data_size}")

    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
    
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]) # add any available GPU here
    with strategy.scope():
        # build base
        base = EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape= (536, 512, 3), # (536, 512, 3) for antiberty
            pooling='avg',
            include_preprocessing=False,
        )

        base.trainable = True
        input1_layer = keras.Input(shape=input1_shape[1:], name = 'input1')
        x1 = tf.expand_dims(input1_layer, axis = -1)
        x1 = layers.Conv2D(filters=3, strides = 1, padding = 'same', kernel_size = 3)(x1)
        x1 = base(x1)

        # embedded coating
        input2_layer = keras.Input(shape = input2_shape[1:], name = 'input2')
        x2 = input2_layer

        # concentration
        input3_layer = keras.Input(shape = input3_shape[1:], name = 'input3')
        x3 = input3_layer

        # variable region location
        input4_layer = keras.Input(shape = input4_shape[1:], name = 'input4')
        x4 = keras.layers.Embedding(input_dim = n_var_locations, output_dim = 3)(input4_layer)
        x4 = layers.Flatten()(x4)

        # concatenate
        x = layers.Concatenate()([x1, x2, x3,x4])
        x = layers.Dense(64, kernel_initializer = 'he_uniform', activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, kernel_initializer = 'he_uniform', activation='sigmoid')(x)
        
        # build
        model = keras.Model(
            inputs = [input1_layer, input2_layer, input3_layer, input4_layer],
            outputs = [outputs]
        )
        logger.success("Model successfully built")

        # compile
        model.compile(
            optimizer= keras.optimizers.Adam(learning_rate = 1e-5) ,
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC()]
        )
        logger.success("Model successfully compiled")
    
    # callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint("models/model-{epoch:03d}-{val_auc:.3f}.h5", save_best_only=False, monitor='val_auc', mode = 'max', save_freq='epoch'),
        keras.callbacks.EarlyStopping(patience = 10, min_delta=0.0005, restore_best_weights = True),
        keras.callbacks.LearningRateScheduler(linear_warmup_scheduler),
        keras.callbacks.TensorBoard(log_dir='logs/log')
    ]
    
    logger.info("training started")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        callbacks = callbacks,
        batch_size = None,
    )
    
    end_time = time.time()
    logger.success(f"Done!. Total time it took is { np.round((end_time - start_time)/60/60, 2)} hours")
    
    
