import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from common import set_seed
from common import get_logger
from common import get_session
from common import get_arguments
from common import search_same
from common import create_stamp
from dataloader import set_dataset
from dataloader import DataLoader
from model import SimSiam
from model import set_lincls
from callback import OptionalLearningRateSchedule
from callback import create_callbacks

import tensorflow as tf


def train_pretext(args, logger, initial_epoch, strategy, num_workers):
    ##########################
    # Dataset
    ##########################
    trainset, valset = set_dataset(args.task, args.dataset, args.data_path)
    steps_per_epoch = args.steps or len(trainset) // args.batch_size

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== TRAINSET ==========")
    logger.info(f"    --> {len(trainset)}")
    logger.info(f"    --> {steps_per_epoch}")

    logger.info("=========== VALSET ===========")
    logger.info(f"    --> {len(valset)}")


    ##########################
    # Model & Generator
    ##########################
    with strategy.scope():
        model = SimSiam(args, logger, num_workers=num_workers)

        lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr_scheduler, momentum=.9),
            loss=tf.keras.losses.cosine_similarity,
            run_eagerly=False)

    train_generator = DataLoader(args, args.task, 'train', trainset, args.batch_size, num_workers).dataloader


    ##########################
    # Train
    ##########################
    callbacks, initial_epoch = create_callbacks(args, logger, initial_epoch)
    if callbacks == -1:
        logger.info('Check your model.')
        return
    elif callbacks == -2:
        return

    model.fit(
        train_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,)


def train_lincls(args, logger, initial_epoch, strategy, num_workers):
    # assert args.snapshot is not None, 'pretrained weight is needed!'
    ##########################
    # Dataset
    ##########################
    trainset, valset = set_dataset(args.task, args.dataset, args.data_path)
    steps_per_epoch = args.steps or len(trainset) // args.batch_size
    validation_steps = len(valset) // args.batch_size

    logger.info("TOTAL STEPS OF DATASET FOR TRAINING")
    logger.info("========== TRAINSET ==========")
    logger.info(f"    --> {len(trainset)}")
    logger.info(f"    --> {steps_per_epoch}")

    logger.info("=========== VALSET ===========")
    logger.info(f"    --> {len(valset)}")
    logger.info(f"    --> {validation_steps}")


    ##########################
    # Model & Generator
    ##########################
    train_generator = DataLoader(args, args.task, 'train', trainset, args.batch_size, num_workers).dataloader
    val_generator = DataLoader(args, args.task, 'val', valset, args.batch_size, num_workers).dataloader
        
    with strategy.scope():
        backbone = SimSiam(args, logger)
        model = set_lincls(args, backbone.encoder)

        lr_scheduler = OptionalLearningRateSchedule(args, steps_per_epoch, initial_epoch)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(lr_scheduler, momentum=.9),
            metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(1, 'acc1', dtype=tf.float32),
                     tf.keras.metrics.SparseTopKCategoricalAccuracy(5, 'acc5', dtype=tf.float32)],
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='loss'),
            run_eagerly=False)


    ##########################
    # Train
    ##########################
    callbacks, initial_epoch = create_callbacks(args, logger, initial_epoch)
    if callbacks == -1:
        logger.info('Check your model.')
        return
    elif callbacks == -2:
        return

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps)


def main():
    set_seed()
    args = get_arguments()
    if args.task == 'pretext':
        if args.dataset == 'imagenet':
            args.lr = 0.5 * float(args.batch_size / 256)
        elif args.dataset == 'cifar10':
            args.lr = 0.03 * float(args.batch_size / 256)
    else:
        if args.dataset == 'imagenet' and args.freeze:
            args.lr = 30. * float(args.batch_size / 256)
        else:# args.dataset == 'cifar10':
            args.lr = 1.8 * float(args.batch_size / 256)

    args, initial_epoch = search_same(args)
    if initial_epoch == -1:
        # training was already finished!
        return

    elif initial_epoch == 0:
        # first training or training with snapshot
        args.stamp = create_stamp()

    get_session(args)
    logger = get_logger("MyLogger")
    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))


    ##########################
    # Strategy
    ##########################
    if len(args.gpus.split(',')) > 1:
        # strategy = tf.distribute.experimental.CentralStorageStrategy()
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

    num_workers = strategy.num_replicas_in_sync
    assert args.batch_size % num_workers == 0

    logger.info('{} : {}'.format(strategy.__class__.__name__, num_workers))
    logger.info("BATCH SIZE PER REPLICA : {}".format(args.batch_size // num_workers))


    ##########################
    # Training
    ##########################
    if args.task == 'pretext':
        train_pretext(args, logger, initial_epoch, strategy, num_workers)
    else:
        train_lincls(args, logger, initial_epoch, strategy, num_workers)
    
    
if __name__ == "__main__":
    main()