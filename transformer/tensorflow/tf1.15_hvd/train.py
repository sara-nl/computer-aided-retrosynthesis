import os
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import horovod.tensorflow as hvd
import tensorflow as tf

from data.preprocessing import data_generator, val_generator
from callbacks import GenCallback, LRTensorBoard, SaveModelCallback, CyclicLR


def train(mdl, opts):
    train_file = opts.train

    NTRAIN = sum(1 for _ in open(train_file))
    print("Number of points: ", NTRAIN)

    if opts.exp_range_schedule:
        lr_callback = CyclicLR(base_lr=0.001, max_lr=0.009, step_size=2200, mode='exp_range', gamma=0.99994)
    else:
        lr_callback = GenCallback(opts)

    if opts.horovod:
        callback = [hvd.keras.callbacks.BroadcastGlobalVariablesCallback(0),
                    SaveModelCallback(opts),
                    lr_callback]
        if hvd.rank() == 0:
            callback.append(LRTensorBoard(log_dir=opts.output_dir, update_freq='epoch', write_graph=False))

        steps_per_epoch = int((math.ceil(NTRAIN / opts.batch_size)) / hvd.size())

    else:
        callback = [SaveModelCallback(opts),
                    lr_callback,
                    LRTensorBoard(log_dir=opts.output_dir, update_freq='epoch', write_graph=False)]

        steps_per_epoch = int(math.ceil(NTRAIN / opts.batch_size))

    val_dataset = val_generator()

    history = mdl.fit_generator(generator=data_generator(train_file, opts.batch_size),
                                steps_per_epoch=steps_per_epoch,
                                epochs=opts.epochs,
                                use_multiprocessing=True,
                                shuffle=True,
                                validation_data=val_dataset,
                                verbose=1 if not opts.horovod or (opts.horovod and hvd.rank() == 0) else 0,
                                callbacks=callback)

    if not opts.horovod or (opts.horovod and hvd.rank() == 0):
        print("Averaging weights")
        f = []

        for i in opts.epochs_to_save:
            f.append(h5py.File(os.path.join(opts.output_dir, "tr-" + str(i) + ".h5"), "r+"))

        keys = list(f[0].keys())
        for key in keys:
            groups = list(f[0][key])
            if len(groups):
                for group in groups:
                    items = list(f[0][key][group].keys())
                    for item in items:
                        data = []
                        for i in range(len(f)):
                            data.append(f[i][key][group][item])
                        avg = np.mean(data, axis=0)
                        del f[0][key][group][item]
                        f[0][key][group].create_dataset(item, data=avg)
        for fp in f:
            fp.close()

        for i in opts.epochs_to_save[1:]:
            os.remove(os.path.join(opts.output_dir, "tr-" + str(i) + ".h5"))
        os.rename(os.path.join(opts.output_dir, "tr-" + str(opts.epochs_to_save[0]) + ".h5"),
                  os.path.join(opts.output_dir, "final.h5"))

        print(f"Final weights are in the file: {opts.output_dir}/final.h5")

        # summarize history for accuracy
        plt.plot(history.history['masked_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f"{opts.output_dir}/accuracy.pdf")

        plt.clf()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(f"{opts.output_dir}/loss.pdf")

        plt.clf()
