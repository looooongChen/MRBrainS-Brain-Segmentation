from model import *
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.io import imsave



def model_fn_base(features, labels, mode, params, net_config, config):

    features = tf.cast(features, tf.float32)

    logits = make_model(features, mode == tf.estimator.ModeKeys.TRAIN, net_config)
    probs = tf.nn.softmax(logits, axis=3)
    preds = tf.argmax(logits, axis=3)


    #########################
    #### prediction mode ####
    #########################
    predictions = {
        "labels": preds,
        "probs": probs
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    #######################
    #### training mode ####
    #######################

    def exclude_batch_norm(name):
        return 'batch_normalization' not in name

    with tf.variable_scope('loss'):
        entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                labels=labels))
        l2_loss = config['weight_decay'] *\
                  tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32))
                            for v in tf.trainable_variables() if exclude_batch_norm(v.name)])
        loss = entropy + l2_loss

        accuracy = tf.metrics.accuracy(preds, labels)
        iou = tf.metrics.mean_iou(labels, preds, net_config['class_num'])

        eval_metrics = {'accuracy': accuracy,
                        "iou": iou}

    # training specification
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(config['init_lr'],
                                                   global_step,
                                                   10000, 0.85, staircase=False)
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=config['momentum'],
            use_nesterov=True
        )

        minimize_op = optimizer.minimize(loss, global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        sum_image = tf.summary.image('images', tf.expand_dims(features[:, :, :, 0], 3))
        sum_gt = tf.summary.image('ground_truth', tf.cast(tf.expand_dims(labels, 3) * 20, tf.uint8))
        sum_pred = tf.summary.image('prediction', tf.cast(tf.expand_dims(preds, 3) * 20, tf.uint8))
        eval_sum = tf.summary.merge([sum_image, sum_gt, sum_pred])
        eval_summary_hook = tf.train.SummarySaverHook(save_steps=1,
                                                      output_dir=config['model_dir'] + "/eval",
                                                      summary_op=eval_sum)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                          eval_metric_ops=eval_metrics,
                                          evaluation_hooks=[eval_summary_hook])


def train(net_config, training_config, train_ds, eval_ds):

    print("=========== training data ==========")
    print(train_ds[0].shape)
    print("=========== val data ==========")
    print(eval_ds[1].shape)
    print("=========== training steps ==========")
    print(len(train_ds[0])*training_config['epoch']/training_config['batch_size'])

    def model_fn(features, labels, mode, params):
        return model_fn_base(features, labels, mode, params, net_config, training_config)

    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    train_input_fn, train_iterator_initializer_hook = \
        get_training_inputs_fn(train_ds, training_config['epoch'],
                               10000, training_config['batch_size'])
    eval_input_fn, eval_iterator_initializer_hook = \
        get_evaluation_inputs_fn(eval_ds, training_config['batch_size'])

    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=training_config['model_dir'])

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        hooks=[train_iterator_initializer_hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      hooks=[eval_iterator_initializer_hook])
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


def single_view_predict(net_config, running_config, dataset, subjects, output_type='label'):

    subjects_num = len(subjects)
    outputs = []
    if subjects_num == 0:
        return outputs

    print("=========== subjects to be predicted ==========")
    print(subjects)

    inputs_np = dataset.generate_ds(subjects, False)
    shape = inputs_np.shape
    print(shape)

    def model_fn(features, labels, mode, params):
        return model_fn_base(features, labels, mode, params, net_config, running_config)

    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    pred_input_fn, pred_iterator_initializer_hook = \
        get_prediction_inputs_fn(inputs_np, running_config['batch_size'])

    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                        model_dir=running_config['model_dir'])

    if output_type == 'probs':
        outputs_np = classifier.predict(pred_input_fn,
                                        predict_keys=['probs'],
                                        hooks=[pred_iterator_initializer_hook])
    else:
        outputs_np = classifier.predict(pred_input_fn,
                                        predict_keys=['labels'],
                                        hooks=[pred_iterator_initializer_hook])

    thinkness = shape[0] / subjects_num

    tmp = []
    for index, layer in enumerate(outputs_np, 1):
        if outputs == 'probs':
            layer = layer['probs']
        else:
            layer = layer['labels']
        tmp.append(layer)
        if index % thinkness == 0:
            outputs.append(np.array(tmp))
            tmp = []

    return outputs

def softmax(data, theta=1, axis=-1):
    """
       Compute the softmax of each element along an axis of X.

       Parameters
       ----------
       X: ND-Array. Probably should be floats.
       theta (optional): float parameter, used as a multiplier
           prior to exponentiation. Default = 1.0
       axis (optional): axis to compute values along. Default is the
           first non-singleton axis.

       Returns an array the same size as X. The result will sum to 1
       along the specified axis.
       """

    # multiply y against the theta parameter,
    data = data * theta
    # subtract the max for numerical stability
    data = data - np.expand_dims(np.max(data, axis=axis), axis)
    # exponentiate y
    data = np.exp(data)
    # take the sum along the specified axis
    sum = np.expand_dims(np.sum(data, axis=axis), axis)
    # finally: divide elementwise
    p = data / sum

    return p


def resize_image(imgs, size, mode='NEAREST'):

    shape = imgs.shape
    labels = tf.placeholder(tf.float32, shape=(shape[0], shape[1], shape[2], shape[3]))
    if mode == 'NEAREST':
        labels_resized = tf.image.resize_images(labels, (size[0], size[1]),
                                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    else:
        labels_resized = tf.image.resize_images(labels, (size[0], size[1]),
                                                method=tf.image.ResizeMethod.BILINEAR, align_corners=True)

    with tf.Session() as sess:
        scaled = sess.run(labels_resized, feed_dict={labels: imgs})

    return scaled


class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        # Initialize the iterator with the data feed_dict
        self.iterator_initializer_func(session)


def get_training_inputs_fn(train_ds, epoch, shuffle_buffer, batch_size):

    iterator_initializer_hook = IteratorInitializerHook()

    def input_fn():
        X_pl = tf.placeholder(train_ds[0].dtype, train_ds[0].shape)
        y_pl = tf.placeholder(train_ds[1].dtype, train_ds[1].shape)

        dataset = tf.data.Dataset.from_tensor_slices((X_pl, y_pl))
        dataset = dataset.repeat(epoch)
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        next_example, next_label = iterator.get_next()

        fn = lambda sess: sess.run(iterator.initializer, feed_dict={X_pl: train_ds[0],
                                                                    y_pl: train_ds[1]})
        iterator_initializer_hook.iterator_initializer_func = fn

        return next_example, next_label

    return input_fn, iterator_initializer_hook


def get_evaluation_inputs_fn(eval_ds, batch_size):

    iterator_initializer_hook = IteratorInitializerHook()

    def input_fn():
        X_pl = tf.placeholder(eval_ds[0].dtype, eval_ds[0].shape)
        y_pl = tf.placeholder(eval_ds[1].dtype, eval_ds[1].shape)

        dataset = tf.data.Dataset.from_tensor_slices((X_pl, y_pl))
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        next_example, next_label = iterator.get_next()

        fn = lambda sess: sess.run(iterator.initializer, feed_dict={X_pl: eval_ds[0],
                                                                    y_pl: eval_ds[1]})
        iterator_initializer_hook.iterator_initializer_func = fn

        return next_example, next_label

    return input_fn, iterator_initializer_hook


def get_prediction_inputs_fn(pred_ds, batch_size):

    iterator_initializer_hook = IteratorInitializerHook()

    def input_fn():
        X = tf.placeholder(pred_ds.dtype, pred_ds.shape)

        dataset = tf.data.Dataset.from_tensor_slices(X)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        next_example = iterator.get_next()

        fn = lambda sess: sess.run(iterator.initializer, feed_dict={X: pred_ds})
        iterator_initializer_hook.iterator_initializer_func = fn

        return next_example

    return input_fn, iterator_initializer_hook

