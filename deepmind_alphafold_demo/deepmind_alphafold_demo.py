"""
DOCSTRING
"""
import argparse
import enum
import json
import logging
import math
import os
import random
import sys
import tensorflow


class Device:
    """
    DOCSTRING
    """
    def __call__(self):
        print(self.get_available_gpus())
        print(self.get_available_cpus())

    def get_available_cpus(self):
        """
        DOCSTRING
        """
        local_device_protos = tensorflow.python.client.device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'CPU']

    def get_available_gpus(self):
        """
        DOCSTRING
        """
        local_device_protos = tensorflow.python.client.device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

class MemDataset:
    """
    DOCSTRING
    """
    pass

class Ops:
    """
    DOCSTRING
    """
    def deconv2d(
        input_,
        output_shape,
        k_h=5,
        k_w=5,
        d_h=2,
        d_w=2,
        stddev=0.02,
        name='deconv2d',
        with_w=False):
        """
        DOCSTRING
        """
        with tensorflow.variable_scope(name):
            #filter: [height, width, output_channels, in_channels]
            w = tensorflow.get_variable(
                'w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                initializer=tensorflow.random_normal_initializer(stddev=stddev))
            deconv = tensorflow.nn.conv2d_transpose(
                input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])
            biases = tensorflow.get_variable(
                'biases', [output_shape[-1]],
                initializer=tensorflow.constant_initializer(0.0))
            deconv = tensorflow.reshape(
                tensorflow.nn.bias_add(deconv, biases), deconv.get_shape())
            if with_w:
              return deconv, w, biases

            else:
              return deconv

    def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
        """
        DOCSTRING
        """
        shape = input_.get_shape().as_list()
        with tensorflow.variable_scope(scope or "Linear"):
            matrix = tensorflow.get_variable(
                "Matrix", [shape[1], output_size], tensorflow.float32,
                tensorflow.random_normal_initializer(stddev=stddev))
            bias = tensorflow.get_variable(
                "bias", [output_size],
                initializer=tensorflow.constant_initializer(bias_start))
            if with_w:
              return tensorflow.matmul(input_, matrix) + bias, matrix, bias
            else:
              return tensorflow.matmul(input_, matrix) + bias

class ResNet:
    """
    DOCSTRING
    """
    def __init__(self, sess, dataset, train_config, model_config):
        self.sess = sess
        self.dataset = dataset
        self.train_config = train_config
        self.model_config = model_config
        self.input_tfrecord_files = tensorflow.placeholder(
            tensorflow.string, shape=[None])
        self.keep_prob = tensorflow.placeholder(tensorflow.float32)
        self.training = tensorflow.placeholder(tensorflow.bool)
        self.x1d_channel_dim = model_config['1d']['channel_dim']
        self.x2d_channel_dim = model_config['2d']['channel_dim']
        self.padding_full_len = 250

    def build_input(self):
        """
        DOCSTRING
        """
        with tensorflow.device('/cpu:0'):
            def parser(record):
                keys_to_features = {
                    'x1d': tensorflow.FixedLenFeature([], tensorflow.string),
                    'x2d': tensorflow.FixedLenFeature([], tensorflow.string),
                    'y': tensorflow.FixedLenFeature([], tensorflow.string),
                    'size': tensorflow.FixedLenFeature([], tensorflow.int64)}
                parsed = tensorflow.parse_single_example(record, keys_to_features)
                x1d = tensorflow.decode_raw(parsed['x1d'], tensorflow.float32)
                x2d = tensorflow.decode_raw(parsed['x2d'] ,tensorflow.float32)
                size = parsed['size']
                x1d = tensorflow.reshape(x1d, tensorflow.stack([size, -1]))
                x2d = tensorflow.reshape(x2d, tensorflow.stack([size, size, -1]))
                y = tensorflow.decode_raw(parsed['y'],tensorflow.int16)
                y = tensorflow.cast(y, tensorflow.float32)
                y = tensorflow.reshape(y, tensorflow.stack([size, size]))
                return x1d, x2d, y, size
            dataset = tensorflow.data.TFRecordDataset(self.input_tfrecord_files)
            dataset = dataset.map(parser, num_parallel_calls=64)
            dataset = dataset.prefetch(1024)
            dataset = dataset.shuffle(buffer_size=512)
            dataset = dataset.padded_batch(
                self.train_config.batch_size,
                padded_shapes=(
                    [self.padding_full_len, self.x1d_channel_dim],
                    [self.padding_full_len, self.padding_full_len, self.x2d_channel_dim],
                    [self.padding_full_len, self.padding_full_len], []),
                padding_values=(0.0, 0.0, -1.0, numpy.int64(self.padding_full_len)))
            iterator = dataset.make_initializable_iterator()
            x1d, x2d, y, size = iterator.get_next()
            return  x1d, x2d, y, size, iterator

    def build_input_test(self):
        """
        DOCSTRING
        """
        with tensorflow.device('/cpu:0'):
            def parser(record):
                keys_to_features = {
                    'x1d': tensorflow.FixedLenFeature([], tensorflow.string),
                    'x2d': tensorflow.FixedLenFeature([], tensorflow.string),
                    'name': tensorflow.FixedLenFeature([], tensorflow.string),
                    'size': tensorflow.FixedLenFeature([], tensorflow.int64)}
                parsed = tensorflow.parse_single_example(record, keys_to_features)
                x1d = tensorflow.decode_raw(parsed['x1d'], tensorflow.float32)
                x2d = tensorflow.decode_raw(parsed['x2d'] ,tensorflow.float32)
                size = parsed['size']
                x1d = tensorflow.reshape(x1d, tensorflow.stack([size, -1]))
                x2d = tensorflow.reshape(x2d, tensorflow.stack([size, size, -1]))
                name = parsed['name']
                return x1d, x2d, name, size
            dataset = tensorflow.data.TFRecordDataset(self.input_tfrecord_files)
            dataset = dataset.map(parser, num_parallel_calls=64)
            dataset = dataset.prefetch(512)
            dataset = dataset.padded_batch(
                self.train_config.batch_size,
                padded_shapes=(
                    [self.padding_full_len, self.x1d_channel_dim],
                    [self.padding_full_len, self.padding_full_len, self.x2d_channel_dim],
                    [], []),
                    padding_values=(0.0, 0.0, "", numpy.int64(self.padding_full_len)))
            iterator = dataset.make_initializable_iterator()
            x1d, x2d, name, size = iterator.get_next()
            return  x1d, x2d, name, size, iterator

    def cnn_with_2dfeature(self, x2d, reuse=False):
        """
        DOCSTRING
        """
        with tensorflow.variable_scope('discriminator', reuse=reuse) as scope:
            block_num, filters = 8, 16
            kernel_size = [4, 4]
            act = tensorflow.nn.relu
            kernel_initializer = tensorflow.glorot_normal_initializer()
            bias_initializer = tensorflow.zeros_initializer()
            kernel_regularizer = None
            bias_regularizer = None
            for i in numpy.arange(block_num):
                inputs = x2d if i == 0 else conv_
                conv_ = tensorflow.layers.conv2d(
                    inputs=inputs, filters=filters,
                    kernel_size=kernel_size, strides=(1,1), padding='same', activation=act,
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
            logits = tensorflow.layers.conv2d(
                inputs=conv_, filters=1,
                kernel_size=kernel_size, strides=(1,1), padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
            logits = tensorflow.reshape(
                logits, (-1, tensorflow.shape(logits)[1], tensorflow.shape(logits)[2]))
            return tensorflow.sigmoid(logits), logits

    def evaluate(self, mode):
        """
        DOCSTRING
        """
        self.sess.run(
            self.iterator.initializer,
            feed_dict={self.input_tfrecord_files:self.dataset.get_chunks(mode)})
        acc = list()
        while True:
            try:
                pred, y, size = self.sess.run([self.pred, self.y, self.size])
                for y_, pred_, size_ in zip(y, pred, size):
                    acc_ = Utils.top_accuracy(pred_[:size_, :size_], y_[:size_, :size_])
                    acc.append(acc_)
            except tensorflow.errors.OutOfRangeError:
                break
        acc = numpy.array(acc)
        acc = numpy.mean(acc, axis=0)
        acc_str = ' '.join(['%.4f ' % acc_ for acc_ in acc])
        logging.info('{:s} acc: {:s}'.format(mode, acc_str))

    def predict(self, output_dir, model_path):
        """
        DOCSTRING
        """
        x1d, x2d, name, size, iterator = self.build_input_test()
        preds, logits = self.resn(x1d, x2d)
        saver = tensorflow.train.Saver()
        saver.restore(self.sess, model_path)
        self.sess.run(
            iterator.initializer,
            feed_dict={self.input_tfrecord_files: self.dataset.get_chunks(RunMode.TEST)})
        while True:
            try:
                preds_, names_, sizes_, = self.sess.run([preds, name, size])
                for pred_, name_, size_ in zip(preds_, names_, sizes_):
                    pred_ = pred_[:size_, :size_]
                    output_path = '{}/{}.concat'.format(output_dir, name_)
                    numpy.savetxt(output_path, pred_)
            except tensorflow.errors.OutOfRangeError:
                break

    def resn(self, x1d, x2d, reuse=False):
        """
        DOCSTRING
        """
        with tensorflow.variable_scope('discriminator', reuse=reuse) as scope:
            act = tensorflow.nn.relu
            filters_1d = self.model_config['1d']['filters']
            kernel_size_1d = self.model_config['1d']['kernel_size']
            block_num_1d = self.model_config['1d']['block_num']
            filters_2d = self.model_config['2d']['filters']
            kernel_size_2d = self.model_config['2d']['kernel_size']
            block_num_2d = self.model_config['2d']['block_num']
            kernel_initializer = tensorflow.variance_scaling_initializer()
            bias_initializer = tensorflow.zeros_initializer()
            if self.train_config.l2_reg <= 0.0:
                kernel_regularizer = None
            else:
                kernel_regularizer = tensorflow.contrib.layers.l2_regularizer(
                    scale=self.train_config.l2_reg)
            bias_regularizer = None
            prev_1d = tensorflow.layers.conv1d(
                inputs=x1d, filters=filters_1d,
                kernel_size=kernel_size_1d, strides=1, padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                use_bias=False)
            for i in numpy.arange(block_num_1d):
                conv_1d = act(prev_1d)
                conv_1d = tensorflow.layers.conv1d(
                    inputs=conv_1d, filters=filters_1d,
                    kernel_size=kernel_size_1d, strides=1, padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                    use_bias=False)
                conv_1d = act(conv_1d)
                conv_1d = tensorflow.layers.conv1d(
                    inputs=conv_1d, filters=filters_1d,
                    kernel_size=kernel_size_1d, strides=1, padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                    use_bias=False)
                prev_1d = tensorflow.add(conv_1d, prev_1d)
            out_1d = tensorflow.expand_dims(prev_1d, axis=3)
            ones = tensorflow.ones((1, self.padding_full_len))
            left_1d = tensorflow.einsum('abcd,de->abce', out_1d, ones)
            left_1d = tensorflow.transpose(left_1d, perm=[0,1,3,2])
            right_1d = tensorflow.transpose(left_1d, perm=[0,2,1,3])
            print('1d shape', left_1d.shape, right_1d.shape)
            input_2d = tensorflow.concat([x2d, left_1d, right_1d], axis=3)
            print('2d shape', input_2d.shape)
            prev_2d = tensorflow.layers.conv2d(
                inputs=input_2d, filters=filters_2d,
                kernel_size=kernel_size_2d, strides=(1,1), padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                use_bias=False)
            for i in numpy.arange(block_num_2d):
                conv_2d = act(prev_2d)
                conv_2d = tensorflow.layers.conv2d(
                    inputs=conv_2d, filters=filters_2d,
                    kernel_size=kernel_size_2d, strides=(1,1), padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                    use_bias=False)
                conv_2d = act(conv_2d)
                conv_2d = tensorflow.layers.conv2d(
                    inputs=conv_2d, filters=filters_2d,
                    kernel_size=kernel_size_2d, strides=(1,1), padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                    use_bias=False)
                prev_2d =  tensorflow.add(conv_2d, prev_2d)
            logits = tensorflow.layers.conv2d(
                inputs=prev_2d, filters=1,
                kernel_size=kernel_size_2d, strides=(1,1), padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                use_bias=True)
            logits = tensorflow.squeeze(logits, 3)
            logits_tran = tensorflow.transpose(logits, perm=[0, 2, 1])
            logits = (logits + logits_tran) / 2.0
            return tensorflow.sigmoid(logits), logits
    
    def resn_with_2dfeature(self, x2d, reuse=False):
        """
        DOCSTRING
        """
        with tensorflow.variable_scope('discriminator', reuse=reuse) as scope:
            block_num, filters = 8, 32
            kernel_size = [4, 4]
            act = tensorflow.nn.relu
            kernel_initializer = tensorflow.glorot_normal_initializer()
            bias_initializer = tensorflow.zeros_initializer()
            kernel_regularizer = None
            bias_regularizer = None
            prev = tensorflow.layers.conv2d(
                inputs=x2d, filters=filters,
                kernel_size=kernel_size, strides=(1,1), padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
            for i in numpy.arange(block_num):
                conv_ = act(prev)
                conv_ = tensorflow.layers.conv2d(
                    inputs=conv_, filters=filters,
                    kernel_size=kernel_size, strides=(1,1), padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
                conv_ = act(conv_)
                conv_ = tensorflow.layers.conv2d(
                    inputs=conv_, filters=filters,
                    kernel_size=kernel_size, strides=(1,1), padding='same',
                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
                prev = tensorflow.add(conv_, prev)
            logits = tensorflow.layers.conv2d(
                inputs=prev, filters=1,
                kernel_size=kernel_size, strides=(1,1), padding='same',
                kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer)
            logits = tensorflow.reshape(
                logits, (-1, tensorflow.shape(logits)[1], tensorflow.shape(logits)[2]))
            return tensorflow.sigmoid(logits), logits

    def train(self):
        """
        DOCSTRING
        """
        self.x1d, self.x2d, self.y, self.size, self.iterator = self.build_input()
        with tensorflow.device('/gpu:0'):
            self.pred, logits = self.resn(self.x1d, self.x2d)
            if self.train_config.down_weight >= 1.0:
                mask = tensorflow.greater_equal(self.y, 0.0)
                labels = tensorflow.boolean_mask(self.y, mask)
                logits = tensorflow.boolean_mask(logits, mask)
                self.loss = tensorflow.reduce_mean(
                    tensorflow.nn.sigmoid_cross_entropy_with_logits(
                        labels=labels, logits=logits))
            else:
                mask_pos = tensorflow.equal(self.y, 1.0)
                label_pos = tensorflow.boolean_mask(self.y, mask_pos)
                logit_pos = tensorflow.boolean_mask(logits, mask_pos)
                loss_pos = tensorflow.reduce_mean(
                    tensorflow.nn.sigmoid_cross_entropy_with_logits(
                        labels=label_pos, logits=logit_pos))
                mask_neg = tensorflow.equal(self.y, 0.0)
                label_neg = tensorflow.boolean_mask(self.y, mask_neg)
                logit_neg = tensorflow.boolean_mask(logits, mask_neg)
                loss_neg = tensorflow.reduce_mean(
                    tensorflow.nn.sigmoid_cross_entropy_with_logits(
                        labels=label_neg, logits=logit_neg))
                self.loss = loss_neg * self.train_config.down_weight + loss_pos
            update_ops = tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS)
            with tensorflow.control_dependencies(update_ops):
                if self.train_config.op_alg == 'adam':
                    optim = tensorflow.train.AdamOptimizer(
                        self.train_config.learn_rate,
                        beta1=self.train_config.beta1).minimize(self.loss)
                elif self.train_config.op_alg == 'sgd':
                    optim = tensorflow.train.GradientDescentOptimizer(
                            self.train_config.learn_rate).minimize(self.loss)
        tensorflow.summary.scalar('train_loss', self.loss)
        merged_summary = tensorflow.summary.merge_all()
        train_writer = tensorflow.summary.FileWriter(
            self.train_config.summary_dir, self.sess.graph)
        tensorflow.global_variables_initializer().run()
        steps, saver = 0, tensorflow.train.Saver()
        for epoch in numpy.arange(self.train_config.epoch):
            self.sess.run(
                self.iterator.initializer,
                feed_dict={self.input_tfrecord_files:self.dataset.get_chunks(RunMode.TRAIN)})
            train_loss = 0.0
            while True:
                try:
                    _, _loss, summary = self.sess.run([optim, self.loss, merged_summary])
                    train_loss += _loss
                    train_writer.add_summary(summary, steps)
                    steps += 1
                except tensorflow.errors.OutOfRangeError:
                    break
            saver.save(
                self.sess, '{}/model'.format(self.train_config.model_dir), global_step=epoch)
            logging.info('Epoch= {:d} train_loss= {:.4f}'.format(epoch, train_loss))
            self.evaluate(RunMode.VALIDATE)
            if self.train_config.test_file_prefix is not None:
                self.evaluate(RunMode.TEST)
        train_writer.close()

class RunMode(enum.Enum):
    """
    DOCSTRING
    """
    TRAIN = 1
    VALIDATE = 2
    TEST = 3
    UNLABEL = 4

class TFRecordDataset:
    """
    DOCSTRING
    """
    def __init__(
        self,
        train_file_prefix='',
        chunk_num=0,
        val_size = 1,
        test_file_prefix='',
        unlabel_file_prefix='',
        unlabel_chunk_num=0):
        self.chunk_num = chunk_num
        self.train_file_prefix = train_file_prefix
        self.test_file_prefix = test_file_prefix
        self.unlabel_file_prefix = unlabel_file_prefix
        self.unlabel_chunk_num = unlabel_chunk_num
        idx = range(chunk_num)
        if type(val_size) == float:
            train_chunk_num = numpy.ceil(chunk_num * (1.0 - val_size))
        else:
            train_chunk_num = chunk_num - val_size
        self.train_chunks = idx[:train_chunk_num]
        self.val_chunks = idx[train_chunk_num:]

    def get_chunks(self, mode):
        """
        DOCSTRING
        """
        input_file_prefix = self.train_file_prefix
        if mode == RunMode.TRAIN:
            chunks = self.train_chunks
            random.shuffle(chunks)
        elif mode == RunMode.VALIDATE:
            chunks = self.val_chunks
        elif mode == RunMode.TEST:
            input_file_prefix = self.test_file_prefix
            chunks = [0]
        elif model == RunMode.UNLABEL:
            input_file_prefix = self.unlabel_file_prefix
            chunks = range(self.unlabel_chunk_num)
            random.shuffle(chunks)
        return ['{}_{:d}.tfrecord'.format(input_file_prefix, c) for c in chunks]

class Utils:
    """
    DOCSTRING
    """
    def conv_out_size_same(self, size, stride):
        """
        DOCSTRING
        """
        return int(math.ceil(float(size) / float(stride)))

    def top_accuracy(self, pred=None, truth=None, ratio=[1, 0.5, 0.2, 0.1]):
        """
        DOCSTRING
        """
        if pred is None:
            print('please provide a predicted contact matrix')
            sys.exit(-1)
        if truth is None:
            print('please provide a true contact matrix')
            sys.exit(-1)
        assert pred.shape[0] == pred.shape[1]
        assert pred.shape == truth.shape
        pred_truth = numpy.dstack((pred, truth))
        M1s = numpy.ones_like(truth, dtype=numpy.int8)
        mask_LR = numpy.triu(M1s, 24)
        mask_MLR = numpy.triu(M1s, 12)
        mask_SMLR = numpy.triu(M1s, 6)
        mask_MR = mask_MLR - mask_LR
        mask_SR = mask_SMLR - mask_MLR
        seqLen = pred.shape[0]
        accs = list()
        for mask in [mask_LR, mask_MR, mask_MLR, mask_SR]:
            res = pred_truth[mask.nonzero()]
            res_sorted = res[(-res[:, 0]).argsort()]
            for r in ratio:
                numTops = int(seqLen * r)
                numTops = min(numTops, res_sorted.shape[0])
                topLabels = res_sorted[:numTops, 1]
                numCorrects = numpy.sum(topLabels == 1.0)
                accuracy = numCorrects * 1./numTops
                accs.append(accuracy)
        return numpy.array(accs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, choices=['gan', 'resn'], default='resn')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learn_rate', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--l2_reg', type=float, default=0.0001)
    parser.add_argument('--down_weight', type=float, default=1.0)
    parser.add_argument('--keep_prob', type=float, default=0.8)
    parser.add_argument('--op_alg', type=str, choices=['sgd', 'adam'], default='adam')
    parser.add_argument('--train_file_prefix', type=str)
    parser.add_argument('--unlabel_file_prefix', type=str)
    parser.add_argument('--test_file_prefix', type=str, default=None)
    parser.add_argument('--chunk_num', type=int, default=4)
    parser.add_argument('--unlabel_chunk_num', type=int)
    parser.add_argument('--gan_gen_iter', type=int, default=5)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--job_dir', type=str)
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--summary_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--mode', type=str, choices=['test', 'train'], default='train')
    args = parser.parse_args()
    if args.job_dir is not None:
        os.makedirs(args.job_dir)
        if args.summary_dir is None:
            args.summary_dir = '{}/summary'.format(args.job_dir)
            os.makedirs(args.summary_dir)
        if args.model_dir is None:
            args.model_dir = '{}/model'.format(args.job_dir)
            os.makedirs(args.model_dir)
        if args.log_file is None:
            args.log_file = '{}/run.log'.format(args.job_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log_file_stream=tensorflow.python.lib.io.file_io.FileIO(args.log_file,'a')
    fh = logging.StreamHandler(log_file_stream)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    model_config = parse_model_config(args.model_config)
    logging.info('train_config: {:s}'.format(args))
    logging.info('model_config: {:s}'.format(json.dumps(model_config)))
    if args.alg == 'resn':
        with tensorflow.Session() as sess:
            if args.mode == 'train':
                dataset = TFRecordDataset(
                        args.train_file_prefix, args.chunk_num, val_size = 1,
                        test_file_prefix = args.test_file_prefix)
                resn_ = ResNet(sess, dataset, train_config=args, model_config=model_config)
                resn_.train()
            elif args.mode == 'test':
                dataset = TFRecordDataset(test_file_prefix = args.test_file_prefix)
                resn_ = ResNet(sess, dataset, train_config=args, model_config=model_config)
                resn_.predict(args.output_dir, args.model_path)

def parse_model_config(json_file):
    """
    DOCSTRING
    """
    with tensorflow.python.lib.io.file_io.FileIO(json_file, 'r') as f:
        config = json.load(f)
    return config

if __name__ == '__main__':
    main()
