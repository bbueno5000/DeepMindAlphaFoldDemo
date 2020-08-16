"""
DOCSTRING
"""
import argparse
import numpy
import pandas
import random
import tensorflow
import tqdm

PADDING_FULL_LEN = 500
SAMPLES_EACH_CHUNK = 512

def _bytes_feature(value):
    """
    DOCSTRING
    """
    return tensorflow.train.Feature(
        bytes_list=tensorflow.train.BytesList(value=[value]))

def _float_list_feature(value):
    """
    DOCSTRING
    """
    return tensorflow.train.Feature(
        float_list=tensorflow.train.FloatList(
            value=numpy.reshape(value,-1)))

def _int64_feature(value):
    """
    DOCSTRING
    """
    return tensorflow.train.Feature(
        int64_list=tensorflow.train.Int64List(value=[value]))

def _int64_list_feature(value):
    """
    DOCSTRING
    """
    return tensorflow.train.Feature(
        int64_list=tensorflow.train.Int64List(
            value=numpy.reshape(value, -1)))

def create_semi(name_list, oned_feature_dir,twod_feature_dir, output_prefix, labeled=True):
    """
    DOCSTRING
    """
    padding_full_num = PADDING_FULL_LEN * PADDING_FULL_LEN
    names = pandas.read_csv(name_list, names=['name'], header=None)
    names = list(names['name'])
    random.shuffle(names)
    SAMPLES_EACH_CHUNK = 1024
    chunk_num = (len(names) - 1) / SAMPLES_EACH_CHUNK + 1
    for i in range(chunk_num):
        with tensorflow.python_io.TFRecordWriter(
            '%s_%s.tfrecord' % (output_prefix, i)) as record_writer:
            start = i * SAMPLES_EACH_CHUNK
            end = min(len(names), (i + 1) * SAMPLES_EACH_CHUNK)
            X1d, X2d, Y = list(), list(), list()
            for name in names[start:end]:
                fea1 = pandas.read_csv(
                    '{}/{}.1dfeat'.format(oned_feature_dir, name),
                    header=None, sep='\s+')
                L = fea1.shape[0]
                if L > PADDING_FULL_LEN:
                    continue
                x1d = fea1.iloc[:, range(23, 43) + range(63, 69)].values
                fea2 = pandas.read_csv(
                    '{}/{}.2dfeat'.format(twod_feature_dir, name),
                    header=None, sep='\s+')
                assert L == int(numpy.sqrt(fea2.shape[0]))
                data = fea2.iloc[:,[2,4,5,7]].values
                data = data.reshape((L, L, -1))
                x2d = data[:L,:L,1:]
                feature={
                    'size': _int64_feature(L),
                    'x1d': _bytes_feature(x1d.astype(numpy.float32).tobytes()),
                    'x2d': _bytes_feature(x2d.astype(numpy.float32).tobytes()),}
                if labeled:
                    y = data[:L, :L, 0].reshape((L, L))
                    y = numpy.where(
                        y > 0.0, (y < 8.0).astype(numpy.int16), y).astype(numpy.int16)
                    y[numpy.tril_indices(y.shape[0], 5)] = -1
                    y_ = numpy.zeros((L, L, 2))
                    for i in range(L):
                        for j in range(L):
                            if y[i,j] < 0:
                                y_[i,j,:] = -1
                            else:
                                y_[i,j,y[i,j]] = 1
                    feature.update({'y':_bytes_feature(y_.astype(numpy.int16).tobytes())})
                example = tensorflow.train.Example(
                    features = tensorflow.train.Features(feature=feature))
                record_writer.write(example.SerializeToString())

def create_supervised(name_list, oned_feature_dir, twod_feature_dir, output_prefix):
    """
    DOCSTRING
    """
    padding_full_num = PADDING_FULL_LEN * PADDING_FULL_LEN
    names = pandas.read_csv(name_list, names=['name'], header=None)
    names = list(names['name'])
    random.shuffle(names)
    SAMPLES_EACH_CHUNK = 1024
    chunk_num = (len(names) - 1) / SAMPLES_EACH_CHUNK + 1
    total_example_num = 0
    total_pos_num = 0
    total_neg_num = 0
    for i in range(chunk_num):
        with tensorflow.python_io.TFRecordWriter(
            '%s_%s.tfrecord' % (output_prefix, i)) as record_writer:
            start = i * SAMPLES_EACH_CHUNK
            end = min(len(names), (i+1) * SAMPLES_EACH_CHUNK)
            X1d, X2d, Y = list(), list(), list()
            for name in names[start:end]:
                fea1 = pandas.read_csv(
                    '{}/{}.1dfeat'.format(oned_feature_dir, name),
                    header=None, sep='\s+')
                L = fea1.shape[0]
                if L > PADDING_FULL_LEN:
                    continue
                x1d = fea1.iloc[:,range(23, 43) + range(63, 69)].values
                fea2 = pandas.read_csv(
                    '{}/{}.2dfeat'.format(twod_feature_dir, name),
                    header=None, sep='\s+')
                assert L == int(numpy.sqrt(fea2.shape[0]))
                data = fea2.iloc[:, [2, 4, 5, 7]].values
                data = data.reshape((L, L, -1))
                x2d = data[:L, :L, 1:]
                y = data[:L, :L, 0].reshape((L, L))
                y = numpy.where(
                    y > 0.0, (y < 8.0).astype(numpy.int16), y).astype(numpy.int16)
                y[numpy.tril_indices(y.shape[0], 5)] = -1
                neg_num = numpy.sum(y == 0)
                pos_num = numpy.sum(y == 1)
                if pos_num < 100:
                    continue
                total_pos_num += pos_num
                total_neg_num += neg_num
                total_example_num += 1
                example = tensorflow.train.Example(features = tensorflow.train.Features(
                    feature={
                        'size': _int64_feature(L),
                        'x1d': _bytes_feature(x1d.astype(numpy.float32).tobytes()),
                        'x2d': _bytes_feature(x2d.astype(numpy.float32).tobytes()),
                        'y': _bytes_feature(y.astype(numpy.int16).tobytes())}))
                record_writer.write(example.SerializeToString())
                print(name)
    print(total_example_num, total_pos_num + total_neg_num, total_pos_num, total_neg_num)

def create_from_ccmpred(
    name_list,
    profile_dir,
    structure_dir,
    ccmpred_dir,
    distcb_dir,
    output_prefix,
    chunk_size):
    """
    DOCSTRING
    """
    PADDING_FULL_LEN = 250
    padding_full_num = PADDING_FULL_LEN * PADDING_FULL_LEN
    names = pandas.read_csv(name_list, names=['name'], header=None)
    names = list(names['name'])
    random.shuffle(names)
    SAMPLES_EACH_CHUNK = chunk_size
    chunk_num = (len(names) - 1) / SAMPLES_EACH_CHUNK + 1
    total_example_num, total_pos_num, total_neg_num = 0, 0, 0
    pbar = tqdm.tqdm(total=len(names))
    for i in range(chunk_num):
        with tensorflow.python_io.TFRecordWriter(
            '%s_%s.tfrecord' % (output_prefix, i)) as record_writer: 
            start = i * SAMPLES_EACH_CHUNK
            end = min(len(names), (i+1) * SAMPLES_EACH_CHUNK)
            if i + 1 == chunk_num:
                end = len(names)
            X1d, X2d, Y = list(), list(), list()
            for name in names[start:end]:
                pbar.update(1)
                profile = numpy.loadtxt('{}/{}.profile'.format(profile_dir, name))
                L = profile.shape[0]
                if L > PADDING_FULL_LEN:
                    continue
                structure = pandas.read_csv('{}/{}.structure'.format(structure_dir, name))
                structure = structure.iloc[:, range(3, 6) + range(16, 19) + range(20, 21)].values
                ccmpred = numpy.loadtxt('{}/{}.cc'.format(ccmpred_dir, name))
                assert L == structure.shape[0]
                assert L == ccmpred.shape[0]
                x1d = numpy.concatenate([profile, structure], axis=1)
                x2d = ccmpred[:, :, numpy.newaxis]
                if distcb_dir is not None:
                    y = numpy.loadtxt('{}/{}.distcb'.format(distcb_dir, name))
                    assert L == y.shape[0]
                    y = numpy.where(
                        y > 0.0, (y < 8.0).astype(numpy.int16), y).astype(numpy.int16)
                    y[numpy.tril_indices(y.shape[0], 4)] = -1
                    neg_num = numpy.sum(y==0)
                    pos_num = numpy.sum(y==1)
                    if pos_num < 100:
                        continue
                    total_pos_num += pos_num
                    total_neg_num += neg_num
                    example = tensorflow.train.Example(features=tensorflow.train.Features(
                        feature={
                            'size': _int64_feature(L),
                            'x1d': _bytes_feature(x1d.astype(numpy.float32).tobytes()),
                            'x2d': _bytes_feature(x2d.astype(numpy.float32).tobytes()),
                            'y': _bytes_feature(y.astype(numpy.int16).tobytes()),
                            'name': _bytes_feature(name) }))
                else:
                    example = tensorflow.train.Example(features=tensorflow.train.Features(
                        feature={
                            'size': _int64_feature(L),
                            'x1d': _bytes_feature(x1d.astype(numpy.float32).tobytes()),
                            'x2d': _bytes_feature(x2d.astype(numpy.float32).tobytes()),
                            'name': _bytes_feature(name) }))
                record_writer.write(example.SerializeToString())
                total_example_num += 1
    pbar.close()
    if distcb_dir is not None:
        print(total_example_num, total_pos_num + total_neg_num, total_pos_num, total_neg_num)
    else:
        print(total_example_num)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--op',
        choices=['create_from_ccmpred', 'create_semi', 'create_supervised'],
        required=True)
    parser.add_argument('--name_list', type=str)
    parser.add_argument('--oned_feature_dir', type=str)
    parser.add_argument('--twod_feature_dir', type=str)
    parser.add_argument('--output_prefix', type=str)
    parser.add_argument('--labeled', action='store_true')
    parser.add_argument('--profile_dir', type=str)
    parser.add_argument('--structure_dir', type=str)
    parser.add_argument('--ccmpred_dir', type=str)
    parser.add_argument('--distcb_dir', type=str)
    parser.add_argument('--chunk_size', type=int, default=512)
    args = parser.parse_args()
    if args.op == 'create_semi':
        create_semi(
            args.name_list,
            args.oned_feature_dir,
            args.twod_feature_dir,
            args.output_prefix,
            args.labeled)
    if args.op == 'create_supervised':
        create_supervised(
            args.name_list,
            args.oned_feature_dir,
            args.twod_feature_dir,
            args.output_prefix)
    if args.op == 'create_from_ccmpred':
        create_from_ccmpred(
            args.name_list,
            args.profile_dir,
            args.structure_dir,
            args.ccmpred_dir,
            args.distcb_dir,
            args.output_prefix,
            args.chunk_size)

if __name__ == '__main__':
    main()
