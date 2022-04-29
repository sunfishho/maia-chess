import argparse
import os
import os.path
import yaml
import sys
import glob
import gzip
import random
import time
import multiprocessing

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}

import maia_chess_backend
import maia_chess_backend.maia

import wandb
import logging
import pdb
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)


import numpy as np
from maia_chess_backend.maia.policy_index import policy_index
from maia_chess_backend.maia.lc0_az_policy_map import *

SKIP = 64

# #New function for mapping policy index to squares
def policy_index_to_squares (index):
    move = policy_index[index]
    start = np.zeros(64)
    end = np.zeros(64)
    promotion = np.zeros(64) #TODO

    start_square = position_to_index(move[0:2])  #Format: a1 -> (0, 0), h8 -> (7, 7), (col, row)
    end_square = position_to_index(move[2:])
    start_col = start_square[0]
    start_row = start_square[1]
    end_col = end_square[0]
    end_row = end_square[1]

    start[start_col + 8*start_row] = 1
    end[end_col + 8*end_row] = 1

    return (start, end, promotion)

def gen_discriminator_data(x, y):
    final_x = np.zeros((1024, 115, 64))
    y = y.numpy()
    # line below is the limiting factor time-wise
    final_x[:, 0:112, :] = x.numpy()
    arg_max_y = np.argmax(y, axis=1)
    concats = []
    for i, arg in enumerate(arg_max_y):
        if i > 2: break
        start, end, promotion = policy_index_to_squares(arg)
        concats.append(np.stack([start, end, promotion]))
    concats = np.stack(concats)
    final_x[:, 112:115, :] = concats
    return (final_x, y)

#@maia_chess_backend.logged_main
def main(config_path, name, collection_name):
    output_name = os.path.join('models', collection_name, name + '.txt')

    with open(config_path) as f:
        cfg = yaml.safe_load(f.read())

    maia_chess_backend.printWithDate(yaml.dump(cfg, default_flow_style=False))

    config_dictionary = dict(
                yaml=config_path,
    )

    wandbcfg = {}
    for k,v in cfg.items():
        if isinstance(v, dict):
            for k2,v2 in v.items():
                wandbcfg.update({k2:v2})
        else:
            wandbcfg.update({k:v})
    wandb.init("maia-chess", config=wandbcfg)
    print(wandb.config)

    experimental_parser = cfg['dataset'].get('experimental_v4_only_dataset', False)

    train_chunks = get_latest_chunks(cfg['dataset']['input_train'])
    test_chunks = get_latest_chunks(cfg['dataset']['input_test'])

    shuffle_size = cfg['training']['shuffle_size']
    total_batch_size = cfg['training']['batch_size']
    batch_splits = cfg['training'].get('num_batch_splits', 1)
    if total_batch_size % batch_splits != 0:
        raise ValueError('num_batch_splits must divide batch_size evenly')
    split_batch_size = total_batch_size // batch_splits
    # Load data with split batch size, which will be combined to the total batch size in tfprocess.
    maia_chess_backend.maia.ChunkParser.BATCH_SIZE = split_batch_size

    root_dir = os.path.join('models', collection_name, name)
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    #Change this line to TFProcess to run with original code (i.e. not the discriminator)
    tfprocess_d = maia_chess_backend.maia.TFProcessDiscriminator(cfg, name, collection_name)
    tfprocess_gen = maia_chess_backend.maia.TFProcess(cfg, name, collection_name)

    if experimental_parser:
        assert False
        train_dataset = tf.data.Dataset.from_tensor_slices(train_chunks).shuffle(len(train_chunks)).repeat()\
                         .interleave(lambda x: tf.data.FixedLengthRecordDataset(x, 8292, compression_type='GZIP', num_parallel_reads=1).filter(sample), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                         .shuffle(shuffle_size)\
                         .batch(split_batch_size).map(extract_inputs_outputs).prefetch(4)
    else:
        train_parser = maia_chess_backend.maia.ChunkParser(FileDataSrc(train_chunks.copy()),
                shuffle_size=shuffle_size, sample=SKIP,
                batch_size=maia_chess_backend.maia.ChunkParser.BATCH_SIZE,
                workers=1)
        train_dataset = tf.data.Dataset.from_generator(
            train_parser.parse, output_types=(tf.string, tf.string, tf.string, tf.string))
        train_dataset = train_dataset.map(maia_chess_backend.maia.ChunkParser.parse_function)
        train_dataset = train_dataset.prefetch(4)

    shuffle_size = int(shuffle_size)

    if experimental_parser:
        test_dataset = tf.data.Dataset.from_tensor_slices(test_chunks).shuffle(len(test_chunks)).repeat()\
                         .interleave(lambda x: tf.data.FixedLengthRecordDataset(x, 8292, compression_type='GZIP', num_parallel_reads=1).filter(sample), num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                         .shuffle(shuffle_size)\
                         .batch(split_batch_size).map(extract_inputs_outputs).prefetch(4)
    else:
        test_parser = maia_chess_backend.maia.ChunkParser(FileDataSrc(test_chunks),
                shuffle_size=shuffle_size, sample=SKIP,
                batch_size=maia_chess_backend.maia.ChunkParser.BATCH_SIZE,
                workers=1)
        test_dataset = tf.data.Dataset.from_generator(
            test_parser.parse, output_types=(tf.string, tf.string, tf.string, tf.string))
        # below must be changed to parse_function if it is not the discriminator being used
        test_dataset = test_dataset.map(maia_chess_backend.maia.ChunkParser.parse_function)
        test_dataset = test_dataset.prefetch(4)

    #pdb.set_trace()
    # print('before iter')
    # trainiter = iter(train_dataset)
    # new_data = []
    # for i in range(2):
    #     data = next(trainiter)
    #     x = data[0]
    #     y = data[1]
    #     new_x, new_y = gen_discriminator_data(x, y)
    #     new_data.append((new_x, new_y))
    #
    # print('after loop')
    #
    # discriminator_dataset = tf.data.Dataset.from_tensors(new_data)
    # pdb.set_trace()

    tfprocess_d.init_v2(train_dataset, test_dataset)
    tfprocess_d.restore_v2()

    tfprocess_gen.init_v2(train_dataset, test_dataset)
    tfprocess_gen.restore_v2()

    tfprocess_d.gen_model = tfprocess_gen.model

    # pari: what is the "the 10 samples per test game" mentioned in the
    # comment below? how are these sampled -- aren't we looping through all
    # test positions?

    # If number of test positions is not given
    # sweeps through all test chunks statistically
    # Assumes average of 10 samples per test game.
    # For simplicity, testing can use the split batch size instead of total batch size.
    # This does not affect results, because test results are simple averages that are independent of batch size.
    num_evals = cfg['training'].get('num_test_positions', len(test_chunks) * 10)
    num_evals = max(1, num_evals // maia_chess_backend.maia.ChunkParser.BATCH_SIZE)
    print("Using {} evaluation batches".format(num_evals))

    num_evals_train = cfg['training'].get('num_train_positions', len(train_chunks) * 10)
    num_evals_train = max(1, num_evals_train // maia_chess_backend.maia.ChunkParser.BATCH_SIZE)
    print("Using {} evaluation batches for train".format(num_evals_train))

    num_iterations = 5
    for _ in range(num_iterations):
            tfprocess_gen.process_loop_v2(total_batch_size, num_evals, num_evals_train, batch_splits=batch_splits)
            tfprocess_d.process_loop_v2(total_batch_size, num_evals, num_evals_train, batch_splits=batch_splits)
    if cfg['training'].get('swa_output', False):
        tfprocess_d.save_swa_weights_v2(output_name)
    else:
        tfprocess_d.save_leelaz_weights_v2(output_name)

    train_parser.shutdown()
    test_parser.shutdown()

def get_latest_chunks(path):
    chunks = []
    #maia_chess_backend.printWithDate(f"found {glob.glob(path)} chunk dirs")
    for d in glob.glob(path):
        #maia_chess_backend.printWithDate(f"found {len(chunks)} chunks", end = '\r')
        chunks += glob.glob(os.path.join(d, '*.gz'))
    maia_chess_backend.printWithDate(f"found {len(chunks)} chunks total")
    if len(chunks) < 10:
        print("Not enough chunks {}".format(len(chunks)))
        sys.exit(1)
    if len(chunks) < 1000:
        print("There are not very many chunks so results may be unstable")

    print("sorting {} chunks...".format(len(chunks)), end='')
    chunks.sort(key=os.path.getmtime, reverse=True)
    print("[done]")
    print("{} - {}".format(os.path.basename(chunks[-1]), os.path.basename(chunks[0])))
    random.shuffle(chunks)
    return chunks


class FileDataSrc:
    """
        data source yielding chunkdata from chunk files.
    """
    def __init__(self, chunks):
        self.chunks = []
        self.done = chunks
    def next(self):
        if not self.chunks:
            self.chunks, self.done = self.done, self.chunks
            random.shuffle(self.chunks)
        if not self.chunks:
            return None
        while len(self.chunks):
            filename = self.chunks.pop()
            try:
                with gzip.open(filename, 'rb') as chunk_file:
                    self.done.append(filename)
                    return chunk_file.read()
            except:
                print("failed to parse {}".format(filename))

def extract_inputs_outputs(raw):
    # first 4 bytes in each batch entry are boring.
    # Next 7432 are easy, policy extraction.
    policy = tf.io.decode_raw(tf.strings.substr(raw, 4, 7432), tf.float32)
    # Next are 104 bit packed chess boards, they have to be expanded.
    bit_planes = tf.expand_dims(tf.reshape(tf.io.decode_raw(tf.strings.substr(raw, 7436, 832), tf.uint8), [-1, 104, 8]), -1)
    bit_planes = tf.bitwise.bitwise_and(tf.tile(bit_planes, [1, 1, 1, 8]), [128, 64, 32, 16, 8, 4, 2, 1])
    bit_planes = tf.minimum(1., tf.cast(bit_planes, tf.float32))
    # Next 5 planes are 1 or 0 to indicate 8x8 of 1 or 0.
    unit_planes = tf.expand_dims(tf.expand_dims(tf.io.decode_raw(tf.strings.substr(raw, 8268, 5), tf.uint8), -1), -1)
    unit_planes = tf.cast(tf.tile(unit_planes, [1, 1, 8, 8]), tf.float32)
    # rule50 count plane.
    rule50_plane = tf.expand_dims(tf.expand_dims(tf.io.decode_raw(tf.strings.substr(raw, 8273, 1), tf.uint8), -1), -1)
    rule50_plane = tf.cast(tf.tile(rule50_plane, [1, 1, 8, 8]), tf.float32)
    rule50_plane = tf.divide(rule50_plane, 99.)
    # zero plane and one plane
    zero_plane = tf.zeros_like(rule50_plane)
    one_plane = tf.ones_like(rule50_plane)
    inputs = tf.reshape(tf.concat([bit_planes, unit_planes, rule50_plane, zero_plane, one_plane], 1), [-1, 112, 64])

    # winner is stored in one signed byte and needs to be converted to one hot.
    winner = tf.cast(tf.io.decode_raw(tf.strings.substr(raw, 8275, 1), tf.int8), tf.float32)
    winner = tf.tile(winner, [1,3])
    z = tf.cast(tf.equal(winner, [1., 0., -1.]), tf.float32)

    # Outcome distribution needs to be calculated from q and d.
    best_q = tf.io.decode_raw(tf.strings.substr(raw, 8280, 4), tf.float32)
    best_d = tf.io.decode_raw(tf.strings.substr(raw, 8288, 4), tf.float32)
    best_q_w = 0.5 * (1.0 - best_d + best_q)
    best_q_l = 0.5 * (1.0 - best_d - best_q)

    q = tf.concat([best_q_w, best_d, best_q_l], 1)

    return (inputs, policy, z, q)

def sample(x):
    return tf.math.equal(tf.random.uniform([], 0, SKIP-1, dtype=tf.int32), 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tensorflow pipeline for training Leela Chess.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config', help='config file for model / training')
    args = parser.parse_args()

    collection_name = os.path.basename(os.path.dirname(args.config)).replace('configs_', '')
    name = os.path.basename(args.config).split('.')[0]

    #multiprocessing.set_start_method('spawn')
    main(args.config, name, collection_name)
    #multiprocessing.freeze_support()
