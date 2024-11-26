from time import time, sleep
import  network
import tensorflow as tf
import numpy as np
import os
import get_data as loader
import argparse
import ntpath
from tensorflow.python.client import device_lib

def my_bool(s):
    return s != 'False'

def init_parameters():
    parser = argparse.ArgumentParser(description='bacteria_staining')
    parser.add_argument('--gpu_id', type=int, default=1, help='the ID of the visible GPU device (only used when not in parallel mode)')
    parser.add_argument('--input', type=str, default='df_zstack2', help='only_df,df_zstack1,df_zstack2')
    parser.add_argument('--data_name', type=str, default='exp_1', help='')
    parser.add_argument('--inverted_input', type=my_bool, default=False, help='')
    parser.add_argument('--testbool', type=my_bool, default=True, help='# test')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for data loading')
    parser.add_argument('--image_size', type=int, default=1024, help='img size for data loading')
    parser.add_argument('--n_channels', type=int, default=16, help='')
    parser.add_argument('--n_threads', type=int, default=12, help='')
    parser.add_argument('--n_levels', type=int, default=5, help='')
    parser.add_argument('--test_path', type=str, default='./test_data/input/*.npy', help='')

    tc = parser.parse_args()
    vc = parser.parse_args()
    vc.record_file = vc.test_path

    return tc, vc

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':
    train_config, valid_config = init_parameters()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{train_config.gpu_id}"

    if not os.path.exists(f'{valid_config.data_name}/test_images/'):
        os.makedirs(f'{valid_config.data_name}/test_images/')

    with tf.Graph().as_default(), tf.device('/cpu:0'):


        valid_iterator = loader.get_dataset_iterator_bacteria_npy(valid_config.record_file,  train_config,valid_config,'test')
        valid_x, valid_y, valid_path = valid_iterator.get_next()

        device = get_available_gpus()[0]
        with tf.device(device):

            with tf.compat.v1.variable_scope('Generator'):
                valid_G = network.Generator(valid_x, valid_config)

            SR_out = valid_G.output
            label = valid_y
            input = valid_x
            path = valid_path

        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver = tf.compat.v1.train.Saver(max_to_keep=0)

            saver.restore(sess, tf.train.latest_checkpoint(f'ckpts/'))


            counter = 0
            sess.run(valid_iterator.initializer)
            while True:
                try:
                    res = sess.run([SR_out,label,input,path])

                    sr_out, lab, inp, paths = res[0], res[1], res[2], res[3]
                    ind = 0#np.random.randint(0, SR_out_int.shape[0])

                    hr_crop_size = np.shape(sr_out[ind, :, :, :])[0]#1024

                    sr_out = loader.fixed_cropv2(sr_out[ind, :, :], hr_crop_size, 1)
                    lab = loader.fixed_cropv2(lab[ind, :, :], hr_crop_size, 1)
                    inp = loader.fixed_cropv2(inp[ind, :, :], hr_crop_size, 1)


                    short_path = ntpath.basename(paths[0].decode('utf-8'))
                    name = os.path.splitext(short_path)[0]

                    fold_name = f'test_images'

                    contrast_fac = 1
                    if train_config.input == 'only_df':
                        inp = np.squeeze(inp)
                        inp_df = inp

                        meano = np.mean(inp_df)
                        inp_df = (inp_df - meano) * contrast_fac + meano
                        maxo = np.max(inp_df)
                        inp_df = inp_df/maxo
                        # inp_df = np.clip(inp_df, 0, 1)
                        loader.save_pure_image(inp_df, image_ID=f'{name}_inp_df',
                                               image_save_path=f'{valid_config.data_name}/{fold_name}/')


                    elif train_config.input == 'df_zstack1':
                        inp = np.squeeze(inp)
                        for k in range(0, 3):
                            inp_df = inp[:, :, k:k + 1]
                            meano = np.mean(inp_df)
                            inp_df = (inp_df - meano) * contrast_fac + meano
                            maxo = np.max(inp_df)
                            inp_df = inp_df/maxo
                            # inp_df = np.clip(inp_df, 0, 1)
                            inp_df = np.squeeze(inp_df)
                            loader.save_pure_image(inp_df,
                                                   image_ID=f'{name}_inp_df_0min1plus1_{k}',
                                                   image_save_path=f'{valid_config.data_name}/{fold_name}/')



                    elif train_config.input == 'df_zstack2':
                        inp = np.squeeze(inp)
                        for k in range(0, 5):
                            inp_df = inp[:, :, k:k + 1]
                            meano = np.mean(inp_df)
                            inp_df = (inp_df - meano) * contrast_fac + meano
                            maxo = np.max(inp_df)
                            inp_df = inp_df/maxo
                            # inp_df = np.clip(inp_df, 0, 1)
                            inp_df = np.squeeze(inp_df)
                            loader.save_pure_image(inp_df,
                                                   image_ID=f'{name}_inp_df_0min1plus1_{k}',
                                                   image_save_path=f'{valid_config.data_name}/{fold_name}/')



                    lab2 = np.clip(lab, 0, 1)
                    sr_out2 = np.clip(sr_out, 0, 1)


                    loader.save_pure_image(sr_out2, image_ID=f'{name}_out',
                                       image_save_path=f'{valid_config.data_name}/{fold_name}/')
                    loader.save_pure_image(lab2, image_ID=f'{name}_tar',
                                       image_save_path=f'{valid_config.data_name}/{fold_name}/')

                    ######################################################################################################################
                    # loader.save_label_recon_image_mat(sr_out2,lab2, image_ID=f'{name}',
                    #                    image_save_path=f'{valid_config.data_name}/{fold_name}/')


                    counter += 1
                except tf.errors.OutOfRangeError:
                    break




