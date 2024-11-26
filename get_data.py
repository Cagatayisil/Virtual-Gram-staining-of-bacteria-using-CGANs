import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import scipy.io as sio
import glob
import os


def random_crop(lr_img, hr_img, hr_crop_size=256, scale=1):
    lr_crop_size = hr_crop_size // scale
    #lr_img = lr_img[::scale, ::scale, :]


    lr_img_shape = tf.shape(input=lr_img)[:2]

    lr_w = tf.random.uniform(shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size,:]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size,:]
    #lr_img_cropped = tf.expand_dims(lr_img_cropped, 2)

    return lr_img_cropped, hr_img_cropped

def fixed_crop(lr_img, hr_img, hr_crop_size=256, scale=1):
    lr_crop_size = hr_crop_size // scale
    #lr_img = lr_img[::scale, ::scale, :]


    lr_img_shape = tf.shape(input=lr_img)[:2]

    lr_w = (lr_img_shape[1]-lr_crop_size)// 2
    lr_h = (lr_img_shape[0]-lr_crop_size)// 2

    hr_w = lr_w * scale
    hr_h = lr_h * scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]
    #lr_img_cropped = tf.expand_dims(lr_img_cropped, 2)

    return lr_img_cropped, hr_img_cropped
    
def random_cropv2(lr_img, lr_h,lr_w, lr_crop_size=256,scale = 1):
    lr_crop_size = lr_crop_size // scale
    lr_w = lr_w //scale
    lr_h = lr_h //scale

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    #lr_img_cropped = tf.expand_dims(lr_img_cropped, 2)
    return lr_img_cropped

def fixed_cropv2(lr_img, hr_crop_size=256, scale=1):
    lr_crop_size = int(hr_crop_size // scale)
    #lr_img = lr_img[::scale, ::scale, :]

    lr_img_shape = np.shape(lr_img)[:2]

    lr_w = int((lr_img_shape[1]-lr_crop_size)// 2)
    lr_h = int((lr_img_shape[0]-lr_crop_size)// 2)

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    #lr_img_cropped = tf.expand_dims(lr_img_cropped, 2)

    return lr_img_cropped

def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(pred=rn < 0.5,
                   true_fn=lambda: (lr_img, hr_img),
                   false_fn=lambda: (tf.image.flip_left_right(lr_img),
                            tf.image.flip_left_right(hr_img)))


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rn), tf.image.rot90(hr_img, rn)


def _preprocess_npy_zstack(inp, label, path, train_config):
    shapes = tf.shape(input=inp)
    sized = tf.cast(shapes[0],tf.float32)
    inp = tf.reshape(inp, [sized, sized, 5])
    label = tf.reshape(label, [sized, sized, 3])

    df_zstack = inp

    if train_config.input == 'only_df':
        input = df_zstack[:,:,:1]
    elif train_config.input == 'df_zstack1':
        input = df_zstack[:, :, :3]
    elif train_config.input == 'df_zstack2':
        input = df_zstack[:, :, :5]


    if train_config.inverted_input is True:
        input = 1-input

    #(input, label) = random_crop(input, label, train_config.image_size, 1)  # opt.scale

    if train_config.testbool is False:
        (input, label) = random_crop(input, label, train_config.image_size,1)#opt.scale
        (input, label) = random_flip(input, label)
        (input, label) = random_rotate(input, label)
    else:
        (input, label) = fixed_crop(input, label, train_config.image_size,1)#opt.scale

    return input, label, path
    

# Load the numpy files
def map_func(inp_path):
    #print(inp_path)
    inp = np.load(inp_path)

    lr_h = 0
    inp = inp[lr_h:inp.shape[0]-lr_h, lr_h:inp.shape[1]-lr_h,:]

    # print(inp.shape) 
    tar = np.load(inp_path.decode('utf-8').replace('input','label'))
    # print(tar.shape) 

    avg_max = 0.1760

    df_zstack = inp[:,:,:5]
    # bf_in = inp[:,:,5:]

    inp = df_zstack/avg_max
    # inp = np.concatenate((df_zstack,bf_in), axis=2)
    # print(inp.shape) 

    return inp, tar, inp_path
  
  
def get_dataset_iterator_bacteria_npy(filename,  train_config,valid_config, data_type):

    K = tf.reduce_sum(input_tensor=tf.cast(tf.logical_or(tf.equal(data_type, 'train'), tf.equal(data_type, 'valid')), tf.int32))

    def fshuffle(train_config): return 99999

    def fnoshuffle(): return 1

    shuffle_sz = tf.cast(tf.cond(pred=tf.equal(K, 1), true_fn=lambda: fshuffle(train_config), false_fn=fnoshuffle), dtype=tf.int64)

    # print(filename)
    image_list = glob.glob(filename)#+glob.glob(filename[1])

###########TRAINING ONLY
    if train_config.testbool is False:
        random.shuffle(image_list)

    dataset = tf.data.Dataset.from_tensor_slices(image_list)
    
    dataset = dataset.shuffle(shuffle_sz, reshuffle_each_iteration=True)

    dataset = dataset.map(lambda item1 : tf.numpy_function(
              map_func, [item1], [tf.float32, tf.float32, tf.string]),
              num_parallel_calls=train_config.n_threads)


    dataset = dataset.map(lambda x, y, z : _preprocess_npy_zstack(x, y, z, train_config), num_parallel_calls=train_config.n_threads)

    def batch_sz_train(train_config): return train_config.batch_size

    def batch_sz_valid(valid_config): return valid_config.batch_size

    if train_config.testbool is False:
        BS = tf.cast(tf.cond(pred=tf.equal(K,1),true_fn=lambda: batch_sz_train(train_config),false_fn=lambda: batch_sz_valid(valid_config)), dtype=tf.int64)
    else:
        BS = tf.cast(1, dtype=tf.int64)
    dataset = dataset.batch(BS, drop_remainder=True)
    dataset = dataset.prefetch(1)
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)

    return iterator
    
def save_single_image(img, tit, vmin,vmax,image_ID='img0', image_save_path='./'):


	fig, ax = plt.subplots()
	ax.set_title(tit)
	x1 = plt.imshow(img, cmap='gray',vmin=vmin, vmax=vmax)
	plt.colorbar(x1, ax=ax)

	plt.tight_layout()
	plt.savefig(f'{image_save_path}/{image_ID}.png', bbox_inches='tight')
	plt.close(fig)
	plt.clf()


def save_pure_image(img, image_ID='img0', image_save_path='./'):
    imgk = img * 255
    pil_img = Image.fromarray(imgk.astype(np.uint8))

    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
        
    # asd = os.path.join(f'{image_save_path}')

    # pil_img.save(os.path.join(f'{image_save_path}',f'{image_ID}.jpg')) 

    pil_img.save(f'{image_save_path}/{image_ID}.jpg')

def save_single_image_mat(img, image_ID='img0', image_save_path='./'):

    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)

    # asd = os.path.join(f'{image_save_path}')

    sio.savemat(f'{image_save_path}/{image_ID}.mat',
                {f'img': np.array(img)})
    # sio.savemat(os.path.join(f'{image_save_path}',f'{image_ID}.mat'),
    #             {f'img': np.array(img)})
             
def save_label_recon_image_mat(img, lab, image_ID='img0', image_save_path='./'):

    sio.savemat(f'{image_save_path}/{image_ID}.mat',
                {f'out': np.array(img),f'tar': np.array(lab)})                
