import os
import sys
import getopt
import numpy as np
import pix2pix as m
import random
from tqdm import tqdm
from keras.optimizers import Adam
from utils import MyDict, log, save_weights, load_weights, load_losses, create_expt_dir
from scipy.misc import imread, imresize
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

# add to ~/.config/matplotlib/matplotlibrc line backend : Agg


def read_resize(fn):
    rgb = imresize(imread(fn), (256, 256)).astype(np.float)
    label = imresize(imread(fn.replace('.jpg', '.png')), (256, 256)).astype(np.float)
    return rgb / 127.5 - 1, label / 127.5 - 1


def datagen(batch_size=16):
    pool = ThreadPool(cpu_count())
    datapath = '../../datasets/facades/'
    imgnames = [os.path.join(root, f)
                 for root, _, files in os.walk(datapath)
                 for f in files if f.endswith('.jpg')]
    n = len(imgnames)
    k = 0
    while True:
        x = np.zeros((batch_size, 256, 256, 3))
        y = np.zeros((batch_size, 256, 256, 3))

        batch = []
        for i in range(batch_size):
            batch.append(imgnames[k])
            k = (k + 1) % n
            if not k:
                random.shuffle(imgnames)

        result = pool.map(read_resize, batch)

        for i, r in enumerate(result):
            x[i] = r[1]
            y[i] = r[0]
            # img = imread(datapath + 'input_1024/' + imgnames[k]).astype(np.float)
            # dif = np.load(datapath + 'output-input_1024/' + imgnames[k]+'.npy').astype(np.float)
            # X[i] = img / 127.5 - 1
            # y[i] = dif / 255.
            # k = (k + 1) % n
            # if not k:
            #     random.shuffle(imgnames)
        yield x, y


def discriminator_generator(it, atob, dout_size):
    """
    Generate batches for the discriminator.

    Parameters:
    - it: an iterator that returns a pair of images;
    - atob: the generator network that maps an image to another representation;
    - dout_size: the size of the output of the discriminator.
    """
    while True:
        # Fake pair
        a_fake, _ = it.next()
        b_fake = atob.predict(a_fake)

        # Real pair
        a_real, b_real = it.next()

        # Concatenate the channels. Images become (ch_a + ch_b) x 256 x 256
        fake = np.concatenate((a_fake, b_fake), axis=-1)
        real = np.concatenate((a_real, b_real), axis=-1)

        # Concatenate fake and real pairs into a single batch
        batch_x = np.concatenate((fake, real), axis=0)

        # 1 is fake, 0 is real
        batch_y = np.ones((batch_x.shape[0], 1) + dout_size)
        batch_y[fake.shape[0]:] = 0

        yield batch_x, batch_y


def train_discriminator(d, it, samples_per_batch=20):
    """Train the discriminator network."""
    return d.fit_generator(it, steps_per_epoch=samples_per_batch*2, epochs=1, verbose=False)


def pix2pix_generator(it, dout_size):
    """
    Generate data for the generator network.

    Parameters:
    - it: an iterator that returns a pair of images;
    - dout_size: the size of the output of the discriminator.
    """
    for a, b in it:
        # 1 is fake, 0 is real
        y = np.zeros((a.shape[0],)+dout_size + (1,))
        yield [a, b], y


def train_pix2pix(pix2pix, it, samples_per_batch=2):
    """Train the generator network."""
    return pix2pix.fit_generator(it, epochs=1, steps_per_epoch=samples_per_batch, verbose=False)


def evaluate(models, generators, losses, val_samples=192):
    """Evaluate and display the losses of the models."""
    # Get necessary generators
    d_gen = generators.d_gen_val
    p2p_gen = generators.p2p_gen_val

    # Get necessary models
    d = models.d
    p2p = models.p2p

    # Evaluate
    d_loss = d.evaluate_generator(d_gen, val_samples)
    p2p_loss = p2p.evaluate_generator(p2p_gen, val_samples)

    losses['d_val'].append(d_loss)
    losses['p2p_val'].append(p2p_loss)

    print ''
    print ('Train Losses of (D={0} / P2P={1});\n'
           'Validation Losses of (D={2} / P2P={3})'.format(
                losses['d'][-1], losses['p2p'][-1], d_loss, p2p_loss))

    return d_loss, p2p_loss


def model_creation(d, atob, params):
    """Create all the necessary models."""
    opt = Adam(lr=params.lr, beta_1=params.beta_1)
    p2p = m.pix2pix(atob, d, params.a_ch, params.b_ch, alpha=params.alpha, opt=opt,
                    is_a_binary=params.is_a_binary, is_b_binary=params.is_b_binary)

    models = MyDict({
        'atob': atob,
        'd': d,
        'p2p': p2p,
    })

    return models


def generators_creation(it_train, it_val, models, dout_size):
    """Create all the necessary data generators."""
    # Discriminator data generators
    d_gen = discriminator_generator(it_train, models.atob, dout_size)
    d_gen_val = discriminator_generator(it_val, models.atob, dout_size)

    # Workaround to make tensorflow work. When atob.predict is called the first
    # time it calls tf.get_default_graph. This should be done on the main thread
    # and not inside fit_generator. See https://github.com/fchollet/keras/issues/2397
    next(d_gen)

    # pix2pix data generators
    p2p_gen = pix2pix_generator(it_train, dout_size)
    p2p_gen_val = pix2pix_generator(it_val, dout_size)

    generators = MyDict({
        'd_gen': d_gen,
        'd_gen_val': d_gen_val,
        'p2p_gen': p2p_gen,
        'p2p_gen_val': p2p_gen_val,
    })

    return generators


def train_iteration(models, generators, losses, params):
    """Perform a train iteration."""
    # Get necessary generators
    d_gen = generators.d_gen
    p2p_gen = generators.p2p_gen

    # Get necessary models
    d = models.d
    p2p = models.p2p

    # Update the dscriminator
    dhist = train_discriminator(d, d_gen, samples_per_batch=params.samples_per_batch)
    losses['d'].extend(dhist.history['loss'])

    # Update the generator
    p2phist = train_pix2pix(p2p, p2p_gen, samples_per_batch=params.samples_per_batch)
    losses['p2p'].extend(p2phist.history['loss'])


def train(models, it_train, it_val, params):
    """
    Train the model.

    Parameters:
    - models: a dictionary with all the models.
        - atob: a model that goes from A to B.
        - d: the discriminator model.
        - p2p: a Pix2Pix model.
    - it_train: the iterator of the training data.
    - it_val: the iterator of the validation data.
    - params: parameters of the training procedure.
    - dout_size: the size of the output of the discriminator model.
    """
    # Create the experiment folder and save the parameters
    create_expt_dir(params)

    # Get the output shape of the discriminator
    dout_size = models.d.output_shape[-3:-1]
    # Define the data generators
    generators = generators_creation(it_train, it_val, models, dout_size)

    # Define the number of samples to use on each training epoch
    train_samples = params.train_samples
    if params.train_samples == -1:
        train_samples = 8
    batches_per_epoch = train_samples // params.samples_per_batch

    # Define the number of samples to use for validation
    val_samples = params.val_samples
    if val_samples == -1:
        val_samples = 8

    losses = {'p2p': [], 'd': [], 'p2p_val': [], 'd_val': []}
    if params.continue_train:
        losses = load_losses(log_dir=params.log_dir, expt_name=params.expt_name)

    for e in tqdm(range(params.epochs)):

        for b in range(batches_per_epoch):
            train_iteration(models, generators, losses, params)

        # Evaluate how the models is doing on the validation set.
        evaluate(models, generators, losses, val_samples=val_samples)

        if (e + 1) % params.save_every == 0:
            save_weights(models, log_dir=params.log_dir, expt_name=params.expt_name)
            log(losses, models.atob, it_val, log_dir=params.log_dir, expt_name=params.expt_name,
                is_a_binary=params.is_a_binary, is_b_binary=params.is_b_binary)


if __name__ == '__main__':
    a = sys.argv[1:]

    params = MyDict({
        # Model
        'nfd': 32,  # Number of filters of the first layer of the discriminator
        'nfatob': 32,  # Number of filters of the first layer of the AtoB model
        'alpha': 100,  # The weight of the reconstruction loss of the atob model
        # Train
        'epochs': 1000,  # Number of epochs to train the model
        'batch_size': 16,  # The batch size
        'samples_per_batch': 5,  # The number of samples to train each model on each iteration
        'save_every': 10,  # Save results every 'save_every' epochs on the log folder
        'lr': 2e-4,  # The learning rate to train the models
        'beta_1': 0.5,  # The beta_1 value of the Adam optimizer
        'continue_train': False,  # If it should continue the training from the last checkpoint
        # File system
        'log_dir': 'log',  # Directory to log
        'expt_name': None,  # The name of the experiment. Saves the logs into a folder with this name
        'base_dir': '../../datasets/retouch/',  # Directory that contains the data
        'train_dir': 'train',  # Directory inside base_dir that contains training data
        'val_dir': 'val',  # Directory inside base_dir that contains validation data
        'train_samples': -1,  # The number of training samples. Set -1 to be the same as training examples
        'val_samples': -1,  # The number of validation samples. Set -1 to be the same as validation examples
        'load_to_memory': True,  # Whether to load the images into memory
        # Image
        'a_ch': 3,  # Number of channels of images A
        'b_ch': 3,  # Number of channels of images B
        'is_a_binary': False,  # If A is binary, its values will be either 0 or 1
        'is_b_binary': False,  # If B is binary, the last layer of the atob model is followed by a sigmoid
        'is_a_grayscale': False,  # If A is grayscale, the image will only have one channel
        'is_b_grayscale': False,  # If B is grayscale, the image will only have one channel
        'target_size': 256,  # The size of the images loaded by the iterator. DOES NOT CHANGE THE MODELS
        'rotation_range': 0.,  # The range to rotate training images for dataset augmentation
        'height_shift_range': 0.,  # Percentage of height of the image to translate for dataset augmentation
        'width_shift_range': 0.,  # Percentage of width of the image to translate for dataset augmentation
        'horizontal_flip': False,  # If true performs random horizontal flips on the train set
        'vertical_flip': False,  # If true performs random vertical flips on the train set
        'zoom_range': 0.,  # Defines the range to scale the image for dataset augmentation
    })

    param_names = [k + '=' for k in params.keys()] + ['help']

    try:
        opts, args = getopt.getopt(a, '', param_names)
    except getopt.GetoptError:
        sys.exit()

    for opt, arg in opts:
        if opt == '--help':
            sys.exit()
        elif opt in ('--nfatob' '--nfd', '--a_ch', '--b_ch', '--epochs', '--batch_size',
                     '--samples_per_batch', '--save_every', '--train_samples', '--val_samples',
                     '--target_size'):
            params[opt[2:]] = int(arg)
        elif opt in ('--lr', '--beta_1', '--rotation_range', '--height_shift_range',
                     '--width_shift_range', '--zoom_range', '--alpha'):
            params[opt[2:]] = float(arg)
        elif opt in ('--is_a_binary', '--is_b_binary', '--is_a_grayscale', '--is_b_grayscale',
                     '--continue_train', '--horizontal_flip', '--vertical_flip',
                     '--load_to_memory'):
            params[opt[2:]] = True if arg == 'True' else False
        elif opt in ('--base_dir', '--train_dir', '--val_dir', '--expt_name', '--log_dir'):
            params[opt[2:]] = arg

    dopt = Adam(lr=params.lr, beta_1=params.beta_1)

    # Define the U-Net generator
    unet = m.g_unet(params.a_ch, params.b_ch, params.nfatob,
                    batch_size=params.batch_size, is_binary=params.is_b_binary)

    # Define the discriminator
    d = m.discriminator(params.a_ch, params.b_ch, params.nfd, opt=dopt)

    if params.continue_train:
        load_weights(unet, d, log_dir=params.log_dir, expt_name=params.expt_name)

    ts = params.target_size
    train_dir = os.path.join(params.base_dir, params.train_dir)
    it_train = datagen(batch_size=params.batch_size)
    val_dir = os.path.join(params.base_dir, params.val_dir)
    it_val = datagen(batch_size=params.batch_size // 2)

    models = model_creation(d, unet, params)
    train(models, it_train, it_val, params)
