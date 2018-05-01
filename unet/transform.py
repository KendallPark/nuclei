from threading import Lock
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def double_batch_generator(augmentor, images, labels, batch_size):
    seq = iaa.Sequential(augmentor,
                         random_order=True)

    seq_copy = seq.deepcopy()
    
    aug_seed = 0
    while True:
        x_batch, y_batch = form_double_batch(images, labels, batch_size)
        seq.reseed(aug_seed)
        seq_copy.reseed(aug_seed)
        new_x_batch = seq.augment_images(x_batch)
        new_y_batch = seq_copy.augment_images(y_batch)
        new_x_batch = np.array(new_x_batch).astype('float16')
        new_y_batch = np.array(new_y_batch).astype('float16')
        aug_seed+=1
        yield new_x_batch, new_y_batch

def form_double_batch(x, y, batch_size):
    idx = np.random.randint(0, x.shape[0], int(batch_size))
    return x[idx], y[idx]

def make_double_generator(X_train, Y_train, batch_size, augmentor):
        return double_batch_generator(augmentor, X_train, Y_train, batch_size)