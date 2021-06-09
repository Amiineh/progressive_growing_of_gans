import glob
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from tqdm import tqdm
import tensorflow as tf
import pickle
import imageio
import numpy as np
import dataset
import config
import os
import cv2

result_path = "results/001-pgan-carpet-4l-1024-preset-v2-1gpu-fp32/"
tfr_path = os.path.join("datasets", "carpet-4l-256")
data_path = "F:\\Carpet GAN\\all_data\\4l\\"
latent_dim = 512


# img, *imgs = [Image.open(f) for f in sorted(glob.glob(result_path + 'fakes*.png'))]
def complete_quarter(img, res= None):
    if res is None:
        w, h = img.size
    else:
        w = h = res
    comp = Image.new('RGB', (w * 2, h * 2))
    comp.paste(img, (0, 0))
    mirror = ImageOps.mirror(img)
    comp.paste(mirror, (w, 0))
    flip = ImageOps.flip(img)
    comp.paste(flip, (0, h))
    flip_mirror = ImageOps.mirror(flip)
    comp.paste(flip_mirror, (w, h))
    comp = comp.resize((w * 3, w * 4))
    return comp


def make_gif():
    for idx in tqdm(range(2, 61)):
        imgs = []
        for f in glob.glob(result_path + 'fakes*.png'):
            img = Image.open(f)
            w, h = img.size
            num = w // 15
            imgs.append(img.crop((num * idx, num * idx, num * (idx + 1), num * (idx + 1))).resize((256, 256)))
        imgs[0].save(fp=result_path + 'fake' + str(idx) + '.gif', format='GIF', append_images=imgs, save_all=True,
                     duration=200, loop=True)

def make_gif_interpolated():
    imgs = []
    for i in range(45):
        for f in glob.glob(result_path + 'img' + str(i) + '.png'):
            img = Image.open(f).resize((256, 256))
            comp = complete_quarter(img)
            imgs.append(comp)
    imgs[0].save(fp=result_path + 'interpolated.gif', format='GIF', append_images=imgs,
                 save_all=True, optimize=True,
                 duration=300, loop=True)

# make_gif_interpolated()

def make_comp_gif():
    # imgs = []
    i = 0
    for f in tqdm(glob.glob(result_path + 'fakes*.png')):
        img = Image.open(f)
        l = img.size[0] // 3
        row, col = 2, 1
        img = img.crop((row * l, col * l, (row + 1) * l, (col + 1) * l))
        comp = complete_quarter(img)
        # comp.save(result_path+'comp_fakes/6/' + str(i) + '.png')
        i += 1
        # imgs.append(comp)
    # imgs[1].save(fp=result_path + 'comp' + str(row) + str(col) + '.gif', format='GIF', append_images=imgs,
    #              save_all=True, optimize = True,
    #              duration=400, loop=True)


def make_comp_vid(): # not working now
    imgs = []
    video = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc(*'XVID'), 24, (1200, 800))
    for f in tqdm(glob.glob(result_path + 'fakes*.png')):
        img = Image.open(f)
        l = img.size[0] // 3
        row, col = 0, 0
        img = img.crop((row * l, col * l, (row + 1) * l, (col + 1) * l))
        comp = complete_quarter(img)
        # imgs.append(comp)
        video.write(np.array(comp))


def make_comp_image():
    for f in glob.glob(result_path + 'fakes011980.png'):
        img = Image.open(f)
        l = img.size[0] // 3
        for col in range(3):
            tmp_img = img
            tmp_img = tmp_img.crop((col * l, l, (col + 1) * l, l*2))
            tmp_img = complete_quarter(tmp_img)
            tmp_img.save(result_path + 'comp_fake_' + str(col + 10) + '.png')

make_comp_image()
def get_nearest_neighbor(k=3):
    # training_set = dataset.load_dataset(data_dir=data_path, verbose=True, **config.dataset)
    data, data_raw = [], []
    res = 512
    for f in glob.glob(data_path + "*.jpg"):
        img = Image.open(f)
        data_raw.append(np.array(img.resize((res, res))))
        img = img.filter(ImageFilter.GaussianBlur(2))
        img = img.resize((res, res))
        img = np.array(img).reshape((res * res * 3))
        data.append(img)
    data = np.array(data)
    data_raw = np.array(data_raw)

    for f in tqdm(glob.glob(result_path + '/comp_fake/*1.png')):
        img = Image.open(f)
        img = img.resize((res, res))
        img = img.filter(ImageFilter.GaussianBlur(2))
        img = np.array(img).reshape((res * res * 3))
        distances = np.sqrt(np.sum(np.square(img - data), axis=1))
        knn = data_raw[distances.argsort()[-k:]].reshape((-1, res, res, 3))
        all = Image.new('RGB', [(k + 1) * res, res])
        for i in range(k):
            inn = Image.fromarray(knn[i], 'RGB')
            inn.show()
            inn.save(result_path+'nn_'+str(i)+'.png')
            # all.paste(inn, [(k - i - 1) * res, 0])
        # paste fake image at right most spot
        # all.paste(img, [k * res, 0])
        # draw = ImageDraw.Draw(all)
        # draw.line([k * res, 0, k * res, res], fill=(0, 255, 0), width=5)
        # all.save(os.path.splitext(f)[0] + '_all.jpg')


def interpolate_hypersphere(v1, v2, num_steps):
    v1_norm = tf.norm(v1)
    v2_norm = tf.norm(v2)
    v2_normalized = v2 * (v1_norm / v2_norm)

    vectors = np.zeros([num_steps, ])
    for step in range(num_steps):
        interpolated = v1 + (v2_normalized - v1) * step / (num_steps - 1)
        interpolated_norm = tf.norm(interpolated)
        interpolated_normalized = interpolated * (v1_norm / interpolated_norm)
        # vectors.append(interpolated_normalized)
        vectors[step] = interpolated_normalized
    return np.array(vectors)


def interpolate_hyperline(p1, p2, n_steps=10):
    # interpolate ratios between the points
    ratios = np.linspace(0, 1, num=n_steps)
    # linear interpolate vectors
    vectors = list()
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return np.asarray(vectors)


def interpolate_between_vectors():
    v = np.random.RandomState(1000).randn(2, 512)
    vectors = interpolate_hypersphere(v[0], v[1], 50)
    return vectors


def animate(images):
    images = np.array(images)
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave(result_path + 'animation.gif', converted_images)


# with open(result_path + 'network-final.pkl', 'rb') as file:
#     G, D, Gs = pickle.load(file)
#
#     # Generate latent vectors.
#     latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:])  # 1000 random latents
#     latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]]  # hand-picked top-10
#
#     # Generate dummy labels (not used by the official networks).
#     labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
#     interpolated_images = Gs.run(latents, labels)
#     animate(interpolated_images)

def sep_labeled_dir():
    from carpet_dataset import get_indices
    from shutil import copyfile

    l4s = get_indices([['4', 'l']])
    for l4 in l4s:
        if os.path.exists(data_path + str(l4)):
            copyfile(data_path + str(l4), os.path.join(data_path, '4l', str(l4)))
