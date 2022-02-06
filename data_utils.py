from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import glob
import time


def timeit(func):
    def wrapper(*args,**kwargs):
        t1 = time.time()
        arg = func(*args,**kwargs)
        t2 = time.time()
        print("{} took {} seconds".format(func.__name__,t2-t1))
        return arg
    return wrapper

def timeit_torch(func):
    def wrapper(*args,**kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        arg = func(*args,**kwargs)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        print("{} took {} seconds".format(func.__name__,start.elapsed_time(end)))
        return arg
    return wrapper

def categorical_iou(preds,labels):
    '''

    :param preds: tensor shape (B,C,H,W) between [0,1]
    :param labels: tensor shape (B,C,H,W) between [0,1]
    :return: value beteen [0,1]
    '''
    intersection = (preds*labels).sum()
    union = (preds+labels).clip(0,1).sum()
    return intersection/union


def df_steel_to_pivot(df):
    df = df.fillna(False)
    #df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    #df['ClassId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[1])
    #df= df.drop(['ImageId_ClassId'],1)
    df = df[['ImageId','ClassId','EncodedPixels']].reset_index().drop(['index'],True)
    df_pivot = df.pivot('ImageId','ClassId','EncodedPixels')
    df_pivot.fillna(False,inplace=True)
    df_pivot = df_pivot.reset_index()
    return df_pivot


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background, (width,height)
    Returns run length as string formatted
    """
    assert len(img.shape) == 2, f"Error: Image has wrong dimensions: {img.shape}"

    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(mask_rle: str, label=1, shape=(1600,256)):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction

def tensor_to_set(tensor):
    return set(tensor.flatten().tolist())

def masks_to_one_hot(masks, N):
    '''
    Turns a batch of masks with different categories into one-hot masks
    Important: first mask is removed, as it is the complementary to sum of all other masks
    Parameters
    ----------
    N: number of categories
    masks: (batch_size,1,height,width) contains values from 0...N

    Returns
    -------
    masks_out: (batch_size,N,height,width) with binary content
    '''
    assert (masks >= 0).all(), "Masks contain negative values"
    batch_size = masks.shape[0]

    masks = masks.to(torch.int64)
    masks = masks.permute(0, 2, 3, 1)
    masks_out = torch.nn.functional.one_hot(masks, N + 1).squeeze()
    if batch_size == 1:
        masks_out = masks_out.unsqueeze(dim=0)
    masks_out = masks_out.permute(0, 3, 1, 2)
    return masks_out[:, 1:, :, :]  # zeros represent their own class, but this is redundant


def img_mask_overlay(imgs, masks, figsize=1, alpha=0.5, colors=None):
    """
    Plots images with overlayed masks

    Parameters
    ----------
    imgs: tensor of images, shape:(batch,3,height,width)
    masks: tensor of masks, shape:(batch,nr_categories,height,width)
    masks must be one-hot encoded
    figsize: set size of plot
    alpha: set alpha value of masks overlay
    colors: list of rgb tuples [(255,0,0), ...], len(colors) == nr_categories

    Returns
    -------
    Tensor containing images with mask overlay

    """
    assert imgs.shape[0] == masks.shape[0], "Error: got {} images but {} masks".format(imgs.shape[0], masks.shape[0])

    nr_categories = masks.shape[1]

    img_separator = torch.ones((imgs.shape[0], imgs.shape[1], max(1, int(masks.shape[2] * 0.1)), imgs.shape[3])) * 0.5
    mask_separator = torch.zeros((masks.shape[0], max(1, int(masks.shape[2] * 0.1)), masks.shape[3], 3))
    imgs = torch.cat((imgs, img_separator), axis=2)

    # stack images vertically
    imgs = imgs.permute([0, 2, 3, 1])
    imgs = imgs.reshape([imgs.shape[0] * imgs.shape[1], imgs.shape[2], 3])

    # masks = masks.reshape([1,masks.shape[1],masks.shape[0]*masks.shape[2],masks.shape[3]])

    # gray line between images
    masks_color = torch.empty(masks.shape[0], masks.shape[1], 3)

    if colors == None:
        colorpalette = sns.color_palette()
        colors = [list([i * 255.0 for i in j]) for j in colorpalette[0:nr_categories]]

    # create color tensor and perform dot product. this assigns a certain color to each defect
    # shape of color tensor is (1,nr_classes,3,1,1)
    # shape of mask tensor is (batch,nr_classes,1,height,width)
    # shape of color_mask is (batch,nr_classes,3,height,width)

    colortensor = torch.tensor(colors).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) / 255.0

    colormasks = masks.unsqueeze(2) * colortensor

    colormasks = (colormasks * alpha).sum(axis=1).permute([0, 2, 3, 1])  # collapse the class axis

    colormasks = torch.cat((colormasks, mask_separator), axis=1)

    # change shape to plot images
    colormasks = colormasks.reshape(
        (colormasks.shape[0] * colormasks.shape[1], colormasks.shape[2], colormasks.shape[3]))
    assert colormasks.shape == imgs.shape, "Error: Imgs shape {}, colormasks shape {}".format(imgs.shape,
                                                                                              colormasks.shape)

    imgs_out = imgs + colormasks

    plt.figure(figsize=(round(30 * figsize), round(10 * figsize)), dpi=80)
    plt.imshow(imgs_out)
    plt.show()
    return imgs_out


class SteelDataset(Dataset):

    def __init__(self, root_dir, df, nr_classes, transform=None,mask_transform = None, aug_transform = None):
        """
        Args:
            dataframe: a pandas dataframe.
            root_dir (string): image directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Returns:
            imgs,masks,labels
        """
        self.root_dir = root_dir
        self.df = df
        self.transform = transform
        self.nr_classes = nr_classes
        self.mask_transform = mask_transform
        self.aug_transform = aug_transform

    def _rle2mask(self, mask_rle: str, label=1, shape=(1600, 256)):
        """
        mask_rle: run-length as string formatted (start length)
        shape: (height,width) of array to return
        Returns numpy array, label - mask, 0 - background

        """
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = label
        return mask.reshape(shape).T  # Needed to align to RLE direction


    def _load_mask_one_hot(self, mask, nr_categories):
        """
        :param mask tensor, shape (height,width)

        :returns one hot mask, shape (nr_categories, height,width)
        """
        #assert mask.shape == (self.img_size[0], self.img_size[1]), "Error: mask.shape is {}".format(mask.shape)

        mask = mask.to(torch.int64)
        mask_one_hot = torch.nn.functional.one_hot(mask, nr_categories + 1).squeeze()

        mask_one_hot = mask_one_hot.permute([2, 0, 1]).unsqueeze(0)

        # mask shape: (1,256,1600)

        return mask_one_hot[:, 1:, :, :]

    def _load_img(self, img_path):

        img_pil = Image.open(img_path)
        return img_pil

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        image_pil = self._load_img(img_path)

        mask_rle_combined = np.zeros((256, 1600))
        label = torch.zeros((self.nr_classes))
        for category in range(1, self.nr_classes + 1):
            pxcode = self.df.iloc[idx, category]
            pxcode = False if pxcode == 'False' else pxcode
            if pxcode:
                mask_rle = self._rle2mask(pxcode, category)
                mask_rle_combined += mask_rle
                label[category - 1] = 1

        mask_pil = Image.fromarray(mask_rle_combined)
        mask_np = np.array(mask_pil)
        image_np = np.array(image_pil)
        if self.aug_transform:
            augmented = self.aug_transform(image=image_np, mask=mask_np)
        else:
            augmented = {'image':image_np,'mask':mask_np}

        image_augmented_pil = Image.fromarray(augmented['image'])
        mask_augmented_pil = Image.fromarray(augmented['mask'])


        mask_trafo = self.mask_transform(mask_augmented_pil).squeeze()

        mask_one_hot = self._load_mask_one_hot(mask_trafo, self.nr_classes)

        mask_out = mask_one_hot.squeeze()


        image_torch = self.transform(image_augmented_pil)


        return image_torch, mask_out, label

