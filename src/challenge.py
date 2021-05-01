import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.vision import *
from fastai.callbacks import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
import glob
import matplotlib.pyplot as plt
import nibabel as nib

import scipy.ndimage.measurements as meas
import scipy.ndimage as ndimage
import scipy.ndimage.morphology as mp
import skimage.morphology as mpg

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

def dice(input, target):
    input = F.softmax(input, dim=1)
    target = torch.squeeze(target,dim=1)

    eps = 0.0001
    encoded_target = input.detach() * 0
    encoded_target = encoded_target.scatter_(1, target.unsqueeze(1), 1)

    intersection = input * encoded_target
    numerator = 2 * intersection.sum(0).sum(1).sum(1)

    denominator = input + encoded_target
    denominator = denominator.sum(0).sum(1).sum(1) + eps
    loss_per_channel = numerator / denominator

    return loss_per_channel.sum() / input.size(1)
class DiceLoss(nn.Module):
    """
    Dice Loss for a batch of samples
    """
    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        """
        Forward pass
        :param output: NxCxHxW logits
        :param target: NxHxW LongTensor
        :return: torch.tensor
        """
        output = F.softmax(output, dim=1)
        return self._dice_loss_multichannel(output,target)

    @staticmethod
    def _dice_loss_multichannel(output, target):
        """
        Forward pass
        :param output: NxCxHxW Variable
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :param binary: bool for binarized one chaneel(C=1) input
        :return:
        """
        target = torch.squeeze(target)
        eps = 0.0001
        encoded_target = output.detach() * 0

        encoded_target = encoded_target.scatter_(1, target.unsqueeze(1), 1)

        weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)

class CrossEntropyLoss2d(nn.Module):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """
    def __init__(self):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        """
        Forward pass

        :param inputs: torch.tensor (NxC)
        :param targets: torch.tensor (N)
        :return: scalar
        """
        targets = torch.squeeze(targets)
        return self.nll_loss(inputs, targets)
  
class CombinedLoss(nn.Module):
    """
    A combination of dice  and cross entropy loss
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()

    def forward(self, input, target, weight=True):
        """
        Forward pass

        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """
        weight = self.rtn_weight(torch.squeeze(target))

        # input_soft = F.softmax(input, dim=1)
        y_2 = torch.mean(self.dice_loss(input, target))
        if weight is True:
            y_1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        else:
            y_1 = torch.mean(
                torch.mul(self.cross_entropy_loss.forward(input, target), weight))
        return y_1 + y_2

    def rtn_weight(self, labels):
        labels = labels.cpu().numpy()
        class_weights = np.zeros_like(labels)

        grads = np.gradient(labels) 
        edge_weights = (grads[0] ** 2 + grads[1] ** 2 ) > 0 
        class_weights += 2 * edge_weights
        
        return torch.from_numpy(class_weights).to(0)

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def r_noise(x,use_on_y=True):
    x = random_noise(x)
    return torch.from_numpy(x).type(torch.FloatTensor)

seed = 42
random_seed(seed,True)

path = Path('./')
path.ls()
path_img = Path('./img(random)_10000_3ch')
path_lbl = Path('./gt(random)_10000_3ch')
fnames = get_image_files(path_img)
print(fnames[:3])
lbl_names = get_image_files(path_lbl)
print(lbl_names[:3])

print(len(fnames), len(lbl_names))
img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(5,5), cmap='gray')
get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'
mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5,5), alpha=1)

src_size = np.array(mask.shape[1:])
src_size,mask.data
codes = np.array(['Void', 'Fat', 'Muscle', 'Visceral_fat'], dtype=str); codes
from skimage.util import random_noise

rn = TfmPixel(r_noise)
tfms = get_transforms(flip_vert=True, max_rotate=180.0, max_zoom=1.5, max_warp = 0.2 )
new_tfms = (tfms[0] + [rn()], tfms[1])
new_tfms[0][7].use_on_y = False
new_tfms[0][7].p = 0.5
size = src_size

free = gpu_mem_get_free_no_cache()
# the max size of bs depends on the available GPU RAM
if free > 8200: bs=4
else:           bs=2
print(f"using bs={bs}, have {free}MB of GPU RAM free")

src = (SegmentationItemList.from_folder(path_img)
       .split_by_rand_pct(valid_pct=0.1)
       .label_from_func(get_y_fn, classes=codes))
data = (src.transform(new_tfms, size=size, tfm_y=True)
        .databunch(bs=bs, num_workers=0)
        .normalize(imagenet_stats))

a,b = next(iter(data.train_dl))
arr = a[0,0,:,:].cpu().numpy()
plt.imshow(arr,cmap='gray')

arr = b[0,0,:,:].cpu().numpy()
plt.imshow(arr)

"""
Model
"""

name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']


loss_func = CombinedLoss
metrics = [ dice,acc_camvid ]
wd = 1e-2

learn = unet_learner(data, models.resnet34, loss_func = loss_func(), metrics=metrics)
lr_find(learn)
learn.recorder.plot()
lr = 3e-4
learn.fit_one_cycle(10, lr)

"""
Result
"""
test_img = sorted(glob.glob('./test(random)_10000/*.png'))
n = 19
name = os.path.split(test_img[n])[-1][:-4]
I = plt.imread(test_img[n])
plt.imshow(I)
plt.title(f'{name}_img')
plt.show()
a,b,c = learn.predict(open_image(test_img[n]))
d = np.argmax(np.array(c),axis=0)
plt.imshow(d)
plt.title('prediction')
plt.show()

result_csv_name = 'aimlab_submit.csv'

with open(f'./{result_csv_name}.csv', 'w', encoding='utf-8', newline='') as file :
    wr = csv.writer(file)
    wr.writerow(['ImageId','EncodedPixels'])
    for path in test_img:
        a,b,c = learn.predict(open_image(path))
        d = np.argmax(c,axis=0)
        name = os.path.split(path)[-1][:-4]

        ### postprocessing
        ### label 분리
        gt1 = np.array(d).copy()
        gt2 = np.array(d).copy()
        gt3 = np.array(d).copy()

        all = np.array(d).copy()

        gt1[gt1!=1]=0
        gt2[gt2!=2]=0
        gt2[gt2==2]=1
        gt3[gt3!=3]=0
        gt3[gt3==3]=1

        all[all!=0]=1

        se2 = mpg.disk(20)
        gt_2 = mp.binary_closing(gt2,se2).astype('int')

        a = ndimage.binary_fill_holes(gt2+gt1+gt3).astype('int32') #3
        b = ndimage.binary_fill_holes(gt_2+gt1).astype('int32') #2
        c = ndimage.binary_fill_holes(gt1).astype('int32')#1

        c,num1 = meas.label(c)  
        unique1, counts1 = np.unique(c, return_counts=True)
        idxls = (-counts1).argsort()[:2]
        idx = idxls[1]
        #print(idx)
        c[c!=idx] = 0
        c[c==idx] = 1

        im1 = ((c-gt2)>0).astype('int') ##

        final2 = a+b+(im1)

        final2 = (final2 == 1).astype('int')*3+(final2 == 2).astype('int')*2+(final2 == 3).astype('int')

        plt.figure(dpi=150)
        plt.subplot(1,2,1)
        plt.imshow(d)
        plt.title(name+'_predict')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(final2)
        plt.axis('off')
        plt.title(name+'_postprocessing') 
        plt.show()   

        for idx in range(1,4):
            np_array = np.where(final2 == idx,1,0)
            result = rle_encode(np_array).astype(str)
            result = ' '.join(result)
            wr.writerow([f'{name}_{idx}',result])

