import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal
import random
from motion_blur.generate_PSF import PSF
from motion_blur.generate_trajectory import Trajectory
import torchvision.transforms.functional as F
# Here achieve add motion blur| add spot | low/high light| halo
import util.util as util

class DegradImage(object):

    def __init__(self, imgs, PSFs=None, part=None, path__to_save='/home/sz1/medical/code/', Colors=None, Spots=None, Halos=None):

        self.original = imgs
        # self.original = np.transpose(self.whc,(1,2,0))
        # self.image_path = image_path
        # self.original = misc.imread(self.image_path)
        self.shape =self.original.shape
        if len(self.shape) < 3:
            raise Exception('We support only RGB images yet.')
        elif self.shape[0] != self.shape[1]:
            raise Exception('We support only square images yet.')
        self.path_to_save = path__to_save

        """

        :param image_path: path to square, RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        :param Colors: Color jitter
        :param Spots: add spot jitter
        :param Halos: add halo 
        """
        # if os.path.isfile(image_path):
        #     self.image_path = image_path
        #     self.original = misc.imread(self.image_path)
        #     self.shape = self.original.shape
        #     if len(self.shape) < 3:
        #         raise Exception('We support only RGB images yet.')
        #     elif self.shape[0] != self.shape[1]:
        #         raise Exception('We support only square images yet.')
        # else:
        #     raise Exception('Not correct path to image.')
        # self.path_to_save = path__to_save

        ## motion blur
        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=self.shape[1]).fit()
            else:
                self.PSFs = PSF(canvas=self.shape[1], path_to_save=os.path.join(self.path_to_save,
                                                                                'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        self.part = part
        self.result = []

    def degrad_image(self, save=False, show=False):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        # yN, xN, channel = self.shape
        channel, yN,xN = self.shape
        key, kex = self.PSFs[0].shape

        delta = yN - key
        assert delta >= 0, 'resolution of image should be higher than kernel'
        result=[]
        if len(psf) > 1:
            for p in psf:
                tmp = np.pad(p, delta // 2, 'constant')
                cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                blured =cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
                blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
                blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
                blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
                # blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                # blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)

                result.append(np.abs(blured))
        else:
            psf = psf[0]
            tmp = np.pad(psf, delta // 2, 'constant')
            # cv2.normalize(tmp, tmp, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # blured = cv2.normalize(self.original, self.original, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
            #                        dtype=cv2.CV_32F)
            blured = self.original
            blured[:, :, 0] = np.array(signal.fftconvolve(blured[:, :, 0], tmp, 'same'))
            blured[:, :, 1] = np.array(signal.fftconvolve(blured[:, :, 1], tmp, 'same'))
            blured[:, :, 2] = np.array(signal.fftconvolve(blured[:, :, 2], tmp, 'same'))
            # blured = cv2.normalize(blured, blured, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
            # result.append(np.abs(blured))
        self.result = blured
        if show or save:
            self.__plot_canvas(show, save)
        return self.result

    def __plot_canvas(self, show, save):
        if len(self.result) == 0:
            raise Exception('Please run blur_image() method first.')
        else:
            plt.close()
            plt.axis('off')
            fig, axes = plt.subplots(1, len(self.result), figsize=(10, 10))
            if len(self.result) > 1:
                for i in range(len(self.result)):
                        axes[i].imshow(self.result[i])
            else:
                plt.axis('off')

                plt.imshow(self.result[0])
            if show and save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split('/')[-1]), self.result[0] * 255)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                result = util.numpy2im(self.result)
                util.save_image(result, '/home/sz1/medical/code/blured.jpg')
                psf_show=cv2.normalize(self.PSFs[0], self.PSFs[0], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                psf_show=np.array([psf_show,psf_show,psf_show])
                psf_show =util.numpy2im(psf_show)
                util.save_image(psf_show, '/home/sz1/medical/code/psf.jpg')
                # cv2.imwrite(os.path.join(self.path_to_save, self.image_path.split('/')[-1]), self.result[0] * 255)
            elif show:
                plt.show()


if __name__ == '__main__':
    folder = '/Users/mykolam/PycharmProjects/University/DeblurGAN2/results_sharp'
    folder_to_save = '/Users/mykolam/PycharmProjects/University/DeblurGAN2/blured'
    params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
    for path in os.listdir(folder):
        print(path)
        trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()
        psf = PSF(canvas=64, trajectory=trajectory).fit()
        BlurImage(os.path.join(folder, path), PSFs=psf,
                  path__to_save=folder_to_save, part=np.random.choice([1, 2, 3])).\
            blur_image(save=True)
