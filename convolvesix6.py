import numpy as np
from astropy.io import fits
from PIL import Image
from tqdm import *
import os
from skimage import transform
from astropy.convolution import convolve


def makeDotpic(m, n, dot_num, pixels_value):
	# 构造点源图

	sz1 = m  # 尺寸
	sz2 = n
	img = np.zeros((sz1, sz2), np.float64)
	pos1 = np.random.randint(m, size=(dot_num, 1))  # 随机生成点数
	pos2 = np.random.randint(n, size=(dot_num, 1))
	for i in range(dot_num):
		img[pos1[i], pos2[i]] = np.random.randint(0, pixels_value)
	return img


def psfconv(raw_img, psf_path, over_sample):
	# 卷积六芒星PSF

	psfs = fits.open(psf_path)   # 读出PSFs
	psfs = psfs[0].data
	PSFs = []
	for j in range(psfs.shape[0]):                                                                  # 分视场
		psft = transform.resize(psfs[j], (psfs.shape[1]//over_sample, psfs.shape[2]//over_sample))  # PSF下采样回原来的采样率
		psft = psft/np.sum(psft)
		PSFs.append(psft)                                                              # 归一化PSF

	normsum = np.sum(raw_img)                                  # 卷积前归一化图像
	norm_img = raw_img / normsum
	makepic = convolve(norm_img, PSFs[0])                      # 原图卷积
	makepic *= normsum

	print('PSF Process done!')
	return makepic



def Expandconv(raw_img, expand_path):

	# 模拟生成拓展目标

	extends = []
	for dirpath, dirnames, filesnames in os.walk(expand_path):
		for i in filesnames:
			print(os.path.join(dirpath, i))
			l = Image.open(os.path.join(dirpath, i))
			l = np.array(l)
			extends.append(l[:, :, 0])

	Extends = []
	for j in range(len(extends)):  # 上采样为原图大小两倍，并且归一化
		extendt = transform.resize(extends[j], (raw_img.shape[0] * 2+1, raw_img.shape[0] * 2+1))  # PSF上采样到图像的2倍
		# extendt = transform.resize(extends[j], (extends[j].shape[0] + 1, extends[j].shape[0] + 1))
		extendt = extendt / np.sum(extendt)
		Extends.append(extendt)
	#
	normsum = np.sum(raw_img)  # 卷积前归一化图像
	norm_img = raw_img / normsum
	makepic = convolve(norm_img, Extends[np.random.randint(0, len(Extends))])  # 拓展卷积每张图选择一个卷积
	makepic *= normsum

	return makepic

def CCD_Poisson(imgori, imgsix, output):

	# 生成数据对，模拟CCD成像光子的泊松分布

	seed = np.random.randint(0, 2**16)                      # 统一分布种子
	np.random.seed(seed)
	possion_img = np.random.poisson(imgori)
	np.random.seed(seed)
	ori_poisson = np.random.poisson(imgsix)
	final_img = np.hstack((ori_poisson, possion_img))        # 拼接一组数据对
	a = fits.PrimaryHDU(final_img)						     # 写成fits对
	ah = fits.HDUList([a])
	ah.writeto(output + str(seed)+'_'+str(imgori.shape) + ".fits")
	print('Process done!')


if __name__ == '__main__':

	over_sample = 2                                             # 采样率
	# npsf        = 4//2   									    # psf个数，视场分割对应多少个
	output = './result/'  									    # 输出路径
	six_psf = './nircam_nrca5_f335m_fovp1001_samp2_npsf4.fits'  # 六芒星点扩散函数路径
	expand_psf = './expand_psf/'                			    # 拓展目标路径

	c2 = makeDotpic(1024, 1024, 20, 2**16)
	d1 = psfconv(c2, six_psf, over_sample)
	
	for z in tqdm(range(1, 5)):
		c = makeDotpic(1024, 1024, 5, 2**10)
		d = Expandconv(c, expand_psf)
		d1 += d
		c2 += d
	CCD_Poisson(d1, c2, output)



