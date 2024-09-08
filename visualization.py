import imgvision as iv
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
#导入高光谱图像
HSI=scipy.io.loadmat('/home/zhangmj/hyperspectralSR/CEGATSR/mcodes/dataset/Cave_x2/evals/block_balloons_ms_1.mat')
HSI2=(HSI['gt'])
# HSI = np.load('/home/zhangmj/hyperspectralSR/CEGATSR/datasets/Chikusei_x2/test/Chikusei_test.mat',allow_pickle=True)
print(HSI2.shape)
#(512,512,31)  该光谱图像是 空间维度512×512，光谱维度31（400nm~700nm 10nm间隔）

#光谱图像的RGB显示
	#创建转换器
convertor = iv.spectra()
	#RGB图像转换
RGB = convertor.space(HSI2)
	# RGB图像显示
plt.imshow(RGB)
plt.show()