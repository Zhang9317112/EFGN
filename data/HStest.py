import numpy as np
import torch.utils.data as data
import scipy.io as sio
import torch
import os
import utils


def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

class HSTestData(data.Dataset):
    def __init__(self, image_dir, use_3D=False):
        self.image_folders = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        self.image_files = []
        for i in self.image_folders:
            if os.path.isdir(i):
                images = os.listdir(i)
                for j in images:
                    if is_mat_file(j):
                        full_path = os.path.join(i, j)
                        self.image_files.append(full_path)
            else:
                if is_mat_file(i):
                    self.image_files.append(i)

        self.use_3Dconv = use_3D

    def __getitem__(self, index):
        file_index = index
        load_dir = self.image_files[file_index]
        # print(load_dir)
        data = sio.loadmat(load_dir)
        # print(data)
        lms = np.array(data['ms_bicubic'][...], dtype=np.float32)
        ms = np.array(data['ms'][...], dtype=np.float32)
        gt = np.array(data['gt'][...], dtype=np.float32)




        if self.use_3Dconv:
            ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
            ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
            lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
            gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
        else:
            ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
            lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
            gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        #ms = torch.from_numpy(ms.transpose((2, 0, 1)))
        #lms = torch.from_numpy(lms.transpose((2, 0, 1)))
        #gt = torch.from_numpy(gt.transpose((2, 0, 1)))
        # print("ms2=",ms,"ms2=",lms,"ms2=",gt)
        return ms, lms, gt

    def __len__(self):
        return len(self.image_files)

# class HSTestData(data.Dataset):
#     def __init__(self, image_dir, use_3D=False):
#         test_data = sio.loadmat(image_dir)
#         self.use_3Dconv = use_3D
#         self.ms = np.array(test_data['ms'][...], dtype=np.float32)
#         self.lms = np.array(test_data['ms_bicubic'][...], dtype=np.float32)
#         self.gt = np.array(test_data['gt'][...], dtype=np.float32)
#
#     def __getitem__(self, index):
#         gt = self.gt[index, :, :, :]
#         ms = self.ms[index, :, :, :]
#         lms = self.lms[index, :, :, :]
#         if self.use_3Dconv:
#             ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
#             ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
#             lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
#             gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
#         else:
#             ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
#             lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
#             gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
#         #ms = torch.from_numpy(ms.transpose((2, 0, 1)))
#         #lms = torch.from_numpy(lms.transpose((2, 0, 1)))
#         #gt = torch.from_numpy(gt.transpose((2, 0, 1)))
#         return ms, lms, gt
#
#     def __len__(self):
#         return self.gt.shape[0]