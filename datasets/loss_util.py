import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from models.PCToImg import project_point_cloud_to_image


class Reconstruction_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.R = np.load('E:\\Cui\\MANet-master\\data\\R.npy')
        self.T = np.load('E:\\Cui\\MANet-master\\data\\T.npy')
        self.K = np.array([[500, 0, 320],
                           [0, 500, 240],
                           [0, 0, 1]], dtype=np.float32)

    def forward(self, output_pc: torch.Tensor, target: torch.Tensor):
        """
        运行逻辑：对于输入图像进行三维到二维映射，并计算二维映射与原图像的损失
        :param output_pc: an [N x C x T] tensor, 点云数据.
        :param target: an [N X 3 X W X H] tensor, 原始图像(如果使用transform进行处理的话应当进行相应还原).
        :return: an [N] tensor.损失
        """
        # 将点云转换为多视角图像
        output_pc = output_pc.detach().cpu().numpy()
        output_ims = []
        for i, pc in enumerate(output_pc):
            output_im = project_point_cloud_to_image(point_cloud=pc, num_views=target.shape[1],
                                                     image_shape=[1300, 1300], R=self.R, T=self.T,
                                                     K=self.K)
            output_ims.append(output_im)
        output_ims = np.stack(output_ims, axis=0)
        output_ims = np.reshape(output_ims, target.size())
        output_ims = torch.from_numpy(output_ims)
        output_ims = output_ims.to(torch.float32)
        loss = Reconstruction_loss.loss_cal(output_ims, target)
        return loss

    @staticmethod
    def loss_cal(recon_image: torch.Tensor, target_image: torch.Tensor):
        """
        计算二维图像间损失
        :param recon_image: an [N X 3 X W X H] tensor,重构图像
        :param target_image: an [N X 3 X W X H] tensor,目标图像
        :return: an [N] tensor.损失
        """
        if recon_image.shape != target_image.shape:
            raise ValueError("Both tensors must have the same shape")

        recon_image = recon_image.to(target_image.device)

        mse = nn.MSELoss()
        loss = mse(recon_image, target_image)

        return loss
