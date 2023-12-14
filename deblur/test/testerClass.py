import setup_path

import os,sys,math
import argparse
import torch
import cv2 as cv
# aux = os.getcwd()
# dir_name = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(dir_name,'../dataset/'))
# sys.path.append(os.path.join(dir_name,'..'))

from deblur.dataset.dataset_motiondeblur import *
from deblur.dataset import together
import deblur.utils as utils

from skimage import img_as_ubyte


class TesterClass:
    def expand2square(self, timg,factor=16.0):
        _, _, h, w = timg.size()

        X = int(math.ceil(max(h,w)/float(factor))*factor)

        img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
        mask = torch.zeros(1,1,X,X).type_as(timg)

        img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
        mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)

        return img, mask
    
    def join_image(self, images):
        if len(images) == 1:
            return images[0]
        return together.unir_imagens(images)


    def teste_gopro_hide_r(self, imagePath=None, model='deblur'):
        parser = argparse.ArgumentParser(description='Image motion deblurring evaluation on GoPro/HIDE')
        parser.add_argument('--input_dir', default='',
            type=str, help='Directory of validation images')
        parser.add_argument('--result_dir', default='',
            type=str, help='Directory for results')
        if model == 'deblur':
            parser.add_argument('--weights', default='model_zoo/Uformer_B_gopro.pth',
                type=str, help='Path to weights')
        else:
            parser.add_argument('--weights', default='model_zoo/Uformer_B_sidd.pth',
                type=str, help='Path to weights')
        parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
        parser.add_argument('--arch', default='Uformer_B', type=str, help='arch')
        parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
        parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
        parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
        parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
        parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
        parser.add_argument('--query_embed', action='store_true', default=False, help='query embedding for the decoder')
        parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

        # args for vit
        parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
        parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
        parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
        parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
        parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
        parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
        parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
        parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
        args = parser.parse_args()


        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

        test_dataset = get_validation_deblur_data(args.input_dir, imagePath=imagePath)

        model_restoration= utils.get_arch(args)

        utils.load_checkpoint(model_restoration,args.weights)
        print("===>Testing using weights: ", args.weights)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model_restoration.to(device)
        model_restoration.eval()

        with torch.no_grad():
            images = test_dataset.__getimg__(0)
            processed = []
            for data_test_aux in images:
                _, h, w= data_test_aux.shape
                image = data_test_aux.unsqueeze(0)
                rgb_noisy, mask = self.expand2square(image.to(device), factor=128) 

                rgb_restored = model_restoration(rgb_noisy)
                rgb_restored = torch.masked_select(rgb_restored,mask.bool()).reshape(1,3,h,w)
                rgb_restored = torch.clamp(rgb_restored,0,1).cpu().numpy().squeeze().transpose((1,2,0))
                rgb_restored *= 255
                rgb_restored = rgb_restored.astype(np.uint8)
                processed.append(img_as_ubyte(rgb_restored))

        imagem_final = self.join_image(processed)
        imagem_final = self.refinar_linhas(imagem_final)
        imagem_final = imagem_final[:, :, ::-1]
        return imagem_final

    def refinar_linhas(self, image):
        num_horizontal_patches = 2
        num_vertical_patches = 2
        adjust = 0
        patch_size = image.shape[0] // num_horizontal_patches
        blur_radius = 3
        kernel_size = 3

        for i in range(1, num_horizontal_patches):
            x = i * patch_size
            line = image[:, (x - blur_radius):(x + blur_radius-adjust), :]
            for i in range(1, blur_radius):
                line[:, i, :] = line[:, 0, :]
                line[:, -i, :] = line[:, -1, :]
            # image[:, (x - blur_radius):(x + blur_radius), :] = cv.medianBlur(line, kernel_size)
            image[:, (x - blur_radius):(x + blur_radius-adjust), :] = cv.GaussianBlur(line, (kernel_size, kernel_size), 0)
            image[:, (x - blur_radius):(x + blur_radius-adjust), :] = line
        
        for i in range(1, num_vertical_patches):
            y = i * patch_size
            line = image[(y - blur_radius):(y + blur_radius-adjust), :, :]
            for i in range(1, blur_radius):
                line[i, :, :] = line[0, :, :]
                line[-i, :, :] = line[-1, :, :]
            # image[(y - blur_radius):(y + blur_radius), :, :] = cv.medianBlur(line, kernel_size)
            image[(y - blur_radius):(y + blur_radius-adjust), :, :] = cv.GaussianBlur(line, (kernel_size, kernel_size), 0)
            image[(y - blur_radius):(y + blur_radius-adjust), :, :] = line

        return image

def chamar_deblur(imagePath):
    tester = TesterClass()
    return tester.teste_gopro_hide_r(imagePath=imagePath, model='deblur')

def chamar_denoiser(imagePath):
    tester = TesterClass()
    return tester.teste_gopro_hide_r(imagePath=imagePath, model='denoiser')

#a = chamar_deblur("C:\\Users\\Danilo\\code\\Uformer-main\\dataset\\deblurring\\test\\input\\4_XIAOMI-PROCOFONE-F1_F.jpg")
#a = chamar_deblur("C:\\Users\\Danilo\\code\\Uformer-main\\dataset\\deblurring\\test_gopro\\input\\GOPR0384_11_00-000001.png")
#cv.imwrite('a.png', cv.cvtColor(a, cv.COLOR_RGB2BGR))