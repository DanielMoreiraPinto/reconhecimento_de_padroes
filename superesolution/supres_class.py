import setup_path

import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch
import requests
# print(os.getcwd())
from superesolution.models.network_swin2sr import Swin2SR as net
from superesolution.utils import util_calculate_psnr_ssim as util

# os.chdir('superesolution/swin2sr')

class SuperResolution:
    def main(self, img_path):
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', type=str, default='compressed_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                        'gray_dn, color_dn, jpeg_car, color_jpeg_car')
        parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
        parser.add_argument('--model_path', type=str,
                            default='model_zoo/Swin2SR_CompressedSR_X4_48.pth')
        parser.add_argument('--training_patch_size', type=int, default=48, help='patch size used in training Swin2SR. '
                                        'Just used to differentiate two different settings in Table 2 of the paper. '
                                        'Images are NOT tested patch by patch.')
        parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
        parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
        parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
        parser.add_argument('--folder_lq', type=str, default='inputs/', help='input low-quality test image folder')
        parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
        parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
        parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
        parser.add_argument('--save_img_only', default=True, action='store_true', help='save image and do not evaluate')
        args = parser.parse_args()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'loading model from {args.model_path}')

        model = self.define_model(args)
        model.eval()
        model = model.to(device)

        # setup folder and path
        folder, save_dir, border, window_size = self.setup(args)
        # os.makedirs(save_dir, exist_ok=True)

        # read image
        imgname, img_lq, _ = self.get_image_pair(args, img_path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = self.test(img_lq, model, args, window_size)
            
            if args.task == 'compressed_sr':
                output = output[0][..., :h_old * args.scale, :w_old * args.scale]
            else:
                output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        # cv2.imwrite(f'{save_dir}/{imgname}_Swin2SR.png', output)
        return output

    def define_model(self, args):
        # 001 classical image sr
        if args.task == 'classical_sr':
            model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
            param_key_g = 'params'

        # 002 lightweight image sr
        # use 'pixelshuffledirect' to save parameters
        elif args.task in ['lightweight_sr']:
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
            param_key_g = 'params'
            
        elif args.task == 'compressed_sr':
            model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle_aux', resi_connection='1conv')
            param_key_g = 'params'                

        # 003 real-world image sr
        elif args.task == 'real_sr':
            if not args.large_model:
                # use 'nearest+conv' to avoid block artifacts
                model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                            mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
            else:
                # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
                model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                            num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                            mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
            param_key_g = 'params_ema'

        # 006 grayscale JPEG compression artifact reduction
        # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
        elif args.task == 'jpeg_car':
            model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                        img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='', resi_connection='1conv')
            param_key_g = 'params'

        # 006 color JPEG compression artifact reduction
        # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
        elif args.task == 'color_jpeg_car':
            model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                        img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='', resi_connection='1conv')
            param_key_g = 'params'

        pretrained_model = torch.load(args.model_path)
        model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

        return model


    def setup(self, args):
        # 001 classical image sr/ 002 lightweight image sr
        if args.task in ['classical_sr', 'lightweight_sr', 'compressed_sr']:
            save_dir = f'results/swin2sr_{args.task}_x{args.scale}'
            folder = args.folder_lq
            border = args.scale
            window_size = 8

        # 003 real-world image sr
        elif args.task in ['real_sr']:
            save_dir = f'results/swin2sr_{args.task}_x{args.scale}'
            if args.large_model:
                save_dir += '_large'
            folder = args.folder_lq
            border = 0
            window_size = 8

        # 006 JPEG compression artifact reduction
        elif args.task in ['jpeg_car', 'color_jpeg_car']:
            save_dir = f'results/swin2sr_{args.task}_jpeg{args.jpeg}'
            folder = args.folder_gt
            border = 0
            window_size = 7

        return folder, save_dir, border, window_size


    def get_image_pair(self, args, path):
        (imgname, imgext) = os.path.splitext(os.path.basename(path))

        # 001 classical image sr/ 002 lightweight image sr (load lq-gt image pairs)
        if args.task in ['classical_sr', 'lightweight_sr']:
            if args.save_img_only:
                img_gt = None
                img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.            
            else:
                img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
                img_lq = cv2.imread(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(
                    np.float32) / 255.            
            
        elif args.task in ['compressed_sr']:
            if args.save_img_only:
                img_gt = None
                img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.            
            else:
                img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
                img_lq = cv2.imread(f'{args.folder_lq}/{imgname}.jpg', cv2.IMREAD_COLOR).astype(
                    np.float32) / 255.        

        # 003 real-world image sr (load lq image only)
        elif args.task in ['real_sr', 'lightweight_sr_infer']:
            img_gt = None
            img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        # 006 grayscale JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
        elif args.task in ['jpeg_car']:
            img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img_gt.ndim != 2:
                img_gt = util.bgr2ycbcr(img_gt, y_only=True)
            result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
            img_lq = cv2.imdecode(encimg, 0)
            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

        # 006 JPEG compression artifact reduction (load gt image and generate lq image on-the-fly)
        elif args.task in ['color_jpeg_car']:
            img_gt = cv2.imread(path)
            result, encimg = cv2.imencode('.jpg', img_gt, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg])
            img_lq = cv2.imdecode(encimg, 1)
            img_gt = img_gt.astype(np.float32)/ 255.
            img_lq = img_lq.astype(np.float32)/ 255.

        return imgname, img_lq, img_gt


    def test(self, img_lq, model, args, window_size):
        if args.tile is None:
            # test the image as a whole
            output = model(img_lq)
        else:
            # test the image tile by tile
            b, c, h, w = img_lq.size()
            tile = min(args.tile, h, w)
            assert tile % window_size == 0, "tile size should be a multiple of window_size"
            tile_overlap = args.tile_overlap
            sf = args.scale

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
            output = E.div_(W)

        return output

def aumentar_resolucao(img_path):
    sr = SuperResolution()
    img = sr.main(img_path)
    return img


# img = aumentar_resolucao('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\superesolution\\inputs\\shanghai.jpg')
# cv2.imwrite('D:\\daniel_moreira\\reconhecimento_de_padroes\\reconhecimento_de_padroes\\superesolution\\results\\shanghai.jpg', img)