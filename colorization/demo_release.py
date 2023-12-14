
import argparse
import matplotlib.pyplot as plt

from colorization.colorizers import *

def colorize_image(path):
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--img_path', type=str, default=path)
	parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
	parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
	opt = parser.parse_args()

	# load colorizers
	colorizer_eccv16 = eccv16(pretrained=True).eval()
	colorizer_siggraph17 = siggraph17(pretrained=True).eval()
	if(opt.use_gpu):
		colorizer_eccv16.cuda()
		colorizer_siggraph17.cuda()

	# default size to process images is 256x256
	# grab L channel in both original ("orig") and resized ("rs") resolutions
	img = load_img(opt.img_path)
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
	out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
	out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

	'''
	plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
	plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)

	
	plt.figure(figsize=(12,8))
	plt.subplot(2,2,1)
	plt.imshow(img)
	plt.title('Original')
	plt.axis('off')

	plt.subplot(2,2,2)
	plt.imshow(img_bw)
	plt.title('Input')
	plt.axis('off')

	plt.subplot(2,2,3)
	plt.imshow(out_img_eccv16)
	plt.title('Output (ECCV 16)')
	plt.axis('off')

	plt.subplot(2,2,4)
	plt.imshow(out_img_siggraph17)
	plt.title('Output (SIGGRAPH 17)')
	plt.axis('off')
	plt.show()
	'''

	print("Imagem colorida")
	return tensor_to_image(out_img_eccv16)


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  # return PIL.Image.fromarray(tensor)
  return tensor
# out_img_eccv16 = colorize_image('C:\\Users\\joao_\\Documents\\Trabalho-ReconhecimentoPadroes\\datasets\\teste.jpg')
# print('tipo ', type(out_img_eccv16))