from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import torch
from torchvision import transforms

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR

def parse_args():
    parser = argparse.ArgumentParser(description='Simple testing function for Monodepthv2 models.')
    parser.add_argument('--image_path', type=str, help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str, help='name of a pretrained model to use', choices=[
        "mono_640x192", "stereo_640x192", "mono+stereo_640x192", "mono_no_pt_640x192", "stereo_no_pt_640x192",
        "mono+stereo_no_pt_640x192", "mono_1024x320", "stereo_1024x320", "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str, help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda", help='if set, disables CUDA', action='store_true')
    parser.add_argument("--pred_metric_depth", help='if set, predicts metric depth instead of disparity.', action='store_true')
    return parser.parse_args()

def test_simple(args):
    assert args.model_name is not None, "You must specify the --model_name parameter"
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    if os.path.isfile(args.image_path):
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    if not paths:
        print("No images found in the specified directory.")
        sys.exit()

    print("-> Predicting on {:d} test images".format(len(paths)))

    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            if image_path.endswith("_disp.jpg"):
                continue

            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((640, 192), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (original_height, original_width), mode="bilinear", align_corners=False)

            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

            if args.pred_metric_depth:
                name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
            else:
                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))

            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            normalized_disp = (disp_resized_np - disp_resized_np.min()) / (disp_resized_np.max() - disp_resized_np.min())
            normalized_disp = (normalized_disp * 255).astype(np.uint8)
            im = pil.fromarray(normalized_disp, 'L')
            
            name_dest_im = os.path.join(output_directory, "{}_depth.png".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))

    print('-> Done!')

if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
