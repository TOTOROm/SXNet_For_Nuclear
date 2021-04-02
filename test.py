import os
from argparse import ArgumentParser
import cv2
import json
import time
from utils.metric import *
from data import dataset_img2img
from pathlib import Path
from train import choose_model
from ptflops import get_model_complexity_info
from matplotlib import pyplot as plt


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--save_dir", type=str, default='results2/infreadEN/DnCNN-bz256_ep120_mse')
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--test_ep", type=int, default=120)
    parser.add_argument("--save_images", type=bool, default=True)
    parser.add_argument("--show_images", type=bool, default=True)

    parser.add_argument("--test_set", type=str, default='')
    parser.add_argument("--h5_dir", type=str, default='infread')
    parser.add_argument("--model_name", type=str, default='')
    parser.add_argument("--noise_name", type=str, default='')
    parser.add_argument("--n_loss", type=str, default='')
    parser.add_argument("--mode", choices=['n2c', 'n2n'], default='')
    parser.add_argument("--t_loss", type=str, default='')
    parser.add_argument("--bz", type=int, default=0)
    parser.add_argument("--ep", type=int, default=0)
    parser.add_argument("--data_root", type=str, default='')
    return parser.parse_args()


def merge_args_from_train(save_dir, args):
    train_args_path = save_dir + '/train_args.json'
    if os.path.exists(train_args_path):
        print('merging args from:', train_args_path)
        with open(train_args_path, 'r') as f:
            train_args_dict = json.load(f)
        for k, v in train_args_dict.items():
            if k in args.__dict__ and k != 'gpu':
                print('change test_args: [{}] to: {}'.format(k, v))
                args.__dict__[k] = v
    else:
        print("No 'train_args.json', so did not merge args from train_args.")
    return args


def test(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args = merge_args_from_train(args.save_dir, args)
    model_name = args.model_name + '-bz' + str(args.bz) + '_ep' + str(args.ep) + '_' + args.n_loss
    if '_t' in model_name:
        model_name = model_name + '_' + args.t_loss
    if args.mode == 'n2n':
        model_name = model_name + '-' + args.mode
    if not os.path.exists(args.save_dir):
        args.save_dir = 'results2' + args.h5_dir.replace('/', '_') + '/' + args.noise_name + '/' + model_name
    print('existing save_dir:', args.save_dir)

    # source_h5_path_test = args.data_root + '/' + args.h5_dir + '/' + args.mode + '_' + args.noise_name + '_' + args.test_set + '_noised_test.h5'
    # target_h5_path_test = args.data_root + '/' + args.h5_dir + '/' + args.mode + '_' + args.noise_name + '_' + args.test_set + '_clean_test.h5'
    source_h5_path_test = '/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN_noised_test.h5'
    target_h5_path_test = '/home/ipsg/code/sx/datasets/infread/images/n2c_infreadEN_clean_test.h5'
    test_set = dataset_img2img(source_h5_path_test, target_h5_path_test)
    print('source_h5_path_test:', source_h5_path_test)

    model_path = args.save_dir + '/' + str(args.test_ep) + '.pth'
    state_dict = torch.load(model_path)
    net = choose_model(args.model_name, 'test')
    print('loading:', model_path)

    net.load_state_dict(state_dict)
    net.eval()
    net.cuda()

    with torch.no_grad():
        f, p = get_model_complexity_info(net, (3, 480, 640), as_strings=True, print_per_layer_stat=False, verbose=False)
    print('FLOPs:', f, 'Parms:', p)

    test_psnr = 0
    test_ssim = 0
    fw_times = []
    for i, pair in enumerate(test_set):
        with torch.no_grad():
            batch_datas = pair[0].unsqueeze(0).cuda()
            batch_labels = pair[1].unsqueeze(0).cuda()
            fw_s = time.clock()
            batch_inferences = net(batch_datas)
            fw_time = time.clock() - fw_s
            fps = np.round(1 / fw_time, 3)
            fw_times.append(fw_time)
            # print(batch_datas.shape, batch_labels.shape, batch_inferences.shape)
            psnr_batch = batch_PSNR(batch_inferences, batch_labels, data_range=1.0)
            test_psnr += psnr_batch
            ssim_batch = ssim(batch_inferences, batch_labels, data_range=1.0, win_size=11).item()
            test_ssim += ssim_batch
            print('image:{}, fps:{}, psnr:{}, ssim:{}'.format(i, fps, psnr_batch, ssim_batch))

            inference = np.array(batch_inferences.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
            source = np.array(batch_datas.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
            target = np.array(batch_labels.cpu().squeeze(0).permute(1, 2, 0) * 255).astype('uint8')
            result = cv2.hconcat([source, inference, target])

            if args.save_images:
                save_images_dir = args.save_dir + '/test_images_from_' + args.h5_dir.replace('/', '_') + '_ep' + str(args.test_ep)
                if not os.path.exists(save_images_dir):
                    os.makedirs(save_images_dir)
                cv2.imwrite(save_images_dir + '/' + args.noise_name + '_clean' + str(i) + '.jpg', target,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cv2.imwrite(save_images_dir + '/' + args.noise_name + '_noised' + str(i) + '.jpg', inference,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cv2.imwrite(save_images_dir + '/' + args.noise_name + '_concat' + str(i) + '.jpg', result,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            if args.show_images:
                plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                plt.pause(0.2)

    fw_fps = 1 / np.mean(fw_times)
    test_psnr /= len(test_set)
    test_ssim /= len(test_set)
    print('fw_fps:{}, psnr:{}, ssim:{}'.format(fw_fps, test_psnr, test_ssim))

    test_info = vars(args)
    test_info.setdefault('fw_fps', fw_fps)
    test_info.setdefault('psnr', test_psnr)
    test_info.setdefault('ssim', test_ssim)
    with open(os.path.join(Path(model_path).parent, 'test_info.json'), 'w') as f:
        json.dump(test_info, f, indent=2)


if __name__ == "__main__":
    args = get_args()
    search = False
    if search:
        root = 'results_pristine_bsd_kodak_mc/gaussian_50'
        model_dirs = [str(i) for i in Path(root).iterdir() if i.is_dir()]
        for model_dir in model_dirs:
            args.save_dir = model_dir
            test(args)
    else:
        test(args)
