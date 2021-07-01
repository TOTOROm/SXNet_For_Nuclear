import os
import json
from argparse import ArgumentParser
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from pathlib import Path

from utils.data import dataset_img2img
from utils.metric import batch_PSNR, ssim
from utils.choices import choose_loss, choose_model


# source activate py36, tmux split-window 9689

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--mode", choices=['n2c', 'n2n'], default='n2c')  # noise2noise or noise2clean
    parser.add_argument("--model_name", type=str, default='BRDNet')
    parser.add_argument("--n_loss", type=str, default='mse')  # nlu noise type
    parser.add_argument("--t_loss", type=str, default='mse')  # tlu noise type

    parser.add_argument("--eval", type=float, default=True)
    parser.add_argument("--nw", type=int, default=0)  # num of workers
    parser.add_argument("--bz", type=int, default=64)  # batch size
    parser.add_argument("--ep", type=int, default=120)  # epochs
    parser.add_argument("--lr", type=float, default=1e-3)  # initial learning rate
    parser.add_argument("--data_root", type=str, default='/home/ipsg/code/sx/datasets/infread/raws/cover_low')
    parser.add_argument("--noise_name", type=str, default='infread')
    # parser.add_argument("--train_set", type=str, default='')
    # parser.add_argument("--val_set", type=str, default='')
    return parser.parse_args()


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # 保存路径
    model_name = args.model_name + '-bz' + str(args.bz) + '_ep' + str(args.ep) + '_' + args.n_loss
    if '_t' in model_name:
        model_name = model_name + '_' + args.t_loss
    if args.mode == 'n2n':
        model_name = model_name + '-' + args.mode
    data_root_name = Path(args.data_root).name

    save_dir = 'results/' + data_root_name + '/' + model_name
    print('save dir:', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'train_args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        print(vars(args))

    # 数据准备
    source_h5_path_train = args.data_root + '/' + args.mode + '_' + args.noise_name + '_noised_train.h5'
    target_h5_path_train = source_h5_path_train.replace('noised', 'clean')
    source_h5_path_test = args.data_root + '/' + args.mode + '_' + args.noise_name + '_noised_test.h5'
    target_h5_path_test = source_h5_path_test.replace('noised', 'clean')

    dataset_train = dataset_img2img(source_h5_path_train, target_h5_path_train)
    dataset_test = dataset_img2img(source_h5_path_test, target_h5_path_test)
    loader_train = DataLoader(dataset=dataset_train, num_workers=args.nw, batch_size=args.bz, shuffle=True)
    loader_test = DataLoader(dataset=dataset_test, num_workers=0, batch_size=1, shuffle=False)
    train_steps, test_num = len(loader_train), len(loader_test)
    print('train_num:', len(dataset_train), 'test_num:', test_num, 'train_steps:', train_steps)

    # 训练准备
    model = choose_model(args.model_name, 'train').cuda()
    criterion_nlu = choose_loss(args.n_loss)
    criterion_tlu = choose_loss(args.n_loss)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 开始训练
    with_tlu = ('_t' in args.model_name)  # whether using tlu
    writer = SummaryWriter(save_dir)
    log_dicts = []
    for epoch in range(args.ep):
        ep = epoch + 1
        model.train()
        ep_loss = 0
        ep_logs = {}
        for i, batch in enumerate(loader_train):
            batch_datas = batch[0].cuda()
            batch_labels = batch[1].cuda()
            fw_nlu, fw_tlu = model(batch_datas)
            n_loss = criterion_nlu(fw_nlu, batch_labels).cuda()
            if with_tlu:
                t_loss = criterion_tlu(fw_tlu, batch_labels).cuda()
                loss = n_loss + 0.1 * t_loss
            else:
                loss = n_loss
            if i % 10 == 0:
                print("[Epoch %d][%d/%d] loss: %.4f " % (ep, i + 1, train_steps, loss.item()))
            ep_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ep_loss /= train_steps
        writer.add_scalar('loss', ep_loss, ep)
        save_pt_dir = args.data_root + '/pts'
        if not os.path.exists(save_pt_dir):
            os.makedirs(save_pt_dir)
        torch.save(model.state_dict(), os.path.join(save_pt_dir, str(ep) + '.pth'))
        ep_logs.setdefault('epoch', ep)
        ep_logs.setdefault('loss', ep_loss)

        # 每轮测试
        if args.eval:
            ep_psnr_n, ep_ssim_n, ep_psnr_t, ep_ssim_t = 0, 0, 0, 0
            model.eval()
            for i, batch in enumerate(loader_test):
                with torch.no_grad():
                    batch_datas = batch[0].cuda()
                    batch_labels = batch[1].cuda()
                    fw_nlu, fw_tlu = model(batch_datas)
                    psnr_n = batch_PSNR(fw_nlu, batch_labels, data_range=1.0)
                    ssim_n = ssim(fw_nlu, batch_labels, data_range=1.0, win_size=11).item()
                    ep_psnr_n += psnr_n
                    ep_ssim_n += ssim_n
                    if with_tlu:
                        psnr_t = batch_PSNR(fw_tlu, batch_labels, data_range=1.0)
                        ssim_t = ssim(fw_tlu, batch_labels, data_range=1.0, win_size=11).item()
                        ep_psnr_t += psnr_t
                        ep_ssim_t += ssim_t
            ep_psnr_n /= test_num
            ep_ssim_n /= test_num
            ep_logs.setdefault('psnr_n', ep_psnr_n)
            ep_logs.setdefault('ssim_n', ep_ssim_n)
            writer.add_scalar('psnr_n', ep_psnr_n, ep)
            writer.add_scalar('ssmi_n', ep_ssim_n, ep)
            if with_tlu:
                ep_psnr_t /= test_num
                ep_ssim_t /= test_num
                ep_logs.setdefault('psnr_t', ep_psnr_t)
                ep_logs.setdefault('ssim_t', ep_ssim_t)
                writer.add_scalar('psnr_t', ep_psnr_t, ep)
                writer.add_scalar('ssmi_t', ep_ssim_t, ep)
            log_dicts.append(ep_logs)
            print(ep_logs)

    # 保存所有记录
    writer.close()
    with open(os.path.join(save_dir, 'logs.json'), 'w') as f:
        json.dump(log_dicts, f, indent=2)


if __name__ == "__main__":
    args = get_args()

    search = False

    if search:
        modes = ['n2n', 'n2c']
        model_names = ['SXNet_8_16_a_tlu', 'SXNet_8_16_a', 'ADNET', 'BRDNET', 'DNCNN']
        # losses = ['mse', 'mse']
        # noise_names = ['gaussian_50']
        if args.gpu == 0:
            args.mode = modes[0]
            for model_name in model_names:
                args.model_name = model_name
                train(args)
        elif args.gpu == 1:
            args.mode = modes[1]
            for model_name in model_names:
                args.model_name = model_name
                train(args)
    else:
        train(args)
