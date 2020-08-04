import os
import json
import h5py
import operator
import random
from argparse import ArgumentParser
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from models.SOTA import DNCNN, ADNET, BRDNET
from models.MY import *
from utils.metric import *
from utils.loss import choose_loss


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--mode", choices=['n2c', 'n2n'], default='n2c')
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--tlu", type=bool, default=True)
    parser.add_argument("--model", type=str, default='BRDNET')
    parser.add_argument("--loss", type=str, default='mse')
    parser.add_argument("--noise", type=str, default='nuclear')
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    return parser.parse_args()


def choose_model(model_name):
    if model_name == 'ADNET':
        return ADNET(3)
    elif model_name == 'DNCNN':
        return DNCNN(3)
    elif model_name == 'BRDNET':
        return BRDNET(3)
    elif model_name == 'feb_rfb_ab_mish_a':
        return feb_rfb_ab_mish_a(3)
    elif model_name == 'feb_mkm':
        return feb_mkm(3)
    elif model_name == 'conv15':
        return conv15(3)
    elif model_name == 'feb_rm':
        return feb_rm(3)
    elif model_name == 'feb':
        return feb(3)
    elif model_name == 'feb_rfb':
        return feb_rfb(3)
    elif model_name == 'feb_ab':
        return feb_ab(3)
    elif model_name == 'feb_mish':
        return feb_mish(3)
    elif model_name == 'feb_rfb_ab':
        return feb_rfb_ab(3)
    elif model_name == 'feb_rfb_mish':
        return feb_rfb_mish(3)
    elif model_name == 'feb_ab_mish':
        return feb_ab_mish(3)
    elif model_name == 'feb_rfb_ab_mish':
        return feb_rfb_ab_mish(3)
    elif model_name == 'feb_rfb_ab_mish_a_add':
        return feb_rfb_ab_mish_a_add(3)


class dataset_img2img(Dataset):
    def __init__(self, source_h5_path, target_h5_path):
        super(dataset_img2img, self).__init__()
        self.source_h5 = h5py.File(source_h5_path, 'r')
        self.target_h5 = h5py.File(target_h5_path, 'r')
        assert operator.eq(list(self.source_h5.keys()), list(self.target_h5.keys()))
        self.keys = list(self.source_h5.keys())
        random.shuffle(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        source = np.array(self.source_h5[key])
        target = np.array(self.target_h5[key])
        return torch.from_numpy(np.float32(source / 255)), torch.from_numpy(np.float32(target / 255))


def loaders(args, dir='h5'):
    noise_info = (args.noise).split(',')
    noise_name = ''.join(noise_info)
    source_h5_path_train = dir + '/' + args.mode + '_' + noise_name + '_source_train.h5'
    target_h5_path_train = dir + '/' + args.mode + '_' + noise_name + '_target_train.h5'
    source_h5_path_test = dir + '/' + args.mode + '_' + noise_name + '_source_test.h5'
    target_h5_path_test = dir + '/' + args.mode + '_' + noise_name + '_target_test.h5'

    dataset_train = dataset_img2img(source_h5_path_train, target_h5_path_train)
    dataset_test = dataset_img2img(source_h5_path_test, target_h5_path_test)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=args.batch, shuffle=True)
    loader_test = DataLoader(dataset=dataset_test, num_workers=0, batch_size=1, shuffle=False)
    print('train num:', len(dataset_train), 'test num:', len(dataset_test), 'NO VAL')
    print('steps:', len(loader_train))
    return loader_train, loader_test


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    model_name = args.mode + '_' + args.model + '_bz' + str(args.batch) + '_ep' + str(args.epochs) + '_' + args.loss
    noise_info = (args.noise).split(',')
    noise_name = ''.join(noise_info)
    save_root = 'results/date-7-4/' + noise_name
    if args.tlu:
        save_root += '_tlu'
    save_dir = os.path.join(save_root, model_name)
    if not os.path.exists(save_dir) and args.save:
        os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f)
    print('save dir:', save_dir)
    loader_train, loader_test = loaders(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = choose_model(args.model).to(device)
    criterion = choose_loss(args.loss)
    model_tlu = tlu(in_c=3, act_name='mish', asy=True, bias=True).to(device)
    criterion_tlu = choose_loss('mse')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    if args.save:
        writer = SummaryWriter(save_dir)
    else:
        writer = SummaryWriter('tmp')
    log_dicts = []
    for epoch in range(args.epochs):
        epoch_ = epoch + 1

        model.train()
        epoch_loss = 0
        for i, batch in enumerate(loader_train):
            batch_datas = batch[0].to(device=device)
            batch_labels = batch[1].to(device=device)
            batch_inferences = model(batch_datas)
            loss = criterion(batch_inferences, batch_labels).to(device)
            if args.tlu:
                batch_inferences_tlu = model_tlu(batch_datas)
                loss_tlu = criterion_tlu(batch_inferences_tlu, batch_labels).to(device)
                loss = loss + 0.1 * loss_tlu
            print("[Epoch %d][%d/%d] loss: %.4f" % (epoch_, i + 1, len(loader_train), loss.item()))
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss /= len(loader_train)

        model.eval()
        epoch_psnr = 0
        epoch_ssim = 0
        for i, batch in enumerate(loader_test):
            batch_datas = batch[0].to(device=device)
            batch_labels = batch[1].to(device=device)
            with torch.no_grad():
                batch_inferences = model(batch_datas)
                psnr_batch = batch_PSNR(batch_inferences, batch_labels, data_range=1.0)
                epoch_psnr += psnr_batch
                ssim_batch = ssim(batch_inferences, batch_labels, data_range=1.0, win_size=11).item()
                epoch_ssim += ssim_batch
        epoch_psnr /= len(loader_test)
        epoch_ssim /= len(loader_test)

        if args.save:
            if epoch_ == args.epochs:
                torch.save(model.state_dict(), os.path.join(save_dir, args.model + '_ep' + str(epoch_) + '.pth'))

            logs_tmp = {}
            logs_tmp.setdefault('epoch', epoch_)
            logs_tmp.setdefault('loss', epoch_loss)
            logs_tmp.setdefault('psnr', epoch_psnr)
            logs_tmp.setdefault('ssim', epoch_ssim)
            log_dicts.append(logs_tmp)
            print(logs_tmp)

            writer.add_scalar('loss', epoch_loss, epoch_)
            writer.add_scalar('psnr', epoch_psnr, epoch_)
            writer.add_scalar('ssmi', epoch_ssim, epoch_)
    writer.close()

    if args.save:
        with open(os.path.join(save_dir, 'logs.json'), 'w') as f:
            json.dump(log_dicts, f)


if __name__ == "__main__":
    args = get_args()
    search = True

    if search:
        models = ['ADNET', 'BRDNET', 'feb_rfb_ab_mish_a_add', 'DNCNN']
        losses = ['l1', 'mse']
        noises = ['nuclear', 'polyu',
                  'gaussian,25', 'gaussian,50', 'gaussian,75',
                  'text,25', 'text,50', 'text,75',
                  'implus,25', 'implus,50', 'implus,75']

        if args.gpu == 0:
            args.tlu = True
            args.model = models[0]
            for l in losses:
                for n in noises:
                    args.loss = l
                    args.noise = n
                    train(args)
        elif args.gpu == 1:
            args.tlu = True
            args.model = models[1]
            for l in losses:
                for n in noises:
                    args.loss = l
                    args.noise = n
                    train(args)
        elif args.gpu == 2:
            args.tlu = True
            args.model = models[2]
            for l in losses:
                for n in noises:
                    args.loss = l
                    args.noise = n
                    train(args)
        elif args.gpu == 3:
            args.tlu = True
            args.model = models[3]
            for l in losses:
                for n in noises:
                    args.loss = l
                    args.noise = n
                    train(args)
        elif args.gpu == 4:
            args.tlu = False
            args.model = models[0]
            for l in losses:
                for n in noises:
                    args.loss = l
                    args.noise = n
                    train(args)
        elif args.gpu == 5:
            args.tlu = False
            args.model = models[1]
            for l in losses:
                for n in noises:
                    args.loss = l
                    args.noise = n
                    train(args)
        elif args.gpu == 6:
            args.tlu = False
            args.model = models[2]
            for l in losses:
                for n in noises:
                    args.loss = l
                    args.noise = n
                    train(args)
        elif args.gpu == 7:
            args.tlu = False
            args.model = models[3]
            for l in losses:
                for n in noises:
                    args.loss = l
                    args.noise = n
                    train(args)
    else:
        train(args)
