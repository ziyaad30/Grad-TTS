import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from data import TextMelDataset, TextMelBatchCollate
from model import GradTTS
from model.utils import fix_len_compatibility
from text import symbols
from utils import load_checkpoint, load_ckpt, plot_tensor, oldest_checkpoint_path


class Trainer:
    def __init__(self, args):
        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.logger = None
        self.test_batch = None
        self.epoch = 0
        self.isRunning = True
        self.load_original = args.load_old
        self.resume = True
        self.pretrained = args.pretrained

        self.out_size = fix_len_compatibility(2 * 22050 // 256)
        self.scalar_interval = 100
        self.log_dir = './logs'
        self.train_filelist_path = './dataset/train.txt'
        self.valid_filelist_path = './dataset/valid.txt'

        self.eval_interval = args.interval
        self.log_interval = 10

        self.batch_size = args.batch
        self.learning_rate = 0.0001
        self.weight_decay = 1e-6
        self.betas = (0.9, 0.999)
        self.eps = 1e-6
        self.n_epochs = args.epochs
        self.seed = 1234
        self.ckpt_dir = args.save_path
        self.milestones = [100000, 150000, 200000]
        self.test_size = 4

        self.iteration = 0
        self.add_blank = True

    def train(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        n_symbols = len(symbols) + 1 if self.add_blank else len(symbols)

        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)
            os.chmod(self.log_dir, 0o775)

        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            os.chmod(self.ckpt_dir, 0o775)

        print('Initializing logger...')
        self.logger = SummaryWriter(log_dir=self.log_dir)

        print('Initializing data loaders...')
        train_dataset = TextMelDataset(self.train_filelist_path, params.add_blank, self.seed,
                                       params.n_fft, params.n_feats, params.sample_rate, params.hop_length,
                                       params.win_length, params.f_min, params.f_max)
        batch_collate = TextMelBatchCollate()
        loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                            collate_fn=batch_collate, drop_last=True,
                            num_workers=0, shuffle=False)
        test_dataset = TextMelDataset(self.valid_filelist_path, params.add_blank, self.seed,
                                      params.n_fft, params.n_feats, params.sample_rate, params.hop_length,
                                      params.win_length, params.f_min, params.f_max)

        print('Initializing model...')
        self.model = GradTTS(n_symbols, params.n_spks, params.spk_emb_dim,
                             params.n_enc_channels, params.filter_channels,
                             params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                             params.enc_kernel, params.enc_dropout, params.window_size,
                             params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale).cuda()

        print('Initializing optimizer and scheduler...')
        # optimizer = torch.optim.Adam(params=self.model.parameters(), lr=params.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=self.betas,
                                          eps=self.eps, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones, gamma=0.5)

        if self.load_original:
            print('Loading old original Grad TTS model...')
            self.model, iteration = load_checkpoint("pretrained_models", self.model)
            self.iteration = 0
        else:
            if self.pretrained:
                try:
                    self.model, self.iteration, self.optimizer, self.scheduler = load_ckpt("pretrained_models",
                                                                                           self.model, self.optimizer,
                                                                                           self.scheduler)
                    self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
                    self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones,
                                                                          gamma=0.5)
                    self.iteration = 0
                    print('Running trainer with pretrained model.')
                except Exception as e:
                    print(e)
            else:
                if self.resume:
                    try:
                        self.model, self.iteration, self.optimizer, self.scheduler = load_ckpt(self.ckpt_dir,
                                                                                               self.model,
                                                                                               self.optimizer,
                                                                                               self.scheduler)
                        self.iteration += 1
                        print('Resuming training...')
                    except (Exception,):
                        print('No checkpoints found, continuing new training!')

        print('Number of encoder + duration predictor parameters: %.2fm' % (self.model.encoder.nparams / 1e6))
        print('Number of decoder parameters: %.2fm' % (self.model.decoder.nparams / 1e6))
        print('Total parameters: %.2fm' % (self.model.nparams / 1e6))

        print('Logging test batch...')
        self.test_batch = test_dataset.sample_test_batch(size=self.test_size)
        for i, item in enumerate(self.test_batch):
            mel = item['y']
            self.logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                                  global_step=0, dataformats='HWC')

        print('Start training...')

        for epoch in range(1, self.n_epochs + 1):
            self.epoch = epoch
            if not self.isRunning:
                self.save_checkpoint()
                break
            self.model.train()
            dur_losses = []
            prior_losses = []
            diff_losses = []
            for batch_idx, batch in enumerate(loader):
                if not self.isRunning:
                    break

                self.model.zero_grad()

                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                dur_loss, prior_loss, diff_loss = self.model.compute_loss(x, x_lengths,
                                                                          y, y_lengths,
                                                                          out_size=self.out_size)
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.decoder.parameters(),
                                                               max_norm=1)
                self.optimizer.step()

                lr = self.scheduler.get_last_lr()[0]

                if self.iteration % self.scalar_interval == 0 and self.iteration > 0:
                    self.logger.add_scalar('training/duration_loss', dur_loss.item(),
                                           global_step=self.iteration)
                    self.logger.add_scalar('training/prior_loss', prior_loss.item(),
                                           global_step=self.iteration)
                    self.logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                           global_step=self.iteration)
                    self.logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                           global_step=self.iteration)
                    self.logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                           global_step=self.iteration)
                    self.logger.add_scalar('training/learning_rate', lr,
                                           global_step=self.iteration)

                    dur_losses.append(dur_loss.item())
                    prior_losses.append(prior_loss.item())
                    diff_losses.append(diff_loss.item())

                if self.iteration % self.log_interval == 0:
                    msg = (f"epoch: {epoch}, step: {self.iteration}, dur_loss: {dur_loss.item():.5f}, "
                           f"prior_loss: {prior_loss.item():.5f}, diff_loss: {diff_loss.item():.5f}, lr: {lr:.7f}")
                    print(msg)

                if self.iteration % self.eval_interval == 0 and self.iteration > 0:
                    self.model.eval()
                    print('Evaluating model...')
                    self.evaluate()
                    print('Evaluation completed.')
                    self.save_checkpoint()
                    self.model.train()

                self.iteration += 1
                self.scheduler.step()

            log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
            log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
            log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
            with open(f'{self.log_dir}/train.log', 'a') as f:
                f.write(log_msg)
        self.save_checkpoint()
        print(f'Total epochs of {self.epoch} completed. Training stopped.')

    def evaluate(self):
        with torch.no_grad():
            for i, item in enumerate(self.test_batch):
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                y_enc, y_dec, attn = self.model(x, x_lengths, n_timesteps=50)
                self.logger.add_image(f'image_{i}/generated_enc',
                                      plot_tensor(y_enc.squeeze().cpu()),
                                      global_step=self.iteration, dataformats='HWC')
                self.logger.add_image(f'image_{i}/generated_dec',
                                      plot_tensor(y_dec.squeeze().cpu()),
                                      global_step=self.iteration, dataformats='HWC')
                self.logger.add_image(f'image_{i}/alignment',
                                      plot_tensor(attn.squeeze().cpu()),
                                      global_step=self.iteration, dataformats='HWC')

                print(f'Eval step: { i +1} / {len(self.test_batch)}')

    def save_checkpoint(self):
        torch.save(
            {
                "iteration": self.iteration,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            },
            f=f"{self.ckpt_dir}/G_{self.iteration}.pt"
        )

        print(f"Checkpoint saved to {self.ckpt_dir}/G_{self.iteration}.pt")

        old_g = oldest_checkpoint_path(self.ckpt_dir)
        if os.path.exists(old_g):
            os.remove(old_g)
            print(f"Removed checkpoint {old_g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_path", type=str, default = './checkpoints', help="model save path")
    parser.add_argument("-e", "--epochs", type=int, default = 2000, help="number of training epochs")
    parser.add_argument("-i", "--interval", type=int, default = 1000, help="each number of steps to save model")
    parser.add_argument("-b", "--batch", type=int, default = 16, help="training batch size")
    parser.add_argument('-o', '--load_old', action = 'store_true', help='load original grad-tts.pt')
    parser.add_argument('-p', '--pretrained', action = 'store_true', help='whether to use model as pretrained model')
    args = parser.parse_args()
    train = Trainer(args)
    train.train()
