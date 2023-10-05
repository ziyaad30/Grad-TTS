import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import GradTTS
from data import TextMelDataset, TextMelBatchCollate
from utils import plot_tensor, save_plot, oldest_checkpoint_path, load_ckpt
from text.symbols import symbols


train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path

nsymbols = len(symbols) + 1 if params.add_blank else len(symbols)
pbar = tqdm


def train(args):
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)
    
    if not os.path.isdir(params.log_dir):
        os.makedirs(params.log_dir)
        os.chmod(params.log_dir, 0o775)
        
    if not os.path.isdir(params.ckpt_dir):
        os.makedirs(params.ckpt_dir)
        os.chmod(params.ckpt_dir, 0o775)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=params.log_dir)

    print('Initializing data loaders...')
    train_dataset = TextMelDataset(train_filelist_path, params.add_blank,
                                   params.n_fft, params.n_feats, params.sample_rate, params.hop_length,
                                   params.win_length, params.f_min, params.f_max)
    batch_collate = TextMelBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=0, shuffle=False)
    test_dataset = TextMelDataset(valid_filelist_path, params.add_blank,
                                  params.n_fft, params.n_feats, params.sample_rate, params.hop_length,
                                  params.win_length, params.f_min, params.f_max)

    print('Initializing model...')
    model = GradTTS(nsymbols, params.n_spks, params.spk_emb_dim,
                    params.n_enc_channels, params.filter_channels,
                    params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                    params.enc_kernel, params.enc_dropout, params.window_size,
                    params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale).cuda()
    
    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)
    '''
    model = load_checkpoint("pretrained_models", model)
    iteration = 0
    '''
    if args.pretrained:
        try:
            model, iteration = load_ckpt("pretrained_models", model)
            iteration = 0
            print('Running trainer with pretrained model.')
        except Exception as e:
            print(e)
            iteration = 0
    else:
        if args.resume:
            try:
                model, iteration = load_ckpt(params.ckpt_dir, model)
                iteration += 1
                print('Resuming training...')
            except:
                iteration = 0
                print('No checkpoints found, continuing new training!')
        else:
            iteration = 0
    
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=params.test_size)
    for i, item in enumerate(test_batch):
        mel = item['y']
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{params.log_dir}/original_{i}.png')

    print('Start training...')
    
    for epoch in range(1, params.n_epochs + 1):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        for batch_idx, batch in enumerate(pbar(loader, desc="Epoch {}".format(epoch), position=1, leave=False)):
            model.zero_grad()
            x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
            y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
            dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                 y, y_lengths,
                                                                 out_size=params.out_size)
            loss = sum([dur_loss, prior_loss, diff_loss])
            loss.backward()

            enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                           max_norm=1)
            dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                           max_norm=1)
            optimizer.step()

            lr = optimizer.param_groups[0]['lr']
            
            if iteration % params.scalar_interval == 0:
                logger.add_scalar('training/duration_loss', dur_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/learning_rate', lr,
                                  global_step=iteration)
            
                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())
                
                
            if iteration % params.log_interval == 0:
                pbar.write(f"step: {iteration}, dur_loss: {dur_loss.item():.5f}, prior_loss: {prior_loss.item():.5f}, diff_loss: {diff_loss.item():.5f}, lr: {lr:.7f}")
                
            if iteration % params.eval_interval == 0 and iteration > 0:
                model.eval()
                pbar.write('Evaluating...')
                evaluate(model, test_batch, iteration, logger)
                pbar.write('Done Evaluating!')
                model.train()
            
            if iteration % params.save_interval == 0 and iteration > 0:
                save_checkpoint(model, iteration)
            iteration += 1

        log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{params.log_dir}/train.log', 'a') as f:
            f.write(log_msg)
    print(f'Total epochs of {n_epochs} reached. Training stopped.')


def evaluate(model, test_batch, iteration, logger):
    with torch.no_grad():
        for i, item in enumerate(tqdm(test_batch, desc="Evaluation", position=0, leave=False)):
            x = item['x'].to(torch.long).unsqueeze(0).cuda()
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50)
            logger.add_image(f'image_{i}/generated_enc',
                             plot_tensor(y_enc.squeeze().cpu()),
                             global_step=iteration, dataformats='HWC')
            logger.add_image(f'image_{i}/generated_dec',
                             plot_tensor(y_dec.squeeze().cpu()),
                             global_step=iteration, dataformats='HWC')
            logger.add_image(f'image_{i}/alignment',
                             plot_tensor(attn.squeeze().cpu()),
                             global_step=iteration, dataformats='HWC')
            save_plot(y_enc.squeeze().cpu(), 
                      f'{params.log_dir}/generated_enc_{i}.png')
            save_plot(y_dec.squeeze().cpu(), 
                      f'{params.log_dir}/generated_dec_{i}.png')
            save_plot(attn.squeeze().cpu(), 
                      f'{params.log_dir}/alignment_{i}.png')


def save_checkpoint(model, iteration):
    torch.save(
        {
            "iteration": iteration,
            "state_dict": model.state_dict(),
        },
        f=f"{params.ckpt_dir}/G_{iteration}.pt"
    )
    
    pbar.write(f"Checkpoint saved to {params.ckpt_dir}/G_{iteration}.pt")
    
    old_g = oldest_checkpoint_path(params.ckpt_dir)
    if os.path.exists(old_g):
        os.remove(old_g)
        pbar.write(f"Removed checkpoint {old_g}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', action = 'store_true', help='whether to resume training')
    parser.add_argument('-p', '--pretrained', action = 'store_true', help='whether to use model as pretrained model')
    args = parser.parse_args()
    
    train(args)
