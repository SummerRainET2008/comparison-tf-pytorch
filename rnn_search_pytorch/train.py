import subprocess
from rnn_search_pytorch.dataset import dataset
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rnn_search_pytorch.util import (
  convert_data, invert_vocab, load_vocab, convert_str, sort_batch
)
import argparse
from rnn_search_pytorch.model import RNNSearch
import numpy as np
import os
import sys
import tempfile
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import param

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser(
  description='Training Attention-based Neural Machine Translation Model'
)
# data
parser.add_argument(
  '--src_layer_num', default=param.enc_layer_num, type=int, help='default 1'
)
parser.add_argument(
  '--src_vocab',
  default="rnn_search_pytorch/corpus/src.voc3.pkl", help='source vocabulary'
)
parser.add_argument(
  '--trg_vocab',
  default="rnn_search_pytorch/corpus/tgt.voc3.pkl", help='target vocabulary'
)
parser.add_argument(
  '--src_max_len', type=int, default=param.max_len_src,
  help='maximum length of source'
)
parser.add_argument(
  '--trg_max_len', type=int, default=param.max_len_trg,
  help='maximum length of target'
)
parser.add_argument(
  '--train_src',
  default="rnn_search_pytorch/corpus/src.txt", help='source for training'
)
parser.add_argument(
  '--train_trg',
  default="rnn_search_pytorch/corpus/tgt.txt", help='target for training'
)
parser.add_argument(
  '--valid_src',
  default="rnn_search_pytorch/corpus/vali.src.txt", help='source for validation'
)
parser.add_argument(
  '--valid_trg',
  default="rnn_search_pytorch/corpus/vali.tgt.txt", nargs='+',
  help='references for validation'
)
parser.add_argument(
  '--vfreq', type=int, default=1500, help='frequency for validation'
)
parser.add_argument(
  '--eval_script', default="scripts/validate.sh", help='script for validation'
)
# model
parser.add_argument(
  '--name', type=str, default='', help='the name of checkpoint'
)
parser.add_argument(
  '--src_emb_dim', type=int, default=param.embedding_dim,
  help='size of source word embedding'
)
parser.add_argument(
  '--target_emb_dim', type=int, default=param.embedding_dim,
  help='size of target word embedding'
)
parser.add_argument(
  '--src_rnn_dim', type=int, default=param.rnn_hidden_dim,
  help='size of source hidden layer'
)
parser.add_argument(
  '--dec_hidden_dim1', type=int, default=param.rnn_hidden_dim,
  help='size of target hidden layer'
)
parser.add_argument(
  '--atten_dim', type=int, default=param.rnn_hidden_dim,
  help='size of target attention layer'
)
parser.add_argument(
  '--dec_hidden_dim2', type=int, default=param.rnn_hidden_dim,
  help='size of maxout layer'
)
parser.add_argument(
  '--enc_emb_dropout', type=float, default=0.3,
  help='dropout rate for encoder embedding'
)
parser.add_argument(
  '--dec_emb_dropout', type=float, default=0.3,
  help='dropout rate for decoder embedding'
)
parser.add_argument(
  '--enc_hid_dropout', type=float, default=0.3,
  help='dropout rate for encoder hidden state'
)
parser.add_argument(
  '--readout_dropout', type=float, default=0.3,
  help='dropout rate for readout layer'
)
# optimization
parser.add_argument(
  '--optim', default='RMSprop', help='optimization algorihtim'
)
parser.add_argument(
  '--batch_size', type=int, default=param.batch_size,
  help='input batch size for training'
)
parser.add_argument(
  '--lr', type=float, default=param.lr, help='learning rate'
)
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument(
  '--grad_clip', type=float, default=5, help='gradient clipping'
)
parser.add_argument(
  '--finetuning', action='store_true', help='whether or not fine-tuning'
)
parser.add_argument(
  '--decay_lr', action='store_true', help='decay learning rate'
)
parser.add_argument(
  '--half_epoch', action='store_true',
  help='decay learning rate at the beginning of epoch'
)
parser.add_argument(
  '--epoch_best', action='store_true', help='store best model for epoch'
)
parser.add_argument(
  '--restore', action='store_true',
  help='decay learning rate at the beginning of epoch'
)
parser.add_argument(
  '--beam_size', type=int, default=10, help='size of beam search'
)
parser.add_argument(
  '--sfreq', type=int, default=500, help='frequency for sampling'
)
parser.add_argument('--seed', type=int, default=123, help='random number seed')
parser.add_argument(
  '--checkpoint', type=str, default='./checkpoint/',
  help='path to save the model'
)
parser.add_argument('--freq', type=int, help='frequency for save')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--gpu_id', type=int, help='')
parser.add_argument(
  '--epoch_num', type=int, default=10, help='default  10.'
)
parser.add_argument('--info', type=str, help='info of model')

opt = parser.parse_args()
print(opt)

# set the random seed manually
if opt.gpu_id:
  opt.seed += opt.gpu_id
torch.manual_seed(opt.seed)

opt.cuda = opt.cuda and torch.cuda.is_available()
if opt.cuda:
  torch.cuda.manual_seed(opt.seed)

device_type= 'cuda' if opt.cuda else 'cpu'
device_ids = None
if opt.gpu_id is not None:
  device_type += ':' + str(opt.gpu_id)
  device_ids = [opt.gpu_id]
device = torch.device(device_type)
# torch.cuda.synchronize()

# load vocabulary for source and target
src_vocab, trg_vocab = {}, {}
src_vocab['stoi'] = load_vocab(opt.src_vocab)
trg_vocab['stoi'] = load_vocab(opt.trg_vocab)
src_vocab['itos'] = invert_vocab(src_vocab['stoi'])
trg_vocab['itos'] = invert_vocab(trg_vocab['stoi'])
UNK = '<unk>'
SOS = '<sos>'
EOS = '<eos>'
PAD = '<pad>'
opt.enc_pad = src_vocab['stoi'][PAD]
opt.dec_sos = trg_vocab['stoi'][SOS]
opt.dec_eos = trg_vocab['stoi'][EOS]
opt.dec_pad = trg_vocab['stoi'][PAD]
opt.src_vob_size = len(src_vocab['stoi'])
opt.tgt_vob_size = len(trg_vocab['stoi'])

train_dataset = dataset(
  opt.train_src, opt.train_trg, opt.src_max_len, opt.trg_max_len
)
valid_dataset = dataset(opt.valid_src, opt.valid_trg)
train_iter = torch.utils.data.DataLoader(
  train_dataset, opt.batch_size, shuffle=True, num_workers=0,
  collate_fn=lambda x: zip(*x)
)
valid_iter = torch.utils.data.DataLoader(
  valid_dataset, 1, shuffle=False, collate_fn=lambda x: zip(*x)
)

model = RNNSearch(opt).to(device)

# initialize the parameters
for p in model.parameters():
  p.data.uniform_(-0.1, 0.1)

if opt.name:
  state_dict = torch.load(os.path.join(opt.checkpoint, opt.name))
  model.load_state_dict(state_dict)

param_list = list(model.parameters())
param_group = param_list

# create the optimizer
optimizer = getattr(optim, opt.optim)(
  param_group, lr=opt.lr, weight_decay=opt.l2
)

opt.score_list = []
opt.epoch_best_score = -float('inf')
opt.cur_lr = ' '.join(map(lambda g: str(g['lr']), optimizer.param_groups))
opt.tmp_name = None
opt.best_name = None
opt.epoch_best_name = None

def save_model(model, batch_idx, epoch, info='tmp'):
  date = time.strftime('%m-%d|%H:%M',time.localtime(time.time()))
  name = 'model_%s_%s_lr_%.1e_cur_lr_%s_l2_%.1e_batch_%d_e%d-%d_%s.%s.pt' \
         % (opt.model, opt.info, opt.lr, opt.cur_lr, opt.l2, opt.batch_size,
            epoch, batch_idx, date, info)
  torch.save(model.state_dict(), os.path.join(opt.checkpoint, name))
  return name

def adjust_learningrate(score_list):
    for k, group in enumerate(optimizer.param_groups):
      group['lr'] = group['lr'] * 0.5

def train(epoch):
  model.train()
  opt.epoch_best_score = -float('inf')
  opt.epoch_best_name = None
  for batch_idx, batch in enumerate(train_iter, start=1):
    start_time = time.time()
    batch = sort_batch(batch)
    src_raw = batch[0]
    tgt_raw = batch[1]
    src_x, src_mask_x = convert_data(
      src_raw, src_vocab, device, True, UNK, PAD, SOS, EOS
    )
    f_tgt_y, f_tgt_mask_y = convert_data(
      tgt_raw, trg_vocab, device, False, UNK, PAD, SOS, EOS
    )
    b_tgt_y, b_tgt_mask_y = convert_data(
      tgt_raw, trg_vocab, device, True, UNK, PAD, SOS, EOS
    )
    # start_time = time.time()
    optimizer.zero_grad()
    if opt.cuda and torch.cuda.device_count() > 1 and opt.gpu_id is None:
      R = nn.parallel.data_parallel(
        model,
        (src_x, src_mask_x, f_tgt_y, f_tgt_mask_y, b_tgt_y, b_tgt_mask_y),
        device_ids
      )
    else:
      R = model(src_x, src_mask_x, f_tgt_y, f_tgt_mask_y, b_tgt_y, b_tgt_mask_y)

    R[0].mean().backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(param_list, opt.grad_clip)
    optimizer.step()

    batch_duration = time.time() - start_time
    print(
      f"epoch[{epoch}], batch: {batch_idx}, batch_data: {src_x.size()}, "
      f"loss: {R[1].item()}, grad_norm: {grad_norm}, lr: {opt.cur_lr}, "
      f"time: {batch_duration}"
    )

    # validation
    if batch_idx % opt.vfreq == 0:
      evaluate(batch_idx, epoch)
      model.train()
      if len(opt.score_list) == 1 or \
        opt.score_list[-1][0] > max(map(lambda x: x[0], opt.score_list[:-1])):
        if opt.best_name is not None:
          os.remove(os.path.join(opt.checkpoint, opt.best_name))
        opt.best_name = save_model(model, batch_idx, epoch, 'best')
      if opt.epoch_best and opt.score_list[-1][0] > opt.epoch_best_score:
        opt.epoch_best_score = opt.score_list[-1][0]
        if opt.epoch_best_name is not None:
          os.remove(os.path.join(opt.checkpoint, opt.epoch_best_name))
        opt.epoch_best_name = save_model(model, batch_idx, epoch, 'epoch-best') 

    # sampling
    if batch_idx % opt.sfreq == 0:
      length = len(src_raw)
      ix = np.random.randint(0, length)
      samp_src_raw = [src_raw[ix]]
      samp_trg_raw = [tgt_raw[ix]]
      samp_src, samp_src_mask = convert_data(
        samp_src_raw, src_vocab, device, True, UNK, PAD, SOS, EOS
      )

      model.eval()
      with torch.no_grad():
        output = model.beamsearch(samp_src, samp_src_mask, opt.beam_size)

      best_hyp, best_score = output[0]
      best_hyp = convert_str([best_hyp], trg_vocab)
      print('--', ' '.join(samp_src_raw[0]))
      print('--', ' '.join(samp_trg_raw[0]))
      print('--', ' '.join(best_hyp[0]))
      print('--', best_score)
      model.train()

    # saving model
    if opt.freq and batch_idx % opt.freq == 0:
      if opt.tmp_name is not None:
        os.remove(os.path.join(opt.checkpoint, opt.tmp_name))
      opt.tmp_name = save_model(model, batch_idx, epoch, 'tmp')

def bleu_script(f):
  ref_stem = opt.valid_trg[0][:-1] + '*'
  cmd = '{eval_script} {refs} {hyp}'.format(eval_script=opt.eval_script, refs=ref_stem, hyp=f)
  p = subprocess.Popen(
    cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
  )
  out, err = p.communicate()
  if p.returncode > 0:
    sys.stderr.write(err)
    sys.exit(1)
  bleu = float(out)
  return bleu


def evaluate(batch_idx, epoch):
  model.eval()
  hyp_list = [] 
  ref_list = []
  start_time = time.time()
  for ix, batch in enumerate(valid_iter, start=1):
    batch = list(batch)
    src_raw = batch[0]
    trg_raw = batch[1]
    src, src_mask = convert_data(
      src_raw, src_vocab, device, True, UNK, PAD, SOS, EOS
    )

    with torch.no_grad():
      output = model.beamsearch(src, src_mask, opt.beam_size, normalize=True)
      best_hyp, best_score = output[0]
      best_hyp = convert_str([best_hyp], trg_vocab)
      hyp_list.append(best_hyp[0])
      ref = map(lambda x: x[0], trg_raw)
      ref_list.append(ref)

  elapsed = time.time() - start_time
  bleu1 = corpus_bleu(
    ref_list, hyp_list, smoothing_function=SmoothingFunction().method1
  )
  hyp_list = map(lambda x: ' '.join(x), hyp_list)
  p_tmp = tempfile.mktemp()
  f_tmp = open(p_tmp, 'w')
  f_tmp.write('\n'.join(hyp_list)) 
  f_tmp.close()

  bleu2 = bleu_script(p_tmp)
  print(
    'BLEU score for {}-{} is {}/{}, {}'.format(
      epoch, batch_idx, bleu1, bleu2, elapsed
    )
  )
  opt.score_list.append((bleu2, batch_idx, epoch))

for epoch in range(opt.epoch_num):
  train(epoch)
  print('-----------------------------------')
  evaluate(len(train_iter), epoch)
  print('-----------------------------------')

  if len(opt.score_list) == 1 or \
      opt.score_list[-1][0] > max(map(lambda x: x[0], opt.score_list[:-1])):
    if opt.best_name is not None:
      os.remove(os.path.join(opt.checkpoint, opt.best_name))
    opt.best_name = save_model(model, len(train_iter), epoch, 'best')

  if opt.epoch_best and opt.score_list[-1][0] > opt.epoch_best_score:
    opt.epoch_best_score = opt.score_list[-1][0]

    if opt.epoch_best_name is not None:
      os.remove(os.path.join(opt.checkpoint, opt.epoch_best_name))

    opt.epoch_best_name = save_model(
      model, len(train_iter), epoch, 'epoch-best'
    )

  if opt.half_epoch:
    cur_lr_list = []

    for k, group in enumerate(optimizer.param_groups):
      group['lr'] = group['lr'] * 0.5
      cur_lr_list.append(group['lr'])
    opt.cur_lr = ' '.join(map(lambda v: str(v), cur_lr_list))
    print('Current learning rate:', opt.cur_lr)

best = max(opt.score_list, key=lambda x: x[0])
print('best BLEU {}-{}: {}'.format(best[2], best[1], best[0]))

