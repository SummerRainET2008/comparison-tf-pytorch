import torch
import torch.nn as nn
import torch.nn.functional as F
from rnn_search_pytorch.beam import Beam
import time
from pa_nlp.nlp import Logger

class Encoder(nn.Module):
  """"encode the input sequence with Bi-GRU"""
  def __init__(self, layer_num, emb_dim, rnn_dim, vob_size, padding_idx,
               emb_dropout, hid_dropout):
    super(Encoder, self).__init__()
    self.nhid = rnn_dim
    self.emb = nn.Embedding(vob_size, emb_dim, padding_idx=padding_idx)
    self.bi_gru = nn.GRU(
      emb_dim, rnn_dim, layer_num, batch_first=True, bidirectional=True
    )
    self.enc_emb_dp = nn.Dropout(emb_dropout)
    self.enc_hid_dp = nn.Dropout(hid_dropout)
    self._layer_num = layer_num

  def init_hidden(self, batch_size):
    weight = next(self.parameters())
    h0 = weight.new_zeros(2 * self._layer_num, batch_size, self.nhid)
    return h0

  def forward(self, x, mask):
    hidden = self.init_hidden(x.size(0))
    #self.bi_gru.flatten_parameters()
    x = self.enc_emb_dp(self.emb(x))
    length = mask.sum(1).tolist()
    total_length = mask.size(1)
    x = torch.nn.utils.rnn.pack_padded_sequence(
      x, length, batch_first=True
    )
    output, hidden = self.bi_gru(x, hidden)
    output = torch.nn.utils.rnn.pad_packed_sequence(
      output, batch_first=True, total_length=total_length
    )[0]
    output = self.enc_hid_dp(output)
    hidden = torch.cat([hidden[0], hidden[1]], dim=-1)
    return output, hidden

class Attention(nn.Module):
  """Attention Mechanism"""
  def __init__(self, hidden_dim, values_dim, atten_dim):
    super(Attention, self).__init__()
    self._query_dense = nn.Linear(hidden_dim, atten_dim)
    self._values_dense = nn.Linear(values_dim, atten_dim)
    self.a2o = nn.Linear(atten_dim, 1)

  def forward(self, query, mask, values):
    shape = values.size()
    attn_h = self._values_dense(values.view(-1, shape[2]))
    attn_h = attn_h.view(shape[0], shape[1], -1)
    attn_h += self._query_dense(query).unsqueeze(1).expand_as(attn_h)
    logit = self.a2o(torch.tanh(attn_h)).view(shape[0], shape[1])
    if mask.any():
      logit.data.masked_fill_(torch.logical_not(mask), -float('inf'))
    logit = F.softmax(logit, dim=1)
    output = torch.bmm(logit.unsqueeze(1), values).squeeze(1)

    return output

class VallinaDecoder(nn.Module):
  def __init__(self, target_emb_dim, hidden_dim, enc_ncontext, atten_dim,
               dec_hidden_dim2, readout_dropout, tgt_vob_size):
    super(VallinaDecoder, self).__init__()
    self.gru1 = nn.GRUCell(target_emb_dim, hidden_dim)
    self.gru2 = nn.GRUCell(enc_ncontext, hidden_dim)
    self.enc_attn = Attention(hidden_dim, enc_ncontext, atten_dim)
    self.e2o = nn.Linear(target_emb_dim, dec_hidden_dim2)
    self.h2o = nn.Linear(hidden_dim, dec_hidden_dim2)
    self.c2o = nn.Linear(enc_ncontext, dec_hidden_dim2)
    self.readout_dp = nn.Dropout(readout_dropout)
    self.affine = nn.Linear(dec_hidden_dim2, tgt_vob_size)

  def forward(self, x, hidden, enc_mask, enc_context):
    hidden = self.gru1(x, hidden)
    attn_enc = self.enc_attn(hidden, enc_mask, enc_context)
    hidden = self.gru2(attn_enc, hidden)
    output = torch.tanh(self.e2o(x) + self.h2o(hidden) + self.c2o(attn_enc))
    output = self.readout_dp(output)
    output = self.affine(output)

    return output, hidden

class RNNSearch(nn.Module):
  def __init__(self, opt):
    super(RNNSearch, self).__init__()
    self.dec_hidden_dim1 = opt.dec_hidden_dim1
    self.dec_sos = opt.dec_sos
    self.dec_eos = opt.dec_eos
    self.dec_pad = opt.dec_pad
    self.enc_pad = opt.enc_pad

    self.tgt_emb = nn.Embedding(
      opt.tgt_vob_size, opt.target_emb_dim, padding_idx=opt.dec_pad
    )
    self.encoder = Encoder(
      opt.src_layer_num, opt.src_emb_dim, opt.src_rnn_dim, opt.src_vob_size,
      opt.enc_pad, opt.enc_emb_dropout, opt.enc_hid_dropout
    )
    self.decoder = VallinaDecoder(
      opt.target_emb_dim, opt.dec_hidden_dim1, 2 * opt.src_rnn_dim,
      opt.atten_dim, opt.dec_hidden_dim2, opt.readout_dropout, opt.tgt_vob_size,
    )
    self.init_affine = nn.Linear(2 * opt.src_rnn_dim, opt.dec_hidden_dim1)
    self.dec_emb_dropout = nn.Dropout(opt.dec_emb_dropout)

  def forward(self, src, src_mask, f_trg, f_trg_mask, b_trg=None,
              b_trg_mask=None):
    # print(src.size(), src_mask, f_trg.size(), f_trg_mask)

    enc_context, _ = self.encoder(src, src_mask)
    enc_context = enc_context.contiguous()

    avg_enc_context = enc_context.sum(1)
    enc_context_len = src_mask.sum(1).unsqueeze(-1).expand_as(avg_enc_context)
    avg_enc_context = avg_enc_context / enc_context_len

    attn_mask = src_mask.byte()

    hidden = torch.tanh(self.init_affine(avg_enc_context))

    loss = 0
    for i in range(f_trg.size(1) - 1):
      logit, hidden = self.decoder(
        self.dec_emb_dropout(self.tgt_emb(f_trg[:, i])), hidden, attn_mask,
        enc_context
      )
      loss += F.cross_entropy(
        logit, f_trg[:, i+1], reduce=False
      ) * f_trg_mask[:, i+1]

    w_loss = loss.sum() / f_trg_mask[:, 1:].sum()
    loss = loss.mean()
    return loss.unsqueeze(0), w_loss.unsqueeze(0)

  def beamsearch(self, src, src_mask, beam_size=10, normalize=False,
                 max_len=None, min_len=None):
    max_len = src.size(1) * 3 if max_len is None else max_len
    min_len = src.size(1) / 2 if min_len is None else min_len

    enc_context, _ = self.encoder(src, src_mask)
    enc_context = enc_context.contiguous()

    avg_enc_context = enc_context.sum(1)
    enc_context_len = src_mask.sum(1).unsqueeze(-1).expand_as(avg_enc_context)
    avg_enc_context = avg_enc_context / enc_context_len

    attn_mask = src_mask.byte()

    hidden = torch.tanh(self.init_affine(avg_enc_context))

    prev_beam = Beam(beam_size)
    prev_beam.candidates = [[self.dec_sos]]
    prev_beam.scores = [0]
    f_done = (lambda x: x[-1] == self.dec_eos)

    valid_size = beam_size

    hyp_list = []
    for k in range(max_len):
      candidates = prev_beam.candidates
      input = src.new_tensor(map(lambda cand: cand[-1], candidates))
      input = self.dec_emb_dropout(self.tgt_emb(input))
      output, hidden = self.decoder(input, hidden, attn_mask, enc_context)
      log_prob = F.log_softmax(self.affine(output), dim=1)
      if k < min_len:
        log_prob[:, self.dec_eos] = -float('inf')
      if k == max_len - 1:
        eos_prob = log_prob[:, self.dec_eos].clone()
        log_prob[:, :] = -float('inf')
        log_prob[:, self.dec_eos] = eos_prob
      next_beam = Beam(valid_size)
      done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)
      hyp_list.extend(done_list)
      valid_size -= len(done_list)

      if valid_size == 0:
        break
      
      beam_remain_ix = src.new_tensor(remain_list)
      enc_context = enc_context.index_select(0, beam_remain_ix)
      attn_mask = attn_mask.index_select(0, beam_remain_ix)
      hidden = hidden.index_select(0, beam_remain_ix)
      prev_beam = next_beam

    score_list = [hyp[1] for hyp in hyp_list]
    hyp_list = [
      hyp[0][1: hyp[0].index(self.dec_eos)]
      if self.dec_eos in hyp[0] else hyp[0][1:]
      for hyp in hyp_list
    ]
    if normalize:
      for k, (hyp, score) in enumerate(zip(hyp_list, score_list)):
        if len(hyp) > 0:
          score_list[k] = score_list[k] / len(hyp)
    score = hidden.new_tensor(score_list)
    sort_score, sort_ix = torch.sort(score)
    output = []
    for ix in sort_ix.tolist():
      output.append((hyp_list[ix], score[ix].item()))
    return output
