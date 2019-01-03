import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
import torch.backends.cudnn as cudnn
import numpy as np
from collections import OrderedDict
from sru import SRU

from weldon import WeldonPool2d
import os


def EncoderImage(pretrain_path, input_size=224, D=2048, D_prime=2400):
    """
    A wrapper to image encoders.
    """
    img_enc = ImageEncoder(input_size, D, D_prime)
    img_enc.load_state_dict(torch.load(pretrain_path))

    return img_enc

class ImageEncoder(nn.Module):
  def __init__(self, input_size, D=2048, D_prime=2400):
    super(ImageEncoder, self).__init__()
    self.resnet = models.resnet152(pretrained=True) # feature extractor
    self.spaConv = nn.Conv2d(D, D_prime, 1)
    self.spool = WeldonPool2d(kmax=15, kmin=15)
    self.dropout = nn.Dropout(p=0.5)
    self.proj = nn.Linear(D_prime, D_prime, bias=True)

  def forward(self, x):
    # x = self.resnet(x) #[B, 2048, 7, 7]
    for n, layer in self.resnet.named_children():
        x = layer(x)
        if n in ['layer4']:
            break
    x = self.spaConv(x) #[B, 2400, 7, 7]
    x = self.spool(x) #[B, 2400]
    x = self.dropout(x)
    x = self.proj(x) #[B, 2400]
    x = self.l2norm(x)

    return x

  def l2norm(self, x):
    norm2 = torch.norm(x, 2, dim=1, keepdim=True)
    x = torch.div(x, norm2)
    return x

# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab, vocab_size, pretrain_path, K=620, d=2400, num_layers=4):
        super(EncoderText, self).__init__()
        self.embed_size = d

        # word embedding
        self.embed = nn.Embedding(vocab_size, K)
        self.rnn = SRU(input_size=K,
                       hidden_size=d,
                       num_layers=num_layers,
                       rnn_dropout=0.25,
                       use_tanh=1)
        if pretrain_path:
            self.load_embedding(vocab,
                                self.load_tables(pretrain_path),
                                K)
        else:
            self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def l2norm(self, x):
        norm2 = torch.norm(x, 2, dim=1, keepdim=True)
        x = torch.div(x, norm2)
        return x

    # https://github.com/ryankiros/skip-thoughts/blob/master/skipthoughts.py#L74
    def load_tables(self, path):
        '''
        Load the tables
        '''
        words = []
        utable = np.load(os.path.join(path, 'utable.npy'))
        with open(os.path.join(path, 'dictionary.txt'), 'rb') as f:
            for line in f:
                words.append(line.decode('utf-8').strip())
        utable = OrderedDict(zip(words, utable))
        return utable

    def load_embedding(self, vocab, table, K=620):
        # word2vec: "Skip-thought vectors" [NIPS2015]
        # https://github.com/ryankiros/skip-thoughts
        print('     load pretrained word2vec embedding...')
        num_embed = len(vocab)
        weights_matrix = torch.zeros((num_embed, K))
        unknown_params = table['UNK'] # dicts: {word: idx}
        unknown = 0
        for idx in range(num_embed):
          try:
            word = vocab.idx2word[idx]
            weights_matrix[idx] = torch.from_numpy(table[word])
          except KeyError:
            weights_matrix[idx] = torch.from_numpy(unknown_params)
            unknown += 1

        # self.embedding = nn.Embedding(num_embed, K)
        self.embed.load_state_dict({'weight': weights_matrix})
        if unknown > 0:
            print('Warning: {}/{} words are not in dictionary, thus set UNK'
                  .format(unknown, num_embed))
        # if self.save:
        #     torch.save(self.embedding, path)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x) # (B, S, E)
        out, _ = self.rnn(x.transpose(1, 0))
        out = out.transpose(1, 0)
        I = torch.LongTensor(lengths).view(-1, 1, 1)

        I = (I.expand(x.size(0), 1, self.embed_size)-1).cuda()

        out = torch.gather(out, 1, I).squeeze(1)

        out = self.l2norm(out)

        return out


class HardNegativeContrastiveLoss(nn.Module):
    def __init__(self, nmax=1, margin=0.2, lbd=0, lbd2=0):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.margin = margin
        self.nmax = nmax
        self.lbd = lbd
        self.lbd2 = lbd2

    def forward(self, imgs, caps):
        scores = torch.mm(imgs, caps.t()) # (160, 160)
        diag = scores.diag() # (160,)

        scores = (scores - 2 * torch.diag(scores.diag())) # ?

        # Sort the score matrix in the caption dimension
        sorted_cap, _ = torch.sort(scores, 0, descending=True)

        # Sort the score matrix in the image dimension
        sorted_img, _ = torch.sort(scores, 1, descending=True)

        # Select the nmax score
        max_c = sorted_cap[:self.nmax, :] # (1, 160)
        max_i = sorted_img[:, :self.nmax] # (160, 1)

        neg_cap = torch.sum(torch.clamp(max_c * (1 - self.lbd) + (
            (self.margin - diag) * (1 + self.lbd)).view(1, -1).expand_as(max_c), min=0))
        neg_img = torch.sum(torch.clamp(max_i * (1 - self.lbd) + (
            (self.margin - diag) * (1 + self.lbd)).view(-1, 1).expand_as(max_i), min=0))

        loss = neg_cap * (1 + self.lbd2) + neg_img * (1 - self.lbd2)

        return loss


class VSE(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt):
        # tutorials/09 - Image Captioning
        # Build Models
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.img_enc = EncoderImage(pretrain_path=('../pth/init.pth'),
                                    input_size=opt.crop_size,
                                    D=opt.D,
                                    D_prime=opt.D_prime
                                    )
        self.txt_enc = EncoderText(vocab=opt.vocab,
                                   vocab_size=opt.vocab_size,
                                   pretrain_path=('../w2v'),
                                   K=opt.K,
                                   d=opt.d,
                                   num_layers=opt.num_layers
                                   )

        # Use muliple GPU
        if torch.cuda.device_count() > 1:
            print('     Use muliple GPUs...')
            self.img_enc = nn.DataParallel(self.img_enc)
            # self.txt_enc = nn.DataParallel(self.txt_enc)
        # GPU mode.
        print('     Change to GPU mode...')
        self.img_enc.to(self.device)
        self.txt_enc.to(self.device)

        cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = HardNegativeContrastiveLoss()
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def set_requires_grad(self, nets, flag=True):
        for net in nets:
            for param in net.parameters():
                param.requires_grad = flag

    def change_training_state(self, epoch):
        # For 1 epoch, only update the weight of the last conv layer
        # After 2 epoch, also update the weight of the text encoder.
        # After 8 epoch, fine-tune the resnet of the image encoder.
        if epoch == 0:
            self.set_requires_grad([self.txt_enc, self.img_enc], False)
            layer = self.img_enc.module if torch.cuda.device_count() > 1 else self.img_enc
            layer.proj.weight.requires_grad = True
            layer.proj.bias.requires_grad = True

            print('Turn on only requires_grad of last conv layer...')

        elif epoch == 2:
            self.set_requires_grad([self.txt_enc], True)
            print('Turn on requires_grad of text encoder...')

        elif epoch == 8: # cuda memory out issue.
            for name, p in self.img_enc.named_parameters():
                if 'layer4' or 'spaConv' in name:
                    p.requires_grad = True
            print('Turn on requires_grad of image encoder...')


    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = images.to(self.device)
        captions = captions.to(self.device)
        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb)
        loss.backward()
        self.optimizer.step()
