import pickle
import os
import time
import shutil

import torch
import data
from vocab import Vocabulary  # NOQA
from model import VSE
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data

import logging
import tensorboard_logger as tb_logger

import argparse


def main():
    parser = argparse.ArgumentParser()
    # Directories.
    parser.add_argument('--data_path', default='/DATA/cvpr19',
                        help='path to datasets')
    parser.add_argument('--vocab_path', default='../vocab/',
                        help='Path to saved vocabulary pickle files.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='beenburger',
                        help='model_name')
    parser.add_argument('--save_dir', type=str, default='coco2',
                        help='save checkpoint and results in DATA_PATH/MODEL_NAME/SAVE_DIR')
    # Dataset.
    parser.add_argument('--data_name', default='coco',
                        help='{coco|ours}')
    parser.add_argument('--use_restval', default='True', type=str2bool,
                        help='Use the restval data for training on MSCOCO.')
    # Model configurations.
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--K', default=620, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--num_layers', default=4, type=int,
                        help='Number of SRU layers.')
    parser.add_argument('--D', type=int, default='2048',
                        help='dimension of image feature from ResNet')
    parser.add_argument('--D_prime', type=int, default='2400',
                        help='dimension of adaptation + pooling')
    parser.add_argument('--d', type=int, default='2400',
                        help='Dimensionality of the joint embedding')
    # Training configurations.
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=160, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--img_size', default=256, type=int,
                        help='image_size')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--learning_rate', default=.001, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='learning rate decay')
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')
    # Miscellaneous.
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='../runs/runX',
                        help='Path to save the model and Tensorboard log.')

    opt = parser.parse_args()
    print(opt)

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab = pickle.load(open(os.path.join(
        opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb'))
    opt.vocab_size = len(vocab)
    opt.vocab = vocab

    # Create directories
    create_directory(opt.data_path, opt.model, opt.save_dir)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.crop_size, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = VSE(opt)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            model.change_training_state(0)
            if start_epoch > 2:
                model.change_training_state(2)
            if start_epoch > 8:
                model.change_training_state(8)
            best_rsum = checkpoint['best_rsum']
            model.optimizer = checkpoint['optim']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    start_epoch = 0
    for epoch in range(start_epoch, opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        model.change_training_state(epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
            'optim': model.optimizer,
            },
            is_best,
            prefix=os.path.join(opt.data_path,
                                opt.model,
                                opt.save_dir))


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(
        model, val_loader, opt.log_step, logging.info)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs)
    logging.info("Image to text: R@1:%.1f, R@5:%.1f, R@10:%.1f, median rank:%.1f, mean rank:%.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs)
    logging.info("Text to image:R@1:%.1f, R@5:%.1f, R@10:%.1f, median rank:%.1f, mean rank:%.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('r1', r1, step=model.Eiters)
    tb_logger.log_value('r5', r5, step=model.Eiters)
    tb_logger.log_value('r10', r10, step=model.Eiters)
    tb_logger.log_value('medr', medr, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('r1i', r1i, step=model.Eiters)
    tb_logger.log_value('r5i', r5i, step=model.Eiters)
    tb_logger.log_value('r10i', r10i, step=model.Eiters)
    tb_logger.log_value('medri', medri, step=model.Eiters)
    tb_logger.log_value('meanr', meanr, step=model.Eiters)
    tb_logger.log_value('rsum', currscore, step=model.Eiters)

    return currscore


def create_directory(data_path, model, save_dir):
    path = os.path.join(data_path, model, save_dir, 'result')
    if not os.path.exists(path):
        os.makedirs(path)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, os.path.join(prefix, filename))
    if is_best:
        shutil.copyfile(os.path.join(prefix, filename),
                        os.path.join(prefix, 'model_best.pth.tar'))
    print('Save checkpoint into {}...'.format(prefix))


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 2 every 1 epochs until 7"""
    if (epoch + 1) in [2, 3, 4, 5, 6]:
        lr = opt.learning_rate * opt.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Decayed learning rate by a factor {} to {}'.format(opt.lr_decay, lr))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(name)
    print(model)
    print("The number of parameters: {}".format(num_params))

def str2bool(v):
    return v.lower() in ('true')


if __name__ == '__main__':
    main()
