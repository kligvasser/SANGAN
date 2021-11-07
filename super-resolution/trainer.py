import logging
import models
import os
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad as torch_grad, Variable
from data import get_loaders
from ast import literal_eval
from utils.recorderx import RecoderX
from utils.misc import save_image, average, mkdir, compute_psnr
from models.modules.losses import PerceptualLoss

class Trainer():
    def __init__(self, args):
        # parameters
        self.args = args
        self.print_model = True
        self.parallel = False
        
        if self.args.use_tensorboard:
            self.tb = RecoderX(log_dir=args.save_path)

        # initialize
        self._init()

    def _init_model(self):
        # initialize model
        if self.args.gen_model_config != '':
            gen_model_config = dict({}, **literal_eval(self.args.gen_model_config))
        else:
            gen_model_config = {}
        if self.args.dis_model_config != '':
            dis_model_config = dict({}, **literal_eval(self.args.dis_model_config))
        else:
            dis_model_config = {}

        g_model = models.__dict__[self.args.gen_model]
        d_model = models.__dict__[self.args.dis_model]
        self.g_model = g_model(**gen_model_config)
        self.d_model = d_model(**dis_model_config)

        # loading weights
        if self.args.gen_to_load != '':
            logging.info('\nLoading gen-model...')
            self.g_model.load_state_dict(torch.load(self.args.gen_to_load, map_location='cpu'))
        if self.args.dis_to_load != '':
            logging.info('\nLoading dis-model...')
            self.d_model.load_state_dict(torch.load(self.args.dis_to_load, map_location='cpu'))

        # to cuda
        self.g_model = self.g_model.to(self.args.device)
        self.d_model = self.d_model.to(self.args.device)

        # parallel
        if self.args.device_ids and len(self.args.device_ids) > 1:
            self.g_model = torch.nn.DataParallel(self.g_model, self.args.device_ids)
            self.d_model = torch.nn.DataParallel(self.d_model, self.args.device_ids)
            self.parallel = True

        # print model
        if self.print_model:
            logging.info(self.g_model)
            logging.info('Number of parameters in generator: {}\n'.format(sum([l.nelement() for l in self.g_model.parameters()])))
            logging.info(self.d_model)
            logging.info('Number of parameters in discriminator: {}\n'.format(sum([l.nelement() for l in self.d_model.parameters()])))
            self.print_model = False

    def _init_optim(self):
        # initialize optimizer
        self.g_optimizer = torch.optim.Adam(self.g_model.parameters(), lr=self.args.lr, betas=self.args.gen_betas)
        self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=self.args.lr, betas=self.args.dis_betas)

        # initialize scheduler
        self.g_scheduler = StepLR(self.g_optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        self.d_scheduler = StepLR(self.d_optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        # initialize criterion
        if self.args.reconstruction_weight:
            self.reconstruction = torch.nn.L1Loss().to(self.args.device)
        if self.args.perceptual_weight:
             self.perceptual = PerceptualLoss(features_to_compute=['conv5_4'], criterion=torch.nn.L1Loss()).to(self.args.device)
            
    def _init(self):
        # init parameters
        self.step = 0
        self.losses = {'D': [], 'D_r': [], 'D_gp': [], 'D_f': [], 'G': [], 'G_recon': [], 'G_perc': [], 'G_adv': [], 'psnr': []}

        # initialize model
        self._init_model()

        # initialize optimizer
        self._init_optim()

    def _save_model(self):
        # save models
        torch.save(self.g_model.state_dict(), os.path.join(self.args.save_path, '{}_e{}.pt'.format(self.args.gen_model, self.epoch + 1)))
        torch.save(self.d_model.state_dict(), os.path.join(self.args.save_path, '{}_e{}.pt'.format(self.args.dis_model, self.epoch + 1)))

    def _set_require_grads(self, model, require_grad):
        for p in model.parameters():
            p.requires_grad_(require_grad)

    def _critic_hinge_iteration(self, inputs, targets):
        # require grads
        self._set_require_grads(self.d_model, True)

        # get generated data
        generated_data = self.g_model(inputs)

        # zero grads
        self.d_optimizer.zero_grad()

        # calculate probabilities on real and generated data
        d_real = self.d_model(targets)
        d_generated = self.d_model(generated_data.detach())

        # create total loss and optimize
        loss_r = F.relu(self.args.hinge_margin - d_real).mean()
        loss_f = F.relu(self.args.hinge_margin + d_generated).mean()
        loss = loss_r + loss_f

        # get gradient penalty
        if self.args.penalty_weight:
            gradient_penalty = self._gradient_penalty(targets, generated_data)
            loss += gradient_penalty

        loss.backward()

        self.d_optimizer.step()

        # record loss
        self.losses['D'].append(loss.data.item())
        self.losses['D_r'].append(loss_r.data.item())
        self.losses['D_f'].append(loss_f.data.item())
        if self.args.penalty_weight:
            self.losses['D_gp'].append(gradient_penalty.data.item())

        # require grads
        self._set_require_grads(self.d_model, False)

    def _critic_wgan_iteration(self, inputs, targets):
        # require grads
        self._set_require_grads(self.d_model, True)

        # get generated data
        generated_data = self.g_model(inputs)

        # zero grads
        self.d_optimizer.zero_grad()

        # calculate probabilities on real and generated data
        d_real = self.d_model(targets)
        d_generated = self.d_model(generated_data.detach())

        # create total loss and optimize
        loss_r = -d_real.mean()
        loss_f = d_generated.mean()
        loss = loss_f + loss_r

        # get gradient penalty
        if self.args.penalty_weight:
            gradient_penalty = self._gradient_penalty(targets, generated_data)
            loss += gradient_penalty

        loss.backward()

        self.d_optimizer.step()

        # record loss
        self.losses['D'].append(loss.data.item())
        self.losses['D_r'].append(loss_r.data.item())
        self.losses['D_f'].append(loss_f.data.item())
        if self.args.penalty_weight:
            self.losses['D_gp'].append(gradient_penalty.data.item())

        # require grads
        self._set_require_grads(self.d_model, False)

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size(0)

        # calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(self.args.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(self.args.device)

        # calculate probability of interpolated examples
        prob_interpolated = self.d_model(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.args.device),
                               create_graph=True, retain_graph=True)[0]

        # gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = gradients.norm(p=2, dim=1)

        # return gradient penalty
        return ((gradients_norm - 1) ** 2).mean()

    def _generator_iteration(self, inputs, targets):
        # zero grads
        self.g_optimizer.zero_grad()

        # get generated data
        generated_data = self.g_model(inputs)
        loss = 0.

        # reconstruction loss
        if self.args.reconstruction_weight:
            loss_recon = self.reconstruction(generated_data, targets)
            loss += loss_recon * self.args.reconstruction_weight
            self.losses['G_recon'].append(loss_recon.data.item())

        # adversarial loss
        if self.args.adversarial_weight:
            d_generated = self.d_model(generated_data)
            loss_adv = -d_generated.mean()
            loss += loss_adv * self.args.adversarial_weight
            self.losses['G_adv'].append(loss_adv.data.item())

        # perceptual loss
        if self.args.perceptual_weight:
            loss_perc = self.perceptual(generated_data, targets)
            loss += loss_perc * self.args.perceptual_weight
            self.losses['G_perc'].append(loss_perc.data.item())

        # backward loss
        loss.backward()
        self.g_optimizer.step()

        # record loss
        self.losses['G'].append(loss.data.item())

    def _train_iteration(self, data):
        # set inputs
        inputs = data['input'].to(self.args.device)
        targets = data['target'].to(self.args.device)

        # critic iteration
        if self.args.adversarial_weight:
            if self.args.wgan:
                self._critic_wgan_iteration(inputs, targets)
            else:
                self._critic_hinge_iteration(inputs, targets)

        # only update generator every |critic_iterations| iterations
        if self.step % self.args.num_critic == 0:
            self._generator_iteration(inputs, targets)

        # logging
        if self.step % self.args.print_every == 0:
            line2print = 'Iteration {}'.format(self.step)
            if self.args.adversarial_weight:
                line2print += ', D: {:.6f}, D_r: {:.6f}, D_f: {:.6f}'.format(self.losses['D'][-1], self.losses['D_r'][-1], self.losses['D_f'][-1])
                if self.args.penalty_weight:
                    line2print += ', D_gp: {:.6f}'.format(self.losses['D_gp'][-1])
            if self.step > self.args.num_critic:
                line2print += ', G: {:.5f}'.format(self.losses['G'][-1])
                if self.args.reconstruction_weight:
                    line2print += ', G_recon: {:.6f}'.format(self.losses['G_recon'][-1])
                if self.args.perceptual_weight:
                    line2print += ', G_perc: {:.6f}'.format(self.losses['G_perc'][-1])
                if self.args.adversarial_weight:
                    line2print += ', G_adv: {:.6f}'.format(self.losses['G_adv'][-1])
            logging.info(line2print)

        # plots for tensorboard
        if self.args.use_tensorboard:
            if self.args.adversarial_weight:
                self.tb.add_scalar('data/loss_d', self.losses['D'][-1], self.step)
            if self.step > self.args.num_critic:
                self.tb.add_scalar('data/loss_g', self.losses['G'][-1], self.step)

    def _eval_iteration(self, data):
        # set inputs
        inputs = data['input'].to(self.args.device)
        targets = data['target']
        paths = data['path']

        # evaluation
        with torch.no_grad():
            outputs = self.g_model(inputs)

        # save image and compute psnr
        self._save_image(outputs, paths[0])
        psnr = compute_psnr(outputs, targets, self.args.scale)

        return psnr

    def _train_epoch(self, loader):
        self.g_model.train()
        self.d_model.train()

        # train over epochs
        for _, data in enumerate(loader):
            self._train_iteration(data)
            self.step += 1

    def _eval_epoch(self, loader):
        self.g_model.eval()
        psnrs = []

        # eval over epoch
        for _, data in enumerate(loader):
            psnr = self._eval_iteration(data)
            psnrs.append(psnr)

        # record psnr
        self.losses['psnr'].append(average(psnrs))
        logging.info('Evaluation: {:.3f}'.format(self.losses['psnr'][-1]))
        if self.args.use_tensorboard:
            self.tb.add_scalar('data/psnr', self.losses['psnr'][-1], self.epoch)

    def _save_image(self, image, path):
        directory = os.path.join(self.args.save_path, 'images', 'epoch_{}'.format(self.epoch + 1))
        save_path = os.path.join(directory, os.path.basename(path))
        mkdir(directory)
        save_image(image.data.cpu(), save_path)

    def _train(self, loaders):
        # run epoch iterations
        for self.epoch in range(self.args.epochs):
            logging.info('\nEpoch {}'.format(self.epoch + 1))

            # train
            self._train_epoch(loaders['train'])

            # scheduler
            self.g_scheduler.step(epoch=self.epoch)
            self.d_scheduler.step(epoch=self.epoch)

            # evaluation
            if ((self.epoch + 1) % self.args.eval_every == 0) or ((self.epoch + 1) == self.args.epochs):
                self._eval_epoch(loaders['eval'])
                self._save_model()

        # best score
        logging.info('Best PSNR Score: {:.2f}\n'.format(max(self.losses['psnr'])))

    def train(self):
        # get loader
        loaders = get_loaders(self.args)

        # run training
        self._train(loaders)

        # close tensorboard
        if self.args.use_tensorboard:
            self.tb.close()

    def eval(self):
        # parameters
        self.epoch = 0

        # get loader
        loaders = get_loaders(self.args)

        # evaluation
        logging.info('\nEvaluating...')
        self._eval_epoch(loaders['eval'])

        # close tensorboard
        if self.args.use_tensorboard:
            self.tb.close()