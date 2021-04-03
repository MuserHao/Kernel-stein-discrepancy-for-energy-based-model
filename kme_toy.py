import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.optim as optim
import networks
import argparse
import utils
import toy_data
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from visualize_flow import visualize_transform
from utils import CHMCSampler
from sklearn.metrics import mutual_info_score
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

def keep_grad(output, input, grad_outputs=None):
    return torch.autograd.grad(output, input, grad_outputs=grad_outputs, retain_graph=True, create_graph=True)[0]


def sample_data(args, batch_size):
    x = toy_data.inf_train_gen(args.data, batch_size=batch_size)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    return x


def approx_jacobian_trace(fx, x):
    eps = torch.randn_like(fx)
    eps_dfdx = keep_grad(fx, x, grad_outputs=eps)
    tr_dfdx = (eps_dfdx * eps).sum(-1)
    return tr_dfdx


def exact_jacobian_trace(fx, x):
    vals = []
    for i in range(x.size(1)):
        fxi = fx[:, i]
        dfxi_dxi = keep_grad(fxi.sum(), x)[:, i][:, None]
        vals.append(dfxi_dxi)
    vals = torch.cat(vals, dim=1)
    return vals.sum(dim=1)


class EBM(nn.Module):
    def __init__(self, net, base_dist=None):
        super(EBM, self).__init__()
        self.net = net
        if base_dist is not None:
            self.base_mu = nn.Parameter(base_dist.loc)
            self.base_logstd = nn.Parameter(base_dist.scale.log())
        else:
            self.base_mu = None
            self.base_logstd = None

    def forward(self, x):
        if self.base_mu is None:
            bd = 0
        else:
            base_dist = distributions.Normal(self.base_mu, self.base_logstd.exp())
            bd = base_dist.log_prob(x).view(x.size(0), -1).sum(1)
        return self.net(x) + bd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data',
        choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', 'mixture',
                 '2spirals', 'checkerboard', 'rings', 'exponential', 'crossrelatedexp', 'correlatedexp'],
        type=str, default='moons'
    )
    parser.add_argument('--niters', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--test_batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--critic_weight_decay', type=float, default=0)
    parser.add_argument('--save', type=str, default='tmp/simulation')
    parser.add_argument('--mode', type=str, default="sm", choices=['sm', 'kme'])
    parser.add_argument('--viz_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=5000)
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--base_dist', action="store_true")
    parser.add_argument('--c_iters', type=int, default=5)
    parser.add_argument('--l2', type=float, default=10.)
    parser.add_argument('--exact_trace', action="store_true")
    parser.add_argument('--n_steps', type=int, default=10)
    args = parser.parse_args()

    # logger
    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    # neural netz
    net = networks.SmallMLP(4)
    # net = networks.LargeMLP(4)


    ebm = EBM(net)
    ebm.to(device)

    # for sampling
    logger.info(ebm)
    epsilon = 0.3

    # optimizers
    optimizer = optim.Adam(ebm.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(.0, .999))

    time_meter = utils.RunningAverageMeter(0.98)
    loss_meter = utils.RunningAverageMeter(0.98)

    ebm.train()
    end = time.time()
    for itr in range(args.niters):

        optimizer.zero_grad()

        x = sample_data(args, args.batch_size)
        x.requires_grad_()

        if args.mode == "sm":
            # score matching for reference
            fx = ebm(x)
            dfdx = torch.autograd.grad(fx.sum(), x, retain_graph=True, create_graph=True)[0]
            eps = torch.randn_like(dfdx)  # use hutchinson here as well
            epsH = torch.autograd.grad(dfdx, x, grad_outputs=eps, create_graph=True, retain_graph=True)[0]

            trH = (epsH * eps).sum(1)
            norm_s = (dfdx * dfdx).sum(1)

            loss = (trH + .5 * norm_s).mean()
            loss.backward()
            optimizer.step()

        elif args.mode == 'kme':
            # kernel mean embedding:
            fx = ebm(x)
            dfdy = torch.autograd.grad(fx.sum(), x, retain_graph=True, create_graph=True)[0]
            dfdy = dfdy[:, 0:2]
            eps = torch.randn_like(dfdy)  # use hutchinson here as well
            kernel = lambda d: torch.exp(-d**2/1)     # Determine kernel sigma here!
            # kernel = lambda d: torch.exp(-abs(d)/1)     # Determine kernel sigma here!
            gram = torch.mm(x[:, 2:4], torch.ones(x[:, 2:4].shape).t()) - torch.mm(torch.ones(x[:, 2:4].shape), x[:, 2:4].t())
            K = kernel(gram) + gram.shape[0] * 0.1 * torch.eye(gram.shape[0])
            # beta = torch.mm(torch.inverse(K), gram)
            beta = torch.inverse(K)

            epsH = torch.autograd.grad(dfdy, x, grad_outputs=eps, create_graph=True, retain_graph=True, allow_unused=True)[0]
            epsH = epsH[:, 0:2]
            trH = (torch.mm(torch.sigmoid(beta), epsH * eps)).sum(1)
            norm_s = (torch.mm(torch.sigmoid(beta), dfdy * dfdy)).sum(1)
            # Add a penalty on the range of fx function
            loss = (trH + .5 * norm_s).mean() + 1.0*abs(fx).mean()
            loss.backward()
            optimizer.step()

        else:
            assert False

        loss_meter.update(loss.item())
        time_meter.update(time.time() - end)

        if itr % args.log_freq == 0:
            log_message = (
                'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.4f}({:.4f})'.format(
                    itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg
                )
            )
            logger.info(log_message)

        if itr % args.save_freq == 0 or itr == args.niters:
            ebm.cpu()
            utils.makedirs(args.save)
            torch.save({
                'args': args,
                'state_dict': ebm.state_dict(),
            }, os.path.join(args.save, 'checkpt.pth'))
            ebm.to(device)

        if itr % args.viz_freq == 0:
            # plot dat
            plt.clf()
            ebm.eval()

            npts = 100
            p_samples = toy_data.inf_train_gen(args.data, batch_size=npts ** 2)
            sampler = CHMCSampler(ebm, epsilon, 5, torch.from_numpy(p_samples[:, 2:4]).float(), device=device, scale_diag=torch.tensor(1.0))
            q_samples, epsilon = sampler.sample(args.n_steps)

            ebm.cpu()

            # plot only y
            p_samples = p_samples[:, 0:2]
            q_samples = q_samples[:, 0:2]

            # print(mutual_info_score(p_samples, q_samples))

            plt.figure(figsize=(4, 3))
            visualize_transform([p_samples, q_samples.detach().cpu().numpy()], ["data", "model"],
                                [], [], npts=npts)
            fig_filename = os.path.join(args.save, 'figs', '{:04d}.png'.format(itr))
            utils.makedirs(os.path.dirname(fig_filename))
            plt.savefig(fig_filename)
            plt.figure()
            sns.distplot(np.reshape(p_samples, (-1)), hist=False)
            sns.distplot(np.reshape(q_samples, (-1)), hist=False)
            plt.savefig(os.path.join(args.save, 'figs', 'density{:04d}.png'.format(itr)))
            plt.close()

            ebm.to(device)
            ebm.train()
        end = time.time()

    logger.info('Training has finished, can I get a yeet?')


if __name__ == "__main__":
    main()
