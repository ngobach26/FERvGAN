from model import Generator, Discriminator
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class Solver(object):
    """Solver for training and testing StarGAN on the RaFD dataset only."""

    def __init__(self, data_loader, config):
        """Initialize configurations."""

        self.data_loader = data_loader

        # Model configurations.
        self.c_dim = config.c_dim               # Number of emotion classes (e.g., 8) in RaFD
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and optionally tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()


    def build_model(self):
        """Create a generator and a discriminator specifically for RaFD."""
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')

        self.G.to(self.device)
        self.D.to(self.device)


    def print_network(self, model, name):
        """Print out the network information."""
        num_params = sum(p.numel() for p in model.parameters())
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))


    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))


    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)


    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr


    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()


    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)


    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm - 1)**2)


    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out


    def create_labels(self, c_org, c_dim=5):
        """
        Generate target domain labels for debugging/testing in RaFD. 
        E.g., if c_dim=5, we create 5 possible one-hot vectors representing each expression.
        """
        c_trg_list = []
        for i in range(c_dim):
            # Create a batch of all "i" labels, then one-hot encode it.
            c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)
            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list


    def classification_loss(self, logit, target):
        """Compute softmax cross entropy loss for RaFD."""
        return F.cross_entropy(logit, target)


    def train(self):
        """Train StarGAN on the RaFD dataset."""

        data_loader = self.data_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, self.c_dim)

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            x_real = x_real.to(self.device)
            label_org = label_org.to(self.device)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            c_org = self.label2onehot(label_org, self.c_dim).to(self.device)
            c_trg = self.label2onehot(label_trg, self.c_dim).to(self.device)

            # ========== Train the discriminator ==========#
            out_src, out_cls = self.D(x_real)
            d_loss_real = -torch.mean(out_src)
            d_loss_cls = self.classification_loss(out_cls, label_org)

            x_fake = self.G(x_real, c_trg)
            out_src, out_cls = self.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Overall D loss.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls*d_loss_cls + self.lambda_gp*d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # ============ Train the generator ============#
            # Only update G every n_critic iterations.
            if (i+1) % self.n_critic == 0:
                x_fake = self.G(x_real, c_trg)
                out_src, out_cls = self.D(x_fake)
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = self.classification_loss(out_cls, label_trg)

                x_reconst = self.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Overall G loss.
                g_loss = g_loss_fake + self.lambda_rec*g_loss_rec + self.lambda_cls*g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec']  = g_loss_rec.item()
                loss['G/loss_cls']  = g_loss_cls.item()

            # =============== Miscellaneous ===============#
            # Print out training info.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Save sample images.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using StarGAN trained."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        data_loader = self.data_loader
        
        # Specify the classes for which to generate images
        target_classes = [2,3,6]  # Classes you want to generate images for
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):
                x_real = x_real.to(self.device)
                c_org = c_org.to(self.device)  # Ensure class labels are on the same device
                
                # Skip the batch if it does not contain source class images (e.g., class 5)
                if (c_org != (7 - 1)).any():  # Adjust for zero-based indexing
                    print(f"Skipping batch {i+1} because it does not contain source class images.")
                    continue
                
                # Generate labels for target classes
                c_trg_list = self.create_labels(c_org, self.c_dim)

                # Iterate over target classes and generate images
                for target_class in target_classes:
                    # Generate labels for the target class
                    c_trg = c_trg_list[target_class - 1]  # Adjust index since classes start from 1
                    x_fake = self.G(x_real, c_trg)  # Generate fake image for the class

                    # Save the generated fake images
                    gen_folder = os.path.join(self.result_dir, f'gen{target_class}')
                    os.makedirs(gen_folder, exist_ok=True)  # Create the folder if it doesn't exist
                    gen_save_path = os.path.join(gen_folder, f'image_{i+1}_fake.jpg')
                    save_image(self.denorm(x_fake.data.cpu()), gen_save_path, nrow=1, padding=0)
                    print(f'Saved generated images for class {target_class} into {gen_save_path}...')

                    # Concatenate the original and generated images for comparison
                    x_concat = torch.cat([self.denorm(x_real), self.denorm(x_fake)], dim=3)  # Concatenate along width

                    # Save the concatenated images for comparison
                    comparison_folder = os.path.join(self.result_dir, f'comparison{target_class}')
                    os.makedirs(comparison_folder, exist_ok=True)  # Create the folder if it doesn't exist
                    comparison_save_path = os.path.join(comparison_folder, f'image_{i+1}_comparison.jpg')
                    save_image(x_concat.data.cpu(), comparison_save_path, nrow=1, padding=0)
                    print(f'Saved comparison images for class {target_class} into {comparison_save_path}...')
    # def test(self):
    #     """Translate images using StarGAN trained."""
    #     # Load the trained generator.
    #     self.restore_model(self.test_iters)
    #     data_loader = self.data_loader
        
    #     with torch.no_grad():
    #         for i, (x_real, c_org) in enumerate(data_loader):
    #             x_real = x_real.to(self.device)
    #             c_trg_list = self.create_labels(c_org, self.c_dim)

    #             # Translate images.
    #             x_fake_list = [x_real]
    #             for c_trg in c_trg_list:
    #                 x_fake_list.append(self.G(x_real, c_trg))

    #             # Save the translated images.
    #             x_concat = torch.cat(x_fake_list, dim=3)
    #             result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
    #             save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
    #             print('Saved real and fake images into {}...'.format(result_path))                
