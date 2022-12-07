import torch
from Dataloader_Stanfordcars import show_tensor_image
import matplotlib.pyplot as plt
from Noise_Schedule import get_index_val_from_list, forward_diffusion_sample
import torch.nn.functional as F
from natsort import natsorted
import os, glob
import math

class plot_timestep_img:
    def __init__(self, img_size, num_t, model, num_images_show, in_channels=1, device1='cpu', device2='cpu'):
        self.img_size = img_size
        self.T = num_t
        self.device1 = device1
        self.device2 = device2
        self.model = model
        self.in_channels = 1
        self.num_images_row = num_images_show

    def linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps)

    def sample_timestep(self, img_rand_noise, t):
        """
        Calls the model to predict the noise in the image and returns
        the denoised image.
        Applies noise to this image, if we are not in the last step yet.
        """
        torch.no_grad()

        betas = self.linear_beta_schedule(timesteps=self.T)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


        betas_t = get_index_val_from_list(betas, t, img_rand_noise.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_val_from_list(sqrt_one_minus_alphas_cumprod, t, img_rand_noise.shape)
        sqrt_recip_alphas_t = get_index_val_from_list(sqrt_recip_alphas, t, img_rand_noise.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
                img_rand_noise - betas_t * self.model(img_rand_noise, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = get_index_val_from_list(posterior_variance, t, img_rand_noise.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(img_rand_noise)
            return model_mean + torch.sqrt(posterior_variance_t) * noise


    def sample_plot_image(self):
        torch.no_grad()
        plt.figure(figsize=(10, 8))
        for rows in range(self.num_images_row):
            # Sample noise
            img = torch.randn((1, self.in_channels, self.img_size, self.img_size), device=self.device1)
            plt.axis('off')
            num_images_each_row = 8
            stepsize = int(math.ceil(self.T / num_images_each_row))

            for i in range(self.T-1, -1, -1):
                t = torch.full((1,), i, device=self.device1, dtype=torch.long)
                img = self.sample_timestep(img, t) #Estimates the image at its previous timestep!
                if i % stepsize == 0:
                    plt.subplot(self.num_images_row, num_images_each_row, rows*int(self.T/stepsize + 1) + int(i / stepsize + 1))
                    show_tensor_image(img.detach().cpu())
                    print(rows*int(self.T/stepsize + 1) + int(i / stepsize + 1))
        plt.show()


def continue_training_func(model, model_name, save_dir_init):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model_dir = save_dir_init + '/experiments/' + model_name
    AllNamesInFolder = natsorted(os.listdir(model_dir))
    for i in range(len(AllNamesInFolder) - 1, -1, -1):
        if not AllNamesInFolder[i].endswith('.pth.tar'):
            AllNamesInFolder.remove(AllNamesInFolder[i])

    best_model = torch.load(model_dir + AllNamesInFolder[-1], map_location=device)['state_dict']
    print('Model: {} loaded!'.format(AllNamesInFolder[-1]))
    model.load_state_dict(best_model)


def save_checkpoint(state, save_dir, filename, max_model_num=3):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    counter = 0
    while len(model_lists) > max_model_num:
        if model_lists[counter].endswith('.pth.tar') and not model_lists[counter].endswith('epoch99.000.pth.tar'):
            os.remove(model_lists[counter])
            model_lists = natsorted(glob.glob(save_dir + '*'))
        counter += 1


## Not really used!
def loss_noise(model, image_0, time):
    image_noisy, noise = forward_diffusion_sample(image_0, time, device1)
    noise_pred = model(image_noisy, time)
    return F.l1_loss(noise, noise_pred)