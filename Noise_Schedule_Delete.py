import torch
import torch.nn.functional as F
from Dataloader_Stanfordcars import load_transformed_dataset, show_tensor_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, img_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """

    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(img_shape) - 1))).to(t.device)

def forward_diffusion_sample(image_0, t, device1="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(image_0).to(image_0.device)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, image_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, image_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device1) * image_0.to(device1) \
           + sqrt_one_minus_alphas_cumprod_t.to(device1) * noise.to(device1), noise.to(device1)


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)
device1 = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


## Test Forward diffusion:
if __name__ == '__main__':
    img_size = 64
    batch_size = 5

    train_path = '/Users/javidabderezaei/Documents/CodesGit/Diffusion_Dataset/stanford_cars/cars_train/'
    train = load_transformed_dataset(train_path, img_size)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    image = next(iter(train_loader))[0]

    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for idx in range(0, T, stepsize):
        device1 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
        image, noise = forward_diffusion_sample(image, t, device1=device1)
        show_tensor_image(image)