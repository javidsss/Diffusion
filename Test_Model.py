import torch
from Model import Unet
from Utils import plot_timestep_img, continue_training_func
import os


def main():
    ## Training
    model = Unet(in_channels=1, out_channels=1, time_emb_dim=32).to(device1)
    continue_training_func(model, model_name, save_dir_init) #Loads the model

    ## Res plotter
    plotter = plot_timestep_img(img_size, T, model, num_images_show, in_channels=1, device1='cpu', device2='cpu')
    with torch.no_grad():
        plotter.sample_plot_image()


if __name__ == '__main__':

    train_path = '/Users/javidabderezaei/Documents/CodesGit/Diffusion_Dataset_Models/stanford_cars/cars_train/'
    save_dir_init = '/Users/javidabderezaei/Documents/CodesGit/Diffusion_Dataset_Models/'

    if not os.path.exists(train_path):
        train_path = '/gscratch/kurtlab/Diffusion/Data/stanford_cars/cars_train/'
        save_dir_init = '/gscratch/kurtlab/Diffusion/'

    model_name = "Diff_Basic_Stanford/"

    ## Hyperparameters
    img_size = 64
    batch_size = 32
    device1 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    T = 300
    num_images_show = 6
    max_epoch = 1000
    lr = 0.001

    main()
