from Noise_Schedule import forward_diffusion_sample
import torch.nn.functional as F
import torch
from Model import Unet
from Dataloader_Stanfordcars import load_transformed_dataset
from torch.utils.data import DataLoader
from Utils import continue_training_func, save_checkpoint
import os


def main():
    ## Dataloader
    train = load_transformed_dataset(train_path, img_size)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)

    ## Training
    model = Unet(in_channels=1, out_channels=1, time_emb_dim=32).to(device1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if continue_training:
        continue_training_func(model, model_name, save_dir_init)


    for epoch in range(epoch_start, max_epoch):
        loss_epoch_sum = 0
        for step, batch in enumerate(train_loader):

            batch = batch.to(device1)
            optimizer.zero_grad()

            time = torch.randint(0, T, (batch_size,), device=device1).long()
            image_noisy, noise = forward_diffusion_sample(batch, time, T)
            noise_pred = model(image_noisy, time)  # This predicts how much noise is in this image
            loss = F.l1_loss(noise, noise_pred)

            loss.backward()
            optimizer.step()

            loss_epoch_sum += loss.item()

            if step % 10 == 0:
                print(f"Step number: {step} | Loss step: {loss.item()}")


        print(f"Epoch {epoch} | Loss epoch: {loss_epoch_sum / len(train_loader)} ")
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
              }, save_dir=save_dir_init+'experiments/'+model_name, filename='epoch{:.3f}.pth.tar'.format(epoch))


if __name__ == '__main__':

    train_path = '/Users/javidabderezaei/Documents/CodesGit/Diffusion_Dataset_models/stanford_cars/cars_train/'
    save_dir_init = '/Users/javidabderezaei/Documents/CodesGit/Diffusion_Dataset/'

    if not os.path.exists(train_path):
        train_path = '/gscratch/kurtlab/Diffusion/Data/stanford_cars/cars_train/'
        save_dir_init = '/gscratch/kurtlab/Diffusion/'

    model_name = "Diff_Basic_Stanford/"

    ## Hyperparameters
    img_size = 64
    batch_size = 32
    device1 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    T = 300
    max_epoch = 1000
    lr = 0.001

    epoch_start = 0
    continue_training = False



    main()
