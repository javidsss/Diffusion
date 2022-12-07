from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image


# def Arrange_1To1(img): ##This also works, you just need to remove the () in the transforms compose!
#     img = img * 2
#     img = img - 1
#     return img

class Arrange_1To1:
    def __call__(self, img):
        img = img * 2
        img = img - 1
        return img


class StanfordData(Dataset):
    def __init__(self, data_path, transforms=None):
        self.paths = data_path
        self.transforms = transforms

    def __getitem__(self, index):
        path = self.paths[index]
        x = Image.open(path).convert('L')

        if self.transforms:
            x = self.transforms(x)

        return x

    def __len__(self):
        return len(self.paths)

def load_transformed_dataset(train_path, IMG_SIZE):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        Arrange_1To1() # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = StanfordData(glob.glob(train_path + '*.jpg'), transforms=data_transform)
    # test = train[0] ##Can be used for debugging the dataloader!
    return train


def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))


if __name__ == '__main__':
    img_size = 64
    batch_size = 128

    train_path = '/Users/javidabderezaei/Documents/CodesGit/Diffusion_Dataset/stanford_cars/cars_train/'
    train = load_transformed_dataset(train_path, img_size)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    image = next(iter(train_loader))[0]

    print(image.shape)
    plt.imshow(image[0, :, :])
    plt.show()
