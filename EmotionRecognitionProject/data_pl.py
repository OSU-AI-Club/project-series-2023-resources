import torch
import torchvision
from torchvision import transforms

#data_pl stands for data preprocessings & loading

#this is for data preprocessing and loading with train data
def train_pl():
    #the transformation we will apply to the images from the FER2013 dataset
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(), # Convert image to tensor
        transforms.Normalize(0.485, 0.229) # Normalize image
    ])

    # loading the data from the directory I have stored the downloaded FER2013 dataset
    train_data = torchvision.datasets.FER2013(root='/Users/raagulsundar/EmotionRecognition/dataset', split = 'train', transform=transform)
    print(f"Length of train data: {len(train_data)}")
    # create dataloaders so that the FER2013 data can be loaded into the model we will implement
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=19, shuffle=True, num_workers=2)

    return train_loader

#this is for data preprocessing and loading with test data
#THIS CODE IS BROKEN FOR SOME REASON. IT BREAKS AT THE DATALOADER LINE OF CODE. POSSIBLY PROBLEM WITH TEST.CSV
def test_pl():
    #the transformation we will apply to the images from the FER2013 dataset
    transform = transforms.Compose([
        # transforms.Grayscale(),
        transforms.ToTensor(), # Convert image to tensor
        transforms.Normalize(0.485, 0.229) # Normalize image
    ])

    # loading the data from the directory I have stored the downloaded FER2013 dataset
    test_data = torchvision.datasets.FER2013(root='/Users/raagulsundar/EmotionRecognition/dataset', split = 'test' ,  transform=transform)
    print(f"Length of test data: {len(test_data)}")
    # create dataloaders so that the FER2013 data can be loaded into the model we will implement
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2) 

    return test_loader

if __name__ == '__main__':
    test_loader = test_pl()

    for i, (images) in enumerate(test_loader):
        if images is None:
            print(f"Images batch at {i} is None.")
        else:
            print(images.shape)
    print('Finished loading data!')
