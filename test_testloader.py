import torch.utils.data

import data_utils as utils
from torchvision import transforms

transform = transforms.Compose(

         [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225])
         ])


test_dataset = utils.SteelTestSet(root_dir='test_images',nr_classes=4,transform=transform)




test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=20)


print(len(test_loader))

print(iter(test_loader).next().shape)