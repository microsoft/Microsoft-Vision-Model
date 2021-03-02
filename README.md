# Microsoft Vision

## Installation
``pip install microsoftvision``


## Usage
Input images should be in <b>BGR</b> format of shape (3 x H x W), where H and W are expected to be at least 224.
The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

Example script:  
```
import microsoftvision
import torch

# This will load pretrained model
model = microsoftvision.models.resnet50(pretrained=True)

# Load model to CPU memory, interface is the same as torchvision
model = microsoftvision.models.resnet50(pretrained=True, map_location=torch.device('cpu')) 
```

Example of creating image embeddings:
```
import microsoftvision
from torchvision import transforms
import torch
from PIL import Image

def get_image():
    img = cv2.imread('example.jpg', cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    img = img[16:256-16, 16:256-16]
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0) # Unsqueeze only required when there's 1 image in images batch

model = microsoftvision.models.resnet50(pretrained=True)
features = model(get_image())
print(features.shape)
```
Should output
```
...
torch.Size([1, 2048])
```



## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
