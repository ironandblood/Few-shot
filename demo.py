import timm
# print(timm.list_models(pretrained=True))
for m in timm.list_models("*224*", pretrained=True):
    print(m)
#%%
model = timm.create_model("vit_base_patch32_224", pretrained=True)
#%%
import torch
i = torch.ones(1, 3, 224, 224)
r = model(i)
#%%
print(r.argmax())

#%%
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
img = Image.open("KR.bmp")
# trans = transforms.ToTensor()
# t = torch.tensor(trans(img))
# print(t.shape)
t2 = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor()
])
# torch.ones().numpy()
plt.imshow(t2(img).permute(1, 2, 0).numpy())
t = t2(img).unsqueeze(0)
print(t.shape)
#%%
res = model(t)
print(res.argmax())

#%%
import json
with open("imagenet_class_index.json", "r") as f:
    label = json.load(f)

#%%
print(label["403"])
#%%

train_transforms = transforms.Compose([
    # 旋转角度为10度
    transforms.RandomRotation(10),
    # 根据提供的最小/最大缩放比例随机缩放图像大小
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    # 随机水平翻转图像
    transforms.RandomHorizontalFlip(),
    # 随机垂直翻转图像
    transforms.RandomVerticalFlip(),
    # 填充图像边界,新边界为图像大小的4%
    transforms.Pad(4),
    # 在给定范围内随机调整图像的亮度,对比度和饱和度
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor(),
    # 归一化到[0,1]之间
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
])
test = Image.open("bobby.jpg")

# torch.ones(1).numpy()
# r = (train_transforms(test).permute(1,2,0).numpy())
# print(type(r))
# plt.imshow(r)
plt.imshow(test)
test

#%%
