import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
import torchvision.models as models
import torchvision
import numpy as np


def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            img = img.cpu().numpy()
            ax.imshow((img * 255).astype(np.uint8))
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])

    fig.savefig('./outpic.jpg',bbox_inches = 'tight')
    plt.show()
    # return axes

def multi_predict(row,col,net):
    num = row * col
    data_transform = transforms.Compose([
        transforms.CenterCrop(120),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.99302036, 0.99302036, 0.99302036], std=[0.05381432, 0.05381432, 0.05381432])
    ])

    test_dataset = torchvision.datasets.ImageFolder(root='/home/kemosheng/code/zxx/pili_learn/data/test',
                                                    transform=data_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=num,shuffle=True)

    json_path = './class_indices.json'
    assert os.path.exists(json_path)
    with open(json_path,'r') as f:
        class_indict = json.load(f)    

    with torch.no_grad():
        for x,y in test_loader:
            x = x.cuda()
            y = y.cuda()        #### 此处十分重要
            break
        trues = [class_indict[str(name)] for name in y.cpu().numpy()]
        outcome = net(x).argmax(axis=1)
        preds = [class_indict[str(name2)] for name2 in outcome.cpu().numpy()]
        titles = [true + '\n' + pred for true,pred in zip(trues,preds)]

        a23 = x.permute(0,2,3,1)

    show_images(a23[0:num],row,col,titles=titles[0:num])

    

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_transform = transforms.Compose([
        transforms.CenterCrop(120),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.99302036, 0.99302036, 0.99302036], std=[0.05381432, 0.05381432, 0.05381432])
    ])

    #load image
    img_path = args.img_path

    assert os.path.exists(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    img_t = data_transform(img)

    img_t = torch.unsqueeze(img_t,dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path)

    with open(json_path,'r') as f:
        class_indict = json.load(f)

    model = models.resnet101()

    in_channel = model.fc.in_features
    out_channel = 24
    model.fc = nn.Linear(in_channel,out_channel)
    model.to(device)

    weights_path = args.weights
    assert os.path.exists(weights_path)

    weights_dict = torch.load(weights_path,map_location=device)
    load_weight = {k:v for k,v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(load_weight,strict=False)


    # start to predict
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img_t.to(device))).cpu()
        predict = torch.softmax(output,dim=0)
        predict_class = torch.argmax(predict).numpy()

    print_res = 'class : {} probability: {:.3}'.format(class_indict[str(predict_class)],
                                                        predict[predict_class].numpy())
    plt.title(print_res)

    val,indice = torch.topk(predict,5,0)

    for step,i in enumerate(indice):
        print('class : {:10} probability: {:.3}'.format(class_indict[str(int(i))],
                                                        val[step].numpy()))

    # plt.show()
    multi_predict(1,8,net=model)
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-w','--weights',type=str)
    parser.add_argument('-im','--img_path',type=str)

    opt = parser.parse_args()

    main(opt)