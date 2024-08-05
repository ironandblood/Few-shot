This is our solution for the AI course project (COMP130031.02) 'Transfer Learning' of Fudan University, 2023 Spring.

# About the project
This project is a course project of AI course, Fudan University, 2023 Spring. The project is about transfer learning, which is a research problem in machine learning that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. In this project, we focus on the transfer learning in the field of image classification to implement a few-shot learning model. 

# About our solution
We are facing two major challenges in this project:
1. **Limited data**: In the few-shot learning scenario, we only have a few labeled examples for each class. This makes it difficult to train a deep neural network from scratch.
2. **Overfitting**: The model may overfit the training data and fail to generalize to unseen data.

To address these challenges, we introduce a few techniques in our solution:
1. **Pre-trained model**: We use a strong pre-trained model, VIT, as the backbone of our few-shot learning model. The pre-trained model is trained on a large dataset and has learned rich features that can be transferred to our few-shot learning task.
2. **Data augmentation**: We apply data augmentation techniques to leverage the limited labeled examples. We use random cropping, horizontal flipping, and color jittering to generate more training samples.
3. **Knowledge Distillation**: We introduce the knowledge distillation technique to leverage large-scale unlabeled data. We use the more powerful pre-trained model with more parameters to generate pseudo-labels for the unlabeled data and train the few-shot learning model with both labeled and pseudo-labeled data.

And you can find more details in our report.

# About the code
## Installation 

Our code base is developed and tested with PyTorch 1.7.0, TorchVision 0.8.1, CUDA 10.2, and Python 3.7.

```Shell
conda create -n baseline python=3.7 -y
conda activate baseline
conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt 
```                                                                                               

## Model
Loading pre-trained weights is allowed. You can use the pre-trained model under **ImageNet-1k**, while other datasets like ImageNet-21k, CC3M, LAION, etc., are not allowed.

## Datasets
Five datasets are given in ```/remote-home/share/course23```, which include:
'10shot_cifar100_20200721','10shot_country211_20210924','10shot_food_101_20211007','10shot_oxford_iiit_pets_20211007','10shot_stanford_cars_20211007'                        

## Run
The executable pretrained models are offered by ```timm```. You can check and use the offered pretrained timm models. 
```Shell
python main.py --model $selected_model --batch-size 64 --data-path $path_to_dataset --output_dir output/$selected_model --epochs 50 --lr 1e-4 --weight-decay 0.01
```

There are three modes to execute the code.
1. Operate on individual dataset seperately. You can change ```--dataset_list``` to achieve it.
2. Operate on known datasets. The dataset which given images belong to will be offered. You can check the ```--known_data_source``` option. 
3. Operate on unknown datasets. The dataset which given image belong to will not be offered. You should predict both **datasets that images belong to** and **images' corresponding labels**. You can check the ```--unknown_data_source``` option.

After obtaining the checkpoint of certain modes, you should operate ```--test_only``` to produce a prediction json file ```pred_all.json```. The file will be produced under your output directory. 
```Shell
python main.py --model $selected_model --batch-size 64 --data-path $path_to_dataset --output_dir output/$selected_model --epochs 50 --lr 1e-4 --weight-decay 0.01 --test_only --resume /path/of/your/trained/model
```