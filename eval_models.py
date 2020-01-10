"""
Evaluate pre-trained models on masked datasets
"""

import logging
import sys

import numpy as np
import torch
import torchvision.models as models
from torchvision import datasets, transforms
from tqdm import tqdm


def run(model, image_net_path, input_size, device, batch_size, num_workers):
    # Create the data_loader
    logger.info('Using path: {}'.format(image_net_path))

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(image_net_path, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    buffer = []

    logger.info('data_loader length is {}'.format(data_loader.__len__()))

    for idx, (data, target) in tqdm(enumerate(data_loader), total=data_loader.__len__()):
        data = data.to(device)

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        eval = init_pred.cpu().numpy().ravel() == target.numpy().ravel()
        buffer.extend(list(eval))

    logger.info('Top-1 Accuracy: {}'.format(np.mean(buffer)))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger = logging.getLogger(__name__)
    logger.info('Starting experiment')

    image_net_paths = sys.argv[1:]
    input_size = 224
    num_workers = 10
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_dict = {
        'inceptionv3': models.inception_v3,
        'densenet121': models.densenet121,
        'vgg16': models.vgg16,
        'resnet50': models.resnet50
    }

    list_models = [
        'inceptionv3',
        'densenet121',
        'vgg16',
        'resnet50'
    ]

    logger.info('Paths: {}'.format(image_net_paths))
    logger.info('Models: {}'.format(list_models))

    for model_ in list_models:
        logger.info('==================================')
        logger.info('Evaluating with {}'.format(model_))
        model = models_dict[model_](pretrained=True)
        model.eval()
        model = model.to(device)

        for image_net_path in image_net_paths:
            run(
                model=model,
                image_net_path=image_net_path,
                input_size=input_size,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device
            )
