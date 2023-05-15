# 2022_10_08_16_00_23 100, 2022_10_08_12_50_34 300
# 50- 2023_03_23_16_48_50
# 20-2023_03_23_18_57_48
# 200- 2023_03_24_10_14_24
# 10 - 2023_03_24_11_47_57
# 35 - 2023_03_24_12_22_40
# 40- 2023_03_24_15_11_06
import re
import os
import argparse
import json
from models import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import DatasetList
from torchvision import transforms
from sklearn.metrics import precision_score, recall_score, f1_score
from data_utils.utils import *
from estimation import  Performance
from models import *


def main():

    parser = argparse.ArgumentParser(description='Testing Parkinson diagnose using deep learning network')
    parser.add_argument('-d', '--datasets', metavar='D', type=str, nargs='?', default='./data/testing_data',
                        help='Path of testing dataset/image')
    parser.add_argument('-m', '--model-type', metavar='M', type=str, nargs='?', default='AlexNet',
                        help='The model of deep learning network', dest='model')
    # model_Mnist_best_X128_1_1.pth.tar
    parser.add_argument('-w', '--weights', metavar='W', type=str, default='./checkpoints/alexnet/2023_05_03_00_05_35/model_AlexNet_best_X256.pth.tar',
                        help='The learned training weights', dest='weights')
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='Using CUDA device', dest='cuda')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=15,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-r', '--results', metavar='R', type=str, default='./outputs',
                        help='Saving output results')
    args = parser.parse_args()
    print(args)


    if not os.path.exists(args.results):
        os.mkdir(args.results)

    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    if args.cuda:
        torch.cuda.manual_seed(123)
        gpu_list = ','.join(str(i) for i in range(args.ngpus))
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model == 'ALEXNET' or args.model == 'AlexNet':
        model = AlexNet()
    elif args.model == 'FCN':
        model = FCN()
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(args.model))
    model.to(device)
    # model = torch.nn.DataParallel(model)

    checkpoint = torch.load(args.weights)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


    transform = transforms.Compose([transforms.ToTensor()])

    dataset = DatasetList(root=args.datasets, transform=transform)
    test_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    pred_labels = []
    target_labels = []

    testing_bar = tqdm(test_dataloader)
    for j, data in enumerate(testing_bar):
        inputs, targets = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)

        pred_data = preds.data.cpu().detach().numpy().flatten()
        target_data = targets.data.cpu().detach().numpy().flatten()
        for k in np.arange(len(inputs)):
            pred_labels.append(pred_data[k])
            target_labels.append(target_data[k])

    target_labels = np.array(target_labels)
    pred_labels = np.array(pred_labels)


    # if np.sum(np.argmax(np.bincount(target_labels)) - np.argmax(np.bincount(pred_labels))):
    #     print('Incorrect classification: %s' % (json_file_path))
    #     print(
    #         '     True label:%s, Predict label:%s' % (target_labels_dict, pred_labels_dict))
    # else:
    #     print('Correct classification: %s' % (json_file_path))
    #     print(
    #         '     True label:%s, Predict label:%s' % (target_labels_dict, pred_labels_dict))

    print(target_labels)
    print(pred_labels)
    print(target_labels - pred_labels)
    metric = Performance(target_labels, pred_labels)
    metric.roc_plot()
    metric.plot_matrix()
    acc_score = metric.accuracy()
    f1_score = metric.f1_score()
    recall_score = metric.recall()
    precision_score = metric.presision()
    specificity = metric.specificity()
    mcc = metric.mcc()
    print("Method1 (On Sequence): Accuracy(ACC) = {:f}, F1_score = {:f}, Recall(Sensitivity,TPR) = {:f}, and Precision(PPV) = {:f}, and Specificity(TNR) = {:f}, and Matthews correlation coefficient(MCC) = {:f}. \n".format(acc_score, f1_score, recall_score, precision_score, specificity, mcc))



if __name__ == '__main__':
    main()
