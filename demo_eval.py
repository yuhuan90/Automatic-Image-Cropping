from croppingModel import build_crop_model
from croppingDataset import setup_test_dataset
import os
import torch
import cv2
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description='Grid anchor based image cropping With Pytorch')
parser.add_argument('--dataset_root', default='dataset/test/', help='root directory path of testing images')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use CUDA to train model')
parser.add_argument('--net_path', default='weights/33_loss_0.10.pth', help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

dataset = setup_test_dataset(dataset_dir = args.dataset_root)


def test():
    for epoch in range(0,1):

        net = build_crop_model(poolsize = 8, reddim = 8, loadweight = False, model_path = None)
        net.load_state_dict(torch.load(args.net_path))
        net.eval()

        if args.cuda:
            net = torch.nn.DataParallel(net,device_ids=[0])
            cudnn.benchmark = True
            net = net.cuda()

        data_loader = data.DataLoader(dataset, args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=False)

        for id, sample in enumerate(data_loader):
            imgpath = sample['imgpath']
            image = sample['image']
            bboxes = sample['bboxes']
            resized_image = sample['resized_image']
            tbboxes = sample['tbboxes']

            roi = []

            for idx in range(0,len(tbboxes['xmin'])):
                roi.append((0, tbboxes['xmin'][idx],tbboxes['ymin'][idx],tbboxes['xmax'][idx],tbboxes['ymax'][idx]))

            if args.cuda:
                resized_image = Variable(resized_image.cuda())
                roi = Variable(torch.Tensor(roi))
            else:
                resized_image = Variable(resized_image)
                roi = Variable(roi)

            #t0 = time.time()
            out = net(resized_image,roi)
            id_out = sorted(range(len(out)), key=lambda k: out[k], reverse = True)

            image = image.cpu().numpy().squeeze(0)
            for id in range(0,4):
                top_box = bboxes[id_out[id]]
                top_box = [top_box[0].numpy()[0],top_box[1].numpy()[0],top_box[2].numpy()[0],top_box[3].numpy()[0]]
                top_crop = image[int(top_box[0]):int(top_box[2]),int(top_box[1]):int(top_box[3])]
                imgname = imgpath[0].split('/')[2]
                cv2.imwrite('dataset/testresult/'+imgname[:-4]+'crop_'+str(id+1)+imgname[-4:],top_crop[:,:,(2, 1, 0)])


            #t1 = time.time()

            #print('timer: %.4f sec.' % (t1 - t0))


if __name__ == '__main__':
    test()
