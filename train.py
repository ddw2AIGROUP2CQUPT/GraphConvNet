import os
import torchvision.models
from torchvision.transforms import transforms
from tqdm import tqdm
from timm.loss import LabelSmoothingCrossEntropy,SoftTargetCrossEntropy
from timm.data.mixup import Mixup
from torch.autograd import Variable
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import time
import torch
import sys
import argparse
from timm.utils import *
from Cpyramid_vig import Cpvig_ti_224_gelu
from test import ConfusionMatrix
from loss import FocalLoss,LDAMLoss,LMFLoss
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.cuda.amp import autocast, GradScaler
def convert_arg_line_to_args(arg_line):
    return arg_line.split()
parser = argparse.ArgumentParser(description="CGNN_OAI", fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args
parser.add_argument('--model_name', type=str, help="the model name",default='CGNNRes18')

# dataset
parser.add_argument('--data_path', type=str, help="the path of your train datasets",default=r'')
# training
parser.add_argument('--batch_size', type=int, help="batch size", default=16)
parser.add_argument('--total_steps', type=int, help='the total iteration number', default=10000)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--lr', type=float, help='initial learning rate', default=0.1)
parser.add_argument('--cnn_lr', type=float, help='initial learning rate', default=0.01)
parser.add_argument('--num_classes', type=int, help='num_classes', default=5)
parser.add_argument('--weight_decay', type=float, help='weight decay factor for optimization', default=1e-4)

# log and save
parser.add_argument('--checkpoint_path', type=str, help='path to a checkpoint to load', default='')
parser.add_argument('--log_directory', type=str, help='directory to save summaries',default=r'')
parser.add_argument('--log_name', type=str, help='name for log_directory', default='1')
parser.add_argument('--log_freq', type=int, help='Logging frequency in global steps', default=20)

# online eval
parser.add_argument('--do_online_eval', help='if set, perform online eval in every eval_freq steps',action='store_true',default=True)
parser.add_argument('--eval_freq', type=int, help='Online evaluation frequency in global steps', default=50)
parser.add_argument('--patience', type=int, help='patience times to adjust lr if eval acc can not be better',default=100)

#
parser.add_argument('--optim', help='the optimizer', type=str, default='sgd')
parser.add_argument('--Mixup', help='whether to use mixup', type=str,default=False)
parser.add_argument('--Ema', help='whether to use Ema', type=str,default=False)
parser.add_argument('--smoothing', type=str, default=False, help='Label smoothing (default: 0.1)')

#pretrained network
parser.add_argument('--net_type', help='the type of network', type=str, default='vgg')
parser.add_argument('--depth', help='the depth of network', type=str, default='16')

#vig network
parser.add_argument('--k', help='neighbor num', type=int, default=9)
parser.add_argument('--conv', help='graph conv layer', type=str, default='mr')
parser.add_argument('--act', help='activation layer', type=str, default='gelu')
parser.add_argument('--norm', help='batch or instance normalization', type=str, default='batch')
parser.add_argument('--bias', help='bias of conv layer True or False', default=True)
parser.add_argument('--dropout', help='dropout rate', type=float, default=0.)
parser.add_argument('--use_dilation', help='use dilated knn or not', default=True)
parser.add_argument('--epsilon', help='stochastic epsilon for gcn', type=float, default=0.2)
parser.add_argument('--use_stochastic', help='stochastic for gcn true or false', default=False)
parser.add_argument('--drop_path', help='dropout rate', type=float, default=0.0)
parser.add_argument('--blocks', help='number of basic blocks in the backbone', type=list,default=[4,3,3,3])
parser.add_argument('--channels', help='number of channels of deep features', type=list,default=[48,96,144,192])
parser.add_argument('--emb_dims', help='dimension of embeddings of last prediction', type=int,default=512)

parser.add_argument('--gpu_ids', help='the index of gpu', type=str, default='0')

if sys.argv.__len__()==2:
    argsfile_with_prefix='@'+sys.argv[1]
    args=parser.parse_args([argsfile_with_prefix])
    print(args)
else:
    args=parser.parse_args()
    print(args)
class MyScheduler:
    def __init__(self, optimizer, min_lr):
        self.optimizer = optimizer
        self.min_lr = min_lr
    def update_lr(self, ):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10

def train_model(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids 
    global arg
    arg =args
    print("total {} GPUs".format(torch.cuda.device_count()))
    random_state = 21
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    torch.set_num_threads(2)
    torch.cuda.empty_cache()

    writer = SummaryWriter(os.path.join(args.log_directory, args.log_name), flush_secs=30)
    command= 'cp ' +sys.argv[0] +' '+ os.path.join(args.log_directory, args.log_name)
    os.system(command)
    command = 'cp ' + sys.argv[1] + ' ' + os.path.join(args.log_directory, args.log_name)
    os.system(command)
    print("total %d steps, batch_size %d" % (args.total_steps, args.batch_size))

    pixel_mean_train, pixel_std_train = 0.60759664, 0.1935792
    data_transform = {
                    "train": transforms.Compose([
                            transforms.ColorJitter(brightness=.33, saturation=.33),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomAffine(degrees=(-10, 10), scale=(0.9, 1.10)),
                            transforms.Resize((224,224)), 
                            transforms.Grayscale(num_output_channels=3),
                            transforms.ToTensor(),
                            transforms.Normalize([pixel_mean_train] * 3, [pixel_std_train] * 3)]),
                    "test": transforms.Compose([
                                    transforms.Resize((224, 224)), 
                                    transforms.ToTensor(),
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.Normalize([pixel_mean_train] * 3, [pixel_std_train] * 3),
                                    ])}
 

    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_path, "train"),transform=data_transform['train'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_path, "test"),transform=data_transform['test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print(len(train_dataset))
    print(len(test_dataset))
    if args.model_name == 'Cpvig_ti_224_gelu':
        net = Cpvig_ti_224_gelu(args=args)
    else:
        net = None
    print(net)
    num_params = sum(p.numel() for p in net.parameters())
    print("== Total number of parameters: {:.2f}M".format(num_params / (1024 * 1024)))
    num_params_update = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("== Total number of learning parameters: {:.2f}M".format(num_params_update / (1024 * 1024)))
    model = net.cuda()
    model = torch.nn.DataParallel(model)
    print("== Model Initialized")
    writer.add_text('parameters', str(num_params_update))
    model_ema = None
    if args.Ema:
        print("USE EMA!")
        model_ema = ModelEma(model,decay=0.99996)
    mixup_fn = None
    if args.Mixup:
        print("USE Mixup!")
        mixup_fn = Mixup(
                mixup_alpha=0.8, cutmix_alpha=1.0,cutmix_minmax=None,
                prob=1.0, switch_prob=0.5, mode='batch',
                label_smoothing=0.1,num_classes=5)
    if args.Mixup:
        criterion_train = SoftTargetCrossEntropy().cuda()
    elif args.smoothing:
        print("USE LabelSmoothing!")
        criterion_train = LabelSmoothingCrossEntropy(smoothing=0.1).cuda()
    else:
        # criterion_train_cnn = torch.nn.CrossEntropyLoss().cuda()
        criterion_train_cnn = LDAMLoss().cuda()
    criterion_val = torch.nn.CrossEntropyLoss().cuda()
    
    base_params = list(map(id, net.cnn_backbone.parameters()))
    graph_params = filter(lambda p: id(p) not in base_params, net.parameters())

    params = [
        {"params": net.cnn_backbone.parameters(), "lr": args.cnn_lr},
        {"params": graph_params, "lr": args.lr},
    ]

    if args.optim in ["sgd", "Sgd", "SGD"]:
        optimizer = torch.optim.SGD(params,  momentum=0.9, nesterov=True,weight_decay=args.weight_decay)
    elif args.optim in ["Adam", "adam", "ADAM"]:
        optimizer = torch.optim.Adam(params, weight_decay=args.weight_decay)
    elif args.optim in ["RMSprop", "Rmsprop", "rmsprop"]:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,weight_decay=args.weight_decay, momentum=0.9)
    else:
        optimizer =None

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=args.patience,min_lr=0.0000000000001)
    global_step = 0
    best_test_acc = 0
    best_test_step = 0

    total_steps = args.total_steps
    start_time = time.time()
    duration = 0
    print('*' * 10, 'start training', '*' * 10)
    model.train()
    while global_step < total_steps:
        for _, data in enumerate(train_loader):
            optimizer.zero_grad()
            before_op_time = time.time()
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            if mixup_fn is not None:
                inputs, labels = mixup_fn(inputs, labels)
            outputs = model(inputs)
            if args.model_name == 'Cpvig_ti_224_gelu':
                loss_list = [criterion_train_cnn(o, labels) / len(outputs) for o in outputs]
                loss = sum(loss_list)
                outputs = outputs[0]+outputs[1]
            else:
                loss = criterion_train(outputs, labels)
            loss.backward()
            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)
            pred = outputs.argmax(dim=1)
            train_correct = (pred == labels).sum().cuda()
            train_acc = train_correct / labels.size(0)
            current_cnn_lr = optimizer.state_dict()['param_groups'][0]['lr'] #cnn的lr
            current_gnn_lr = optimizer.state_dict()['param_groups'][1]['lr'] #gnn的lr
            print(
                '[gobal step/total steps]: [{}/{}], gnn_lr: {:.6f}, cnn_lr: {:.6f}, loss: {:.8f}, train acc：{:.8f}'.format(global_step,total_steps,current_gnn_lr,current_cnn_lr,loss, train_acc))
            if np.isnan(loss.cpu().item()):
                print('NaN in loss occurred. Aborting training.')
                return -1
            duration += time.time() - before_op_time

            if global_step and global_step % args.log_freq == 0:
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (total_steps / global_step - 1.0) * time_sofar

                print_string = ' train_acc: {:.2f} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                print(print_string.format(train_acc, examples_per_sec, loss, time_sofar, training_time_left))

                writer.add_scalar('train_loss', loss, global_step)
                writer.add_scalar('train_acc', train_acc, global_step)
                writer.add_scalar('learning_rate', current_gnn_lr, global_step)
                writer.add_scalar('weight_decay', args.weight_decay, global_step)
                writer.flush()

            if global_step and args.do_online_eval and global_step % args.eval_freq == 0:
                time.sleep(0.1)
                test_acc,test_loss,confusion,test_gnn_acc,test_cnn_acc = test_accuracy(model, test_loader)
                writer.add_scalar('test_acc', test_acc, global_step)
                writer.add_scalar('test_gnn_acc', test_gnn_acc, global_step)
                writer.add_scalar('test_cnn_acc', test_cnn_acc, global_step)

                if test_acc > best_test_acc:
                    old_best_acc = best_test_acc
                    old_best_step = best_test_step
                    old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, 'test_acc', old_best_acc)
                    old_model_path = args.log_directory + '/' + args.log_name +'/model/' + old_best_name
                    if os.path.exists(old_model_path):
                        command = 'rm {}'.format(old_model_path)
                        os.system(command)
                    best_test_acc = test_acc
                    best_test_step = global_step
                    model_save_name = '/model-{}-best_{}_{:.5f}'.format(best_test_step, 'test_acc',best_test_acc)
                    print('New test for{},test_gnn_acc for{},test_cnn_acc for{}. Saving model: {}'.format(best_test_acc,test_gnn_acc,test_cnn_acc, model_save_name))
                    confusion.plot()
                    model_save_path = os.path.join(args.log_directory, args.log_name, 'model')
                    if not os.path.exists(model_save_path):
                        command = 'mkdir ' + model_save_path
                        os.system(command)
                    torch.save(model.state_dict(), model_save_path + model_save_name)
                scheduler.step(test_acc)
            model.train()
            global_step += 1
    print("finished")
def test_accuracy(model,test_loader):
    model.eval()      
    labels_all=[]
    preds_all=[]
    test_total = 0
    correct_gnn = 0
    correct_cnn = 0
    for _, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        output_gnn = outputs[0]
        output_cnn = outputs[1]
        test_total += labels.size(0)
        pred_gnn = output_gnn.argmax(dim=1)
        correct_gnn += (pred_gnn == labels).sum().cuda()
        pred_cnn = output_cnn.argmax(dim=1)
        correct_cnn += (pred_cnn == labels).sum().cuda()
        if args.model_name == 'Cpvig_ti_224_gelu':
            outputs = outputs[0]+outputs[1]
        pred = outputs.argmax(dim=1)
        
        labels_np = labels.cpu().numpy()
        labels = labels_np.tolist()
        labels_all.extend(labels)
        preds_cpu = pred.cpu()
        preds_np = preds_cpu.numpy()
        preds = preds_np.tolist()
        preds_all.extend(preds)
    class_indict = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4'}
    label = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(args,label)
    conf_matrix,mse=confusion.summary(labels_all,preds_all)

    print("In test: confusion matrix is:\n {}".format(conf_matrix))
    acc = 1.0*np.trace(conf_matrix)/np.sum(conf_matrix)
    print('True/Total: {}/{}'.format(np.trace(conf_matrix), np.sum(conf_matrix)))
    print('Acc: {:.3f} ABE: {:.3f}'.format(acc, mse))
    return acc*100,mse,confusion,100 * correct_gnn / test_total,100 * correct_cnn / test_total
if __name__ == '__main__':
    train_model(args)