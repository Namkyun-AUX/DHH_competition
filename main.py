import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from DHH2020_dataset import DHH2020_get_dataset, DHH2020_train, DHH2020_valid, DHH2020_test
import DHH2020_models
import torchvision.models as models
import pandas as pd

gpu = 1
step_size = 50000
gamma = 0.1
do_train = True

def train_model():
    train_model_name = 'EDecoder_net_large_11'
    trainX_0, trainY_0, validX_0, validY_0, trainX_1, trainY_1, validX_1, validY_1 = DHH2020_get_dataset.getDB('./data/train.csv')
    trainDB = DHH2020_train(trainX_1, trainY_1)
    validDB = DHH2020_valid(validX_1, validY_1)

    train_loader = torch.utils.data.DataLoader(trainDB, batch_size=100, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validDB, batch_size=100, shuffle=True)

    # model
    #model = DHH2020_models.base_net()
    #model = DHH2020_models.mid3conv_net()
    #model = DHH2020_models.mid4conv_net()
    #model = DHH2020_models.EDecoder_net()
    model = DHH2020_models.EDecoder_net_large()

    #iter_num = '520000'
    #test_model_name = 'EDecoder_net_rej3_01'
    #model.load_state_dict(torch.load('./weights/' + test_model_name + '/' + test_model_name + '_' + iter_num + '.pth'))
    model.cuda(gpu)

    # loss function
    #criterion1 = nn.CrossEntropyLoss().cuda(gpu) #nn.NLLLoss()
    criterion2 = nn.MSELoss().cuda(gpu)
    #criterion3 = nn.CrossEntropyLoss().cuda(gpu) #nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # train
    step_num = 5
    epochs = 5000000

    lr_loss_pre = 0
    lr_loss_cur = 0

    #rloss1 = []
    rloss2 = []
    #rloss3 = []
    rloss = []
    for epoch in range(epochs):

        # train loop
        train(train_loader, model, criterion2, optimizer, rloss2, rloss)

        if epoch % 1000 == 0:
            #vloss1 = []
            vloss2 = []
            #vloss3 = []
            vloss = []
            
            # valid loop
            #acc = valid(valid_loader, model, criterion2, vloss2, vloss)
            valid(valid_loader, model, criterion2, vloss2, vloss)
            
            print('### iter: {}, lr_rate: {}'.format(epoch, optimizer.param_groups[0]['lr']))
            print('training loss: {}'.format(sum(rloss2) / len(rloss2)))
            #print(sum(rloss1) / len(rloss1), sum(rloss2) / len(rloss2), sum(rloss3) / len(rloss3))

            print('valid loss: {}'.format(sum(vloss2) / len(vloss2)))
            print('-'*50)
            
            lr_loss_cur = lr_loss_cur + sum(rloss) / len(rloss)

            #rloss1 = []
            rloss2 = []
            #rloss3 = []
            rloss = []

        if epoch % 10000 == 0:
            torch.save(model.state_dict(), './weights/' + train_model_name + '/' + train_model_name + '_' + str(epoch) + '.pth')
        
        if epoch % step_size == 0 and step_num > 0:
            
            if lr_loss_pre == 0:
                lr_loss_pre = lr_loss_cur
                lr_loss_cur = 0
                continue
            
            if lr_loss_pre < lr_loss_cur:
                step_num = step_num - 1
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * gamma

            lr_loss_pre = lr_loss_cur
            lr_loss_cur = 0

        if step_num == 0:
            break

def train(train_loader, model, criterion2, optimizer, rloss2, rloss):
    # switch to train mode
    model.train()

    for data, labels in train_loader:
        data = data.cuda(gpu)
        labels = labels.cuda(gpu)

        # forward pass
        data = Variable(data)
        labels = Variable(labels)
        output = model(data)

        #loss1 = criterion1(output[0], labels[:, 0])
        loss2 = criterion2(output[0], labels[:, 0].unsqueeze(1).float())
        #loss3 = criterion3(output[2], labels[:, 0])
        loss = loss2 #+ loss2 + loss3
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #rloss1.append(loss1.item())
        rloss2.append(loss2.item())
        #rloss3.append(loss3.item())
        rloss.append(loss.item())

def valid(valid_loader, model, criterion2, vloss2, vloss):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for step, (data, labels) in enumerate(valid_loader):
            data = data.cuda(gpu)
            labels = labels.cuda(gpu)

            # calculate output
            output = model(data)

            # get loss
            #loss1 = criterion1(output[0], labels[:, 0])            
            loss2 = criterion2(output[0], labels[:, 0].unsqueeze(1).float())
            #loss3 = criterion3(output[2], labels[:, 0])
            loss = loss2 #+ loss2 + loss3

            #vloss1.append(loss1.item())
            vloss2.append(loss2.item())
            #vloss3.append(loss3.item())
            vloss.append(loss.item())

            # get accuracy
            #ans = (output[2].cpu().max(1)[1] == labels[:, 0].cpu())
            #ans = (output[1].cpu().squeeze() > 0.452) == labels[:, 0].cpu()
            #acc = acc + sum(ans)

    #return acc.item() / (step+1)

def test_model():
    iter_num_0 = '270000'
    iter_num_1 = '260000'
    test_model_name = 'EDecoder_net_large_comb'
    test_model_name_0 = 'EDecoder_net_large_00'
    test_model_name_1 = 'EDecoder_net_large_11'
    testDB = DHH2020_test('./data/test.csv')
    test_loader = torch.utils.data.DataLoader(testDB, batch_size=286)

    # model
    #model = DHH2020_models.base_net()
    #model = DHH2020_models.mid3conv_net()
    #model = DHH2020_models.mid4conv_net()
    model_0 = DHH2020_models.EDecoder_net_large()
    model_1 = DHH2020_models.EDecoder_net_large()
    model_0.load_state_dict(torch.load('./weights/' + test_model_name_0 + '/' + test_model_name_0 + '_' + iter_num_0 + '.pth'))
    model_1.load_state_dict(torch.load('./weights/' + test_model_name_1 + '/' + test_model_name_1 + '_' + iter_num_1 + '.pth'))
    model_0.cuda(gpu)
    model_1.cuda(gpu)
    model_0.eval()
    model_1.eval()

    with torch.no_grad():
        for data in test_loader:
            data = data.cuda(gpu)

            # test
            output0 = model_0(data)
            output1 = model_1(data)

            # get accuracy
            ans = (output0[0].cpu().squeeze() < output1[0].cpu().squeeze()).type(torch.uint8)
            #ans = output[0].cpu().max(1)[1]
            #anss = ans & (output[1].cpu() > 0).squeeze()
            #ans = (output.cpu() > 0.435).squeeze().type(torch.uint8)
            #ans = output.cpu().squeeze()
            print(sum(ans))

            result = pd.DataFrame(ans, columns=['action'])
            result.to_csv('./results/' + test_model_name + '_iter_' + iter_num_0 + '.csv', mode='w')

if __name__ == '__main__':
    if do_train:
        train_model()
    else:
        test_model()
