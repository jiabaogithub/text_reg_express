import cv2
import torch
from PIL import Image
from torch.autograd import Variable

import crnn_reg.dataset as dataset
import crnn_reg.models.crnn as crnn
import crnn_reg.params as params
import crnn_reg.utils as utils


def init_crnn_model(model_path):
    # torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = False # zjb 若不关闭会报错：RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED，据说换到cuda10能正常运行，但是其他模型必须运行在cuda9环境下，所以采用了折中的方案

    # net init
    nclass = len(params.alphabet) + 1
    model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)
    if torch.cuda.is_available():
        model = model.cuda()

    # load model
    print('loading pretrained model from %s' % model_path)
    if params.multi_gpu:
        model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model


def pred(model, img_arr):
    transformer = dataset.resizeNormalize((100, 32))
    # image = Image.open(img_path).convert('L')
    image_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    image = Image.fromarray(image_arr)
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    converter = utils.strLabelConverter(params.alphabet)
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))
    return sim_pred


if __name__ == '__main__':
    # _model = init_crnn_model()
    # image_path = ""
    # img_arr
    # rs = pred(_model, img_arr)
    pass
