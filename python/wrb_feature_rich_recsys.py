# %%
import os
from collections import defaultdict
from mxnet import gluon, np, init, npx
from d2l import mxnet as d2l

from mxnet.gluon import nn

npx.set_np()

# %%
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip', 'e18327c48c8e8e5c23da714dd614e390d369843f')
# data_dir = d2l.download_extract('ctr')


# %%
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {
                i: {feat for feat, c in cnt.items() if c >= min_threshold}
                for i, cnt in feat_cnts.items()}
            self.feat_mapper = {
                i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                for i, feat_values in feat_mapper.items()}
            self.defaults = {
                i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array(
            (0, *np.cumsum(self.field_dims).asnumpy()[:-1]))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        feat = np.array([
            self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
            for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']


# %%
# train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))


# %%
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1)**2
        sum_of_square = np.sum(self.embedding(x)**2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x

# %%
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(train_data, shuffle=True,
                                   last_batch='rollover',
                                   batch_size=batch_size,
                                   # num_workers=d2l.get_dataloader_workers())
                                   num_workers=1)
test_iter = gluon.data.DataLoader(test_data, shuffle=False,
                                  last_batch='rollover',
                                  batch_size=batch_size,
                                  # num_workers=d2l.get_dataloader_workers())
                                  num_workers=1)

# %%
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

# %%
