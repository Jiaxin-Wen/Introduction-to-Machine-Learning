from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoConfig
import torch
from torch.nn.utils.rnn import pad_sequence
import argparse
from tqdm import tqdm
import os
import json
import pandas as pd
import csv
from sklearn.metrics import mean_squared_error
import numpy as np

class Helper():
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.add_tokens()  # set all special tokens

    def add_tokens(self):
        self.sos_token = '[SOS]' # sep of summary
        self.tokenizer.add_tokens([self.sos_token])
        self.tokenizer.add_special_tokens({"pad_token": '[PAD]'})

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_vocab_size(self):
        return len(self.tokenizer)

class ReviewDataset(Dataset):
    def __init__(self, split, path, data=None):
        super().__init__()
        with open(path, 'r') as f:
            self.data = json.load(f)    
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ori_item = self.data[idx]
        text = ori_item['summary'] + helper.sos_token + ori_item['reviewText'] # roberta用的是bpe, 不会拆开sos token

        item = helper.tokenizer(text, truncation=True, max_length=256)
        item['input_ids'] = torch.tensor(item['input_ids'], dtype=torch.long)
        item['attention_mask'] = torch.tensor(item['attention_mask'], dtype=torch.float)
        if 'overall' in ori_item:
            label = float(ori_item['overall']) - 1
            item['label'] = torch.tensor(label, dtype=torch.long)
        return item


def pad_collate(batch):
    res = {}
    res['input_ids'] = pad_sequence([x['input_ids'] for x in batch], batch_first=True,
                                    padding_value=helper.tokenizer.pad_token_id)
    res['attention_mask'] = pad_sequence([x['attention_mask'] for x in batch], batch_first=True,
                                         padding_value=0)
    if 'label' in batch[0]:
        res['labels'] = torch.stack([x['label'] for x in batch], dim=0)
    return res

class Roberta(pl.LightningModule):
    def __init__(self, lr=None, load_dir=None):
        super().__init__()

        self.lr = lr

        self.save_hyperparameters()

        if load_dir is None: # 加载预训练参数
            self.model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
        else: # 只加载config, 
            config = AutoConfig.from_pretrained('roberta-base')
            config.num_labels=5
            self.model = RobertaForSequenceClassification(config)

        self.model.resize_token_embeddings(helper.get_vocab_size())

        print('num_labels = ', self.model.num_labels)

    def forward(self, **kargs):
        return self.model(**kargs)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        # token_type_ids = batch['token_type_ids']
        labels = batch['labels']
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits, loss = output.logits, output.loss
        preds = logits.argmax(-1)
        acc = (preds == labels).float().mean().item()

        self.log('train_loss', loss.item())
        self.log('train_acc', acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits, loss = output.logits, output.loss
        preds = logits.argmax(-1)
        acc = (preds == labels).float().mean().item()
        self.log('val_loss', loss.item())
        self.log('val_acc', acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits, loss = output.logits, output.loss
        preds = logits.argmax(-1)
        acc = (preds == labels).float().mean().item()
        self.log('test_loss', loss.item())
        self.log('test_acc', acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def train(args):
    train_set = ReviewDataset('train', args.train_set)
    valid_set = ReviewDataset('valid', args.valid_set)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=pad_collate)
    valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=pad_collate)
    print('train_size = ', len(train_set))
    print('valid_size = ', len(valid_set))
    model = Roberta(args.lr, args.load_dir)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', verbose=True)
    earlystop_callback = EarlyStopping(monitor='val_loss', verbose=True, mode='min')
    trainer = pl.Trainer(gpus=[int(args.gpus)], max_epochs=args.max_epochs,
                         callbacks=[checkpoint_callback, earlystop_callback],
                         default_root_dir=args.save_dir, )
    trainer.fit(model, train_dataloader, valid_dataloader)


def test(args):
    test_set = ReviewDataset("test", args.test_set)
    print('test_size = ', len(test_set))
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=pad_collate)
    device = torch.device(f"cuda:{args.gpus}")
    model = Roberta.load_from_checkpoint(args.load_dir)
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).tolist()

            all_preds += preds
            all_labels += labels

    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    print('rmse = ', rmse)


def predict(args):
    test_set = ReviewDataset("test", args.test_set)
    print('test_size = ', len(test_set))
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=pad_collate)
    device = torch.device(f"cuda:{args.gpus}")
    model = Roberta.load_from_checkpoint(args.load_dir)
    model.to(device)
    model.eval()


    with open(args.predict_out_path, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Id', 'Predicted'])
            all_probs = []
            cnt = 1
            with torch.no_grad():
                for i, batch in tqdm(enumerate(test_dataloader)):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    index = torch.arange(5, device=probs.device) + 1
                    index = index.unsqueeze(0).repeat(probs.size(0), 1)
                    preds = (probs * index).sum(dim=-1)
                    probs = probs.tolist()
                    all_probs += probs
                    for pred in preds:
                        writer.writerow([cnt, float(pred)])
                        cnt += 1

    with open(args.predict_out_path[:-4]+'.json', 'w') as f:  
        json.dump(all_probs, f)

def parse():
    parser = argparse.ArgumentParser(description='finetune bert')

    parser.add_argument('--train_set', type=str, 
                        help='Path of training set')
    parser.add_argument('--valid_set', type=str,
                        help='Path of validation set')
    parser.add_argument('--test_set', type=str,
                        help='Path of test set')

    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of cpu cores to use')
    parser.add_argument('--gpus', default=None, help='Gpus to use')

    parser.add_argument('--seed', type=int, required=False, default=20210509,
                        help='random seed')

    parser.add_argument('--lr', type=float, default=3e-5,
                    help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of mini-batch')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle data')
                        

    parser.add_argument('--load_dir', type=str, default=None, # default值是none
                        help='Directory of checkpoint to load for predicting')
    parser.add_argument('--save_dir', type=str,
                        help='Path to save model')
    parser.add_argument('--predict_out_path', type=str,
                    help='Path of prediction file')

    parser.add_argument('--predict', action='store_true',
                    help='predict result')
    parser.add_argument('--test', action='store_true',
                help='test')

    return parser.parse_args()


helper = Helper()

if __name__ == '__main__':


    args = parse()
    from config import Config
    configs = Config(args=args, file=__file__)
    configs.show()
    args = configs

    # pl.seed_everything(20210509)
    pl.seed_everything(args.seed)

    if args.predict:
        predict(args)
    elif args.test:
        test(args)
    else:
        train(args)