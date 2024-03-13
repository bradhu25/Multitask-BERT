'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
from itertools import cycle

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)
from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask
from datetime import datetime

TQDM_DISABLE=False

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MultitaskBERTConfig(SimpleNamespace):
    def __init__(self, 
                 hidden_size=768, 
                 hidden_dropout_prob=0.1, 
                 hidden_size_aug=204,
                 num_attention_heads = 12,
                 use_pals=False, 
                 **kwargs):
        super().__init__(hidden_size=hidden_size, 
                         hidden_dropout_prob=hidden_dropout_prob, 
                         hidden_size_aug=hidden_size_aug, 
                         num_attention_heads=num_attention_heads,
                         use_pals=use_pals, 
                         **kwargs)

class ProjectedAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encode = nn.Linear(config.hidden_size, config.hidden_size_aug)
        self.decode = nn.Linear(config.hidden_size_aug, config.hidden_size)
        self.attention = nn.MultiheadAttention(config.hidden_size_aug, config.num_attention_heads)
        self.config = config
        
    def forward(self, hidden_states, attention_mask=None):
        # Project down
        hidden_states = self.encode(hidden_states)
        # Apply attention
        attn_output, _ = self.attention(hidden_states, hidden_states, hidden_states, key_padding_mask=attention_mask)
        # Project back up
        hidden_states = self.decode(attn_output)
        return hidden_states

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.config = config
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        self.sentiment_classifier = nn.Linear(config.hidden_size, 5)
        self.paraphrase_classifier = nn.Linear(config.hidden_size * 2, 1)
        self.similarity_classifier = nn.Linear(config.hidden_size * 2, 1)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # PALs
        if self.config.use_pals:
            # Initialize PALs for each task
            self.pal_layers = nn.ModuleDict({
                "sentiment": ProjectedAttentionLayer(config),
                "paraphrase": ProjectedAttentionLayer(config),
                "similarity": ProjectedAttentionLayer(config)
            })

    def forward(self, input_ids, attention_mask, task_name=None):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        ### TODO
        # change to using contextual word embeddings of particular word pieces later
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs["pooler_output"]

        if self.config.use_pals and task_name:
            pooled_output = self.pal_layers[task_name](pooled_output)

        return pooled_output
 
    def predict_sentiment(self, input_ids, attention_mask, pooled_output=None):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        # If pooled_output needs to be computed (non PALs), compute it 
        if pooled_output is None:
            outputs = self.forward(input_ids, attention_mask, task_name="sentiment")
            pooled_output = outputs["pooler_output"]

        if self.config.use_pals:
            pooled_output = self.pal_layers["sentiment"](pooled_output)

        logits = self.sentiment_classifier(pooled_output)
        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2,
                           pooled_output_1=None, pooled_output_2=None):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        # If pooled_output needs to be computed (non PALs), compute it 
        if pooled_output_1 is None:
            pooled_output_1 = self.forward(input_ids_1, attention_mask_1)["pooler_output"]
        if pooled_output_2 is None:
            pooled_output_2 = self.forward(input_ids_2, attention_mask_2)["pooler_output"]

        if self.config.use_pals:
            pooled_output_1 = self.pal_layers["paraphrase"](pooled_output_1)
            pooled_output_2 = self.pal_layers["paraphrase"](pooled_output_2)

        combined_outputs = torch.cat((pooled_output_1, pooled_output_2), dim=1)
        logits = self.paraphrase_classifier(combined_outputs)
        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2,
                           pooled_output_1=None, pooled_output_2=None):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        # If pooled_output needs to be computed (non PALs), compute it 
        if pooled_output_1 is None:
            pooled_output_1 = self.forward(input_ids_1, attention_mask_1)["pooler_output"]
        if pooled_output_2 is None:
            pooled_output_2 = self.forward(input_ids_2, attention_mask_2)["pooler_output"]

        if self.config.use_pals:
            pooled_output_1 = self.pal_layers["similarity"](pooled_output_1)
            pooled_output_2 = self.pal_layers["similarity"](pooled_output_2)

        combined_outputs = torch.cat((pooled_output_1, pooled_output_2), dim=1)
        logits = self.similarity_classifier(combined_outputs)
        return logits

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    # quora data: para
    
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)

    #semeval data: sts
    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = MultitaskBERTConfig(
        hidden_dropout_prob=args.hidden_dropout_prob,
        hidden_size=768, 
        hidden_size_aug=args.hidden_size_aug,
        use_pals=args.use_pals,  # this should be a boolean provided as a command line argument
        num_labels=num_labels,
        data_dir='.',
        option=args.option
    )

    model = MultitaskBERT(config).to(device)

    # maybe alternate loss func?
    
    # sst = {'task_name': "sentiment", 'dataloader': sst_train_dataloader, 'predictor': model.predict_sentiment, 'loss_func': F.cross_entropy}
    # para = {'task_name': "paraphrase", 'dataloader': para_train_dataloader, 'predictor': model.predict_paraphrase, 'loss_func': F.binary_cross_entropy_with_logits}
    # sts = {'task_name': "similarity", 'dataloader': sts_train_dataloader, 'predictor': model.predict_similarity, 'loss_func': F.mse_loss}
    # tasks = [sst, para, sts]
    tasks = [cycle(iter(sst_train_dataloader)), cycle(iter(para_train_dataloader)), cycle(iter(sts_train_dataloader))]
    sizes = [len(sst_train_data), len(para_train_data), len(sts_train_data)]

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    with open(args.logpath, 'a') as file:
        start_time = datetime.now().strftime("%m_%d_%Y %H:%M:%S")
        file.write("Before Epochs: "+ str(start_time) + "\n")
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            num_batches = 0
            # for task in tasks:
            #     for batch in tqdm(task['dataloader'], desc=f'train-{epoch}', disable=TQDM_DISABLE):
            alpha = 1 - (0.8 * epoch/(args.epochs-1))
            probs = []
            for i in range(len(sizes)):
                probs.append(sizes[i] ** alpha)
            probs = probs / np.sum(probs)
            for step in range(args.steps_per_epoch):
                if (step+1)%10 == 0:
                    print("Step " + str(step) + " of Epoch " + str(epoch))
                task = np.random.choice(a=tasks, p=probs)
                batch = next(task)
                task_name, b_ids, b_mask, b_labels = batch['task_name'], batch['token_ids'], batch['attention_mask'], batch['labels']
                b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)

                optimizer.zero_grad()

                # Get the bert representation ONCE
                bert_output = model.bert(b_ids, attention_mask=b_mask)
                pooled_output = bert_output["pooler_output"]

                if task_name == "sentiment":
                    logits = model.predict_sentiment(b_ids, b_mask, pooled_output=pooled_output)
                    loss = F.cross_entropy(logits, b_labels, reduction='sum') / args.batch_size
                
                else:
                    input_ids_1, attention_mask_1 = batch['input_ids_1'].to(device), batch['attention_mask_1'].to(device)
                    input_ids_2, attention_mask_2 = batch['input_ids_2'].to(device), batch['attention_mask_2'].to(device)

                    pooled_output_1 = model.bert(input_ids_1, attention_mask=attention_mask_1)["pooler_output"]
                    pooled_output_2 = model.bert(input_ids_2, attention_mask=attention_mask_2)["pooler_output"]

                    logits = model.predict_similarity(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, pooled_output_1=pooled_output_1, pooled_output_2=pooled_output_2)
                    loss = F.mse_loss(logits.view(-1), b_labels.float(), reduction='mean')

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
                loss.backward()
                optimizer.step()
            
                train_loss += loss.item()
                num_batches += 1

            train_loss = train_loss / (num_batches)

            sent_train_acc, sst_train_y_pred, sst_train_sent_ids, para_train_acc, para_train_y_pred, para_train_sent_ids, sts_train_corr, sts_train_y_pred, sts_train_sent_ids = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
            sent_dev_acc, sst_dev_y_pred, sst_dev_sent_ids, para_dev_acc, para_dev_y_pred, para_dev_sent_ids,sts_dev_corr, sts_dev_y_pred, sts_dev_sent_ids = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)

            if sent_dev_acc > best_dev_acc:
                best_dev_acc = sent_dev_acc
                save_model(model, optimizer, args, config, args.filepath)

            output_message = f"Epoch {epoch}: train loss :: {train_loss :.3f}, sentiment train acc :: {sent_train_acc :.3f}, sentiment dev acc :: {sent_dev_acc :.3f}, para train acc :: {para_train_acc :.3f}, para dev acc :: {para_dev_acc :.3f}, sts train corr :: {sts_train_corr :.3f}, sts dev corr :: {sts_dev_corr :.3f}"
            print(output_message)
            file.write(output_message + "\n")

        end_time = datetime.now().strftime("%m_%d_%Y %H:%M:%S")
        file.write("After Epochs: "+ str(end_time) + "\n")
        dt_format = "%m_%d_%Y %H:%M:%S"
        start_time = datetime.strptime(start_time, dt_format)
        end_time = datetime.strptime(end_time, dt_format)
        file.write("Training Time: " + str(end_time - start_time) + "\n")

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=16)
    parser.add_argument("--steps_per_epoch", type=int, default=3000)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--hidden_size_aug', type=int, default=204)
    parser.add_argument("--use_pals", action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-{args.steps_per_epoch}-scheduled-multitask.pt' # Save path.
    args.logpath = f'{args.option}-{args.epochs}-{args.lr}-{args.steps_per_epoch}-scheduled-multitask-log.txt' # path for saving training epochs
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
