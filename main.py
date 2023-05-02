import os
import copy
import math
import random
import torch
import torch.nn as nn
import argparse
import numpy as np
import cma
import csv
from fastNLP import cache_results, Tester, DataSet
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    BertConfig,
    BertTokenizer,
)
import utils

from models.modeling_roberta import RobertaForMaskedLM
from models.modeling_bert import BertForMaskedLM
from utils import hinge_loss, brier_loss
from sklearn.metrics import f1_score

from dataloader import SST2Loader, AGNewsLoader, YelpPLoader, DBPediaLoader, RTELoader, MRPCLoader, SNLILoader, MNLILoader, IMDBLoader
from metrics import SST2Metric, AGNewsMetric, YelpPMetric, DBPediaMetric, RTEMetric, MRPCMetric, SNLIMetric, MNLIMetric

from algorithm import Ensembles, CMA_ELBO, ABC_SMC, SBI_neural
from uq360.metrics.classification_metrics import expected_calibration_error as ECE

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default='roberta-large', choices=['roberta-base', 'roberta-large','bert-base-uncased', 'bert-large-uncased'], type=str)
parser.add_argument("--n_prompt_tokens", default=50, type=int)
parser.add_argument("--intrinsic_dim", default=500, type=int)
parser.add_argument("--k_shot", default=16, type=int)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--bound", default=100, type=int)  
parser.add_argument("--sigma", default=1, type=float)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--eval_every", default=20, type=int)
parser.add_argument("--device", default='cuda:0', type=str)
parser.add_argument("--random_proj", default='normal', type=str)
parser.add_argument("--loss_type", default='ce', type=str)
parser.add_argument("--cat_or_add", default='add', type=str)
parser.add_argument("--parallel", action='store_true', help='Whether to allow parallel evaluation')
parser.add_argument(
    "--inference_framework",
    default='pt',
    type=str)


parser.add_argument("--task_name", default='sst2', choices=['sst2', 'yelpp','agnews', 'dbpedia', 'mrpc', 'snli', 'rte'], type=str)
parser.add_argument("--alg_name", default='Ensembles', choices=['Ensembles', 'CMA_ELBO', 'ABC_SMC', 'SBI_neural', 'BBT'], type=str)
parser.add_argument("--num_samples", default=100, type=int, help='Number of propmt samples')
parser.add_argument("--budget", default=300, type=int, help='Total iterations for CMA_ES algorithm')
parser.add_argument("--popsize", default=20, type=int, help='Batch size for parallel inference') 
parser.add_argument("--variance", default=50, type=float, help='Variance of prior (normal) distribution')
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

# below are free hyper-params
model_name = args.model_name
task_name = args.task_name
num_samples = args.num_samples
n_prompt_tokens = args.n_prompt_tokens
intrinsic_dim = args.intrinsic_dim
k_shot = args.k_shot
batch_size = args.batch_size
budget = args.budget
bound = args.bound
sigma = args.sigma
if args.popsize > 0:
    popsize = args.popsize
else:
    popsize = 4 + 3 * np.log(intrinsic_dim)
device = args.device
random_proj = args.random_proj
seed = args.seed
loss_type = args.loss_type
print_every = args.print_every
eval_every = args.eval_every
cat_or_add = args.cat_or_add
inference_framework = args.inference_framework

# fixed hyper-params
if cat_or_add == 'add':
    init_prompt_path = None
else:
    init_prompt_path = './nli_base_prompt.pt'

if task_name in ['sst2', 'yelpp', 'rte', 'mrpc']:
    num_labels = 2
elif task_name in ['snli', 'mnli']:
    num_labels = 3
elif task_name in ['agnews']:
    num_labels = 4
elif task_name in ['dbpedia']:
    num_labels = 14
else:
    raise ValueError

# define OOD tasks
if task_name in ['sst2', 'yelpp']:
    ood_name_list = ["rte", "imdb"]
elif task_name in ['mrpc']:
    ood_name_list = ["rte"]
elif task_name in ['snli', 'rte']:
    ood_name_list = ["mrpc", "mnli"]
elif task_name in ['agnews']:
    ood_name_list = ["mrpc"]
elif task_name in ['dbpedia']:
    ood_name_list = ["agnews"]

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class LMForwardAPI:
    def __init__(self, train_data, model_name='roberta-large', n_prompt_tokens=50, task_name='sst2',
                 loss_type='hinge', init_prompt_path=None):

        if model_name in ['roberta-base', 'roberta-large']:
            self.config = RobertaConfig.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForMaskedLM.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
                inference_framework=inference_framework,
            )
            self.model.lm_head.bias = torch.nn.parameter.Parameter(torch.zeros(self.config.vocab_size))
        elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
            self.config = BertConfig.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForMaskedLM.from_pretrained(
                model_name,
                config=self.config,
                n_prompt_tokens=n_prompt_tokens,
            )
        else:
            raise NotImplementedError

        
        if cat_or_add == 'cat':
            self.model.set_concat_prompt(True)
            if init_prompt_path is not None:
                print('Initialize prompt embedding from {}'.format(init_prompt_path))
                self.init_prompt = torch.load(init_prompt_path, map_location='cuda:0').weight.cpu().reshape(-1)
            else:
                print('Initial prompt embedding not found. Initialize to random embedding.')
                self.init_prompt = torch.rand(n_prompt_tokens * self.config.hidden_size)
        else:
            self.init_prompt = None

        self.model.to(device)
        self.model.eval()
        self.linear = torch.nn.Linear(intrinsic_dim, n_prompt_tokens * self.config.hidden_size, bias=False)
        if random_proj == 'normal':
            # calculate std for normal distribution
            if model_name in ['roberta-base', 'roberta-large']:
                embedding = self.model.roberta.get_input_embeddings().weight.clone().cpu()
            elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
                embedding = self.model.bert.get_input_embeddings().weight.clone().cpu()
            else:
                raise NotImplementedError
            mu_hat = np.mean(embedding.reshape(-1).detach().cpu().numpy())
            std_hat = np.std(embedding.reshape(-1).detach().cpu().numpy())
            temp = intrinsic_dim - std_hat * std_hat
            mu = mu_hat / temp
            std = std_hat / np.sqrt(temp)
            print('[Embedding] mu: {} | std: {} [RandProj]  mu: {} | std: {}'.format(mu_hat, std_hat, mu, std))
            for p in self.linear.parameters():
                torch.nn.init.normal_(p, mu, std)
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_dev_loss = math.inf
        self.best_prompt = None
        self.num_call = 0
        # self.save_path = save_path
        self.print_every = print_every
        self.eval_every = eval_every
        self.loss_type = loss_type
        self.train_data = train_data.copy()
        self.dev_data = dev_data.copy()
        
        if task_name == 'sst2':
            self.metric = SST2Metric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'SST2Metric'
        elif task_name == 'agnews':
            self.metric = AGNewsMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'AGNewsMetric'
        elif task_name == 'yelpp':
            self.metric = YelpPMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'YelpPMetric'
        elif task_name == 'dbpedia':
            self.metric = DBPediaMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'DBPediaMetric'
        elif task_name == 'rte':
            self.metric = RTEMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'RTEMetric'
        elif task_name == 'mrpc':
            self.metric = MRPCMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'f1'
            self.metric_name = 'MRPCMetric'
        elif task_name == 'snli':
            self.metric = SNLIMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'SNLIMetric'
        elif task_name == 'mnli':
            self.metric = MNLIMetric(target='labels', pred='logits', tokenizer=tokenizer)
            self.metric_key = 'acc'
            self.metric_name = 'MNLIMetric'
        else:
            raise NotImplementedError

        self.margin = self.metric.margin
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')

    def calc_metric(self, logits, target):
        label_map = self.metric.label_map
        converted_target = target.clone()
        for key, val in label_map.items():
            converted_target[target == key] = val
        interest_index = list(label_map.keys())
        logits = logits[:, interest_index]
        pred_softmax = nn.Softmax(dim=1)(logits)
        pred = logits.argmax(dim=-1)

        if self.metric_key == 'acc':
            perf = (pred == converted_target).sum() / len(target)
        elif self.metric_key == 'f1':
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(), pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f'[Metric] Only support [acc, f1], got {self.metric_key} instead.')

        if self.loss_type == 'hinge':
            loss = hinge_loss(logits, converted_target, margin=self.margin, reduction='sum').item() / len(target)
        elif self.loss_type == 'ce':
            loss = self.ce_loss(logits, converted_target).item()
        elif self.loss_type == 'brier':
            loss = brier_loss(logits, converted_target).item()
        elif self.loss_type == 'perf':
            loss = -1 * perf.item()
        else:
            raise KeyError(f'[Loss] Only support [hinge, ce, perf], got {self.loss_type} instead.')

        return loss, perf.item(), pred_softmax, converted_target

    def test(self, prompt_list=None, test_data=None, weights=None):
        if weights is None:
            weights = torch.ones(len(prompt_list))/len(prompt_list)
        else:
            weights = torch.from_numpy(weights.astype(np.float32))
        flag=True
        count = 0
        perf_list = []
        for prompt_embedding in prompt_list:
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + P_0
            prompt_embedding = prompt_embedding.reshape(n_prompt_tokens, -1).repeat(len(test_data['input_ids']), 1, 1)
            self.model.set_prompt_embedding(prompt_embedding)
            for k, v in test_data.items():
                test_data[k] = v.to(device)
            flag_batch=True
            test_num = v.size()[0]
            for i in range(0, test_num, batch_size):
                if i+batch_size>test_num:
                    batch_index = test_num
                else:
                    batch_index = i+batch_size
                with torch.no_grad():
                    logits = self.model(
                        input_ids=test_data['input_ids'][i:batch_index],
                        attention_mask=test_data['attention_mask'][i:batch_index],
                        mask_pos=test_data['mask_pos'][i:batch_index],
                    )['logits']
                loss, perf, y_statistic, converted_target = self.calc_metric(logits, test_data['labels'][i:batch_index])
                if flag_batch:
                    all_perf = perf*(batch_index - i)
                    all_y_statistic = y_statistic
                    converted_target_all = converted_target
                    flag_batch = False
                else:
                    all_perf += perf*(batch_index - i)
                    all_y_statistic = torch.cat((all_y_statistic, y_statistic), 0)
                    converted_target_all = torch.cat((converted_target_all, converted_target), 0)

            perf_list.append(all_perf/test_num)
            if flag:
                prob_all = all_y_statistic*weights[count]
                pred_all = all_y_statistic.argmax(dim=-1).view(-1,1)
                flag=False
            else:
                prob_all += all_y_statistic*weights[count]
                pred_all = torch.cat((pred_all, all_y_statistic.argmax(dim=-1).view(-1,1)),1)

            count += 1

        emperical_distribution = torch.zeros((pred_all.size()[0],num_labels)).cuda()
        for k in range(pred_all.size()[0]):
            emperical_distribution[k] = torch.histc(pred_all[k], bins=num_labels, min=0, max=num_labels-1)
        emperical_distribution = emperical_distribution/count

        if args.alg_name == 'ABC_SMC' or args.alg_name == 'SBI_neural':
            prob_avg = emperical_distribution
        else:
            prob_avg = prob_all

        pred = prob_avg.argmax(dim=-1)
        uncertainty_score_entropy = utils.Entropy(prob_avg)
        uncertainty_score_confidence, _ = torch.max(prob_avg,1)
        ECE_score = ECE(converted_target_all.cpu().numpy(), prob_avg.cpu().numpy(), pred.cpu().numpy(), num_bins = 10)

        index = ~(pred==converted_target_all)*1 #binary label for misclassfication task
        if self.metric_key == 'acc':
            perf = (pred == converted_target_all).sum() / len(test_data['labels'])
        elif self.metric_key == 'f1':
            perf = f1_score(converted_target_all.detach().cpu().numpy().tolist(), pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f'[Metric] Only support [acc, f1], got {self.metric_key} instead.')
        print('Testing acc on all samples:', perf_list)

        return perf.item(), index.cpu(), [uncertainty_score_entropy, 1-uncertainty_score_confidence], ECE_score


    def ood(self, prompt_list=None, ood_dataset=None, weights=None):
        if weights is None:
            weights = torch.ones(len(prompt_list)) / len(prompt_list)
        else:
            weights = torch.from_numpy(weights.astype(np.float32))
        flag = True
        count = 0
        perf_list = []
        for prompt_embedding in prompt_list:
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + p_0
            prompt_embedding = prompt_embedding.reshape(n_prompt_tokens, -1).repeat(len(ood_dataset['input_ids']), 1, 1)
            self.model.set_prompt_embedding(prompt_embedding)
            for k, v in ood_dataset.items():
                ood_dataset[k] = v.to(device)
            flag_batch = True
            test_num = v.size()[0]
            for i in range(0, test_num, batch_size):
                if i + batch_size > test_num:
                    batch_index = test_num
                else:
                    batch_index = i + batch_size
                with torch.no_grad():
                    logits = self.model(
                        input_ids=ood_dataset['input_ids'][i:batch_index],
                        attention_mask=ood_dataset['attention_mask'][i:batch_index],
                        mask_pos=ood_dataset['mask_pos'][i:batch_index],
                    )['logits']
                loss, perf, y_statistic, converted_target = self.calc_metric(logits, ood_dataset['labels'][i:batch_index])
                if flag_batch:
                    all_y_statistic = y_statistic
                    flag_batch = False
                else:
                    all_y_statistic = torch.cat((all_y_statistic, y_statistic), 0)

            if flag:
                prob_all = all_y_statistic * weights[count]
                pred_all = all_y_statistic.argmax(dim=-1).view(-1, 1)
                flag = False
            else:
                prob_all += all_y_statistic * weights[count]
                pred_all = torch.cat((pred_all, all_y_statistic.argmax(dim=-1).view(-1, 1)), 1)

            count += 1

        emperical_distribution = torch.zeros((pred_all.size()[0], num_labels)).cuda()
        for k in range(pred_all.size()[0]):
            emperical_distribution[k] = torch.histc(pred_all[k], bins=num_labels, min=0, max=num_labels - 1)
        emperical_distribution = emperical_distribution / count

        if args.alg_name == 'ABC_SMC' or args.alg_name == 'SBI_neural':
            prob_avg = emperical_distribution
        else:
            prob_avg = prob_all

        uncertainty_score_entropy = utils.Entropy(prob_avg)
        uncertainty_score_confidence, _ = torch.max(prob_avg, 1)
       

        return [uncertainty_score_entropy, 1-uncertainty_score_confidence]

    def eval(self, prompt_embedding=None, test_data=None, parallel=False):
        if parallel:
            # expand training data to a larger batch for parallel evaluation
            self.train_data['input_ids'] = train_data['input_ids'].clone().repeat(len(prompt_embedding), 1)
            self.train_data['attention_mask'] = train_data['attention_mask'].clone().repeat(len(prompt_embedding), 1)
            self.train_data['mask_pos'] = train_data['mask_pos'].clone().repeat(len(prompt_embedding))
            self.train_data['labels'] = train_data['labels'].clone().repeat(len(prompt_embedding))
        else:
            self.train_data = train_data.copy()

        self.num_call += 1
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        if test_data is None:
            bsz = len(dev_data['input_ids'])  # batch size of dev data is the original batch size of training data
        else:
            bsz = batch_size  # for test data
        tmp_prompt = copy.deepcopy(prompt_embedding)  # list or numpy.ndarray
        if isinstance(prompt_embedding, list):  # multiple queries
            pe_list = []
            for pe in prompt_embedding:
                z = torch.tensor(pe).type(torch.float32)  # z
                z = self.linear(z)  # Az
                if self.init_prompt is not None:
                    z = z + self.init_prompt  # Az + P_0
                pe_list.append(z.reshape(n_prompt_tokens, -1).repeat(bsz, 1, 1))
            prompt_embedding = torch.cat(pe_list)  # num_workers*bsz x prompt_len x dim
            assert len(prompt_embedding) == len(self.train_data['input_ids'])
        elif isinstance(prompt_embedding, np.ndarray):  # single query or None
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + P_0
            prompt_embedding = prompt_embedding.reshape(n_prompt_tokens, -1).repeat(bsz, 1, 1)
        else:
            raise ValueError(
                f'[Prompt Embedding] Only support [list, numpy.ndarray], got `{type(prompt_embedding)}` instead.'
            )
        self.model.set_prompt_embedding(prompt_embedding)

        if isinstance(test_data, DataSet):
            if prompt_embedding.shape[0] > bsz:
                raise ValueError('Provide a single prompt embedding for testing.')
            test_tester = Tester(data=test_data, model=self.model, metrics=self.metric, batch_size=batch_size,
                                 num_workers=1, device=device, use_tqdm=True)
            results = test_tester.test()
            test_acc = results[self.metric_name][self.metric_key]
            return test_acc
        else:
            for k, v in self.train_data.items():
                self.train_data[k] = v.to(device)
            with torch.no_grad():
                logits = self.model(
                    input_ids=self.train_data['input_ids'],
                    attention_mask=self.train_data['attention_mask'],
                    mask_pos=self.train_data['mask_pos'],
                )['logits']

            if parallel:  # we have multiple queries
                all_losses, all_perfs, all_y_statistic = [], [], []
                for i in range(len(logits) // bsz):
                    tmp_logits = logits[i * bsz:i * bsz + bsz]
                    tmp_target = self.train_data['labels'][i * bsz:i * bsz + bsz]
                    tmp_loss, tmp_perf, y_statistic, target = self.calc_metric(tmp_logits, tmp_target)
                    all_losses.append(tmp_loss)
                    all_perfs.append(tmp_perf)
                    all_y_statistic.append(y_statistic)
                loss = min(all_losses)
                best_sol = all_losses.index(loss)  # argmin
                perf = all_perfs[best_sol]  # corresponding performance
                tmp_prompt = tmp_prompt[best_sol]  # numpy.ndarray
                prompt_embedding = pe_list[best_sol]  # to be prepended to the input
            else:  # single query
                loss, perf, y_statistic, target = self.calc_metric(logits, self.train_data['labels'])
        

            if perf > self.best_train_perf:
                self.best_train_perf = perf

            if self.num_call % self.print_every == 0:
                print(
                    '[# API Calls {}] loss: {}. Current perf: {}. Best perf so far: {}'.format(
                        self.num_call,
                        round(float(loss), 4),
                        round(float(perf), 4),
                        round(float(self.best_train_perf), 4)))

            if self.num_call % self.eval_every == 0:
                print('********* Evaluated on dev set *********')
                if parallel:  
                    self.model.set_prompt_embedding(prompt_embedding)
                for k, v in dev_data.items():
                    dev_data[k] = v.to(device)
                with torch.no_grad():
                    logits = self.model(
                        input_ids=dev_data['input_ids'],
                        attention_mask=dev_data['attention_mask'],
                        mask_pos=dev_data['mask_pos'],
                    )['logits']
                dev_loss, dev_perf, _,_ = self.calc_metric(logits, dev_data['labels'])
                
                if dev_perf >= self.best_dev_perf:
                    self.best_dev_loss = dev_loss
                    self.best_dev_perf = dev_perf
                    self.best_prompt = copy.deepcopy(tmp_prompt)

            if parallel:
                return all_losses, all_y_statistic, target, all_perfs
            else:
                return loss, y_statistic, target


    def validation_parallel(self, prompt_list, weights):
        if weights is None:
            weights = torch.ones(len(prompt_list))/len(prompt_list)
        else:
            weights = torch.from_numpy(weights.astype(np.float32))
        flag=True
        count = 0
        for prompt_embedding in prompt_list:
            prompt_embedding = torch.tensor(prompt_embedding).type(torch.float32)  # z
            prompt_embedding = self.linear(prompt_embedding)  # Az
            if self.init_prompt is not None:
                prompt_embedding = prompt_embedding + self.init_prompt  # Az + P_0
            prompt_embedding = prompt_embedding.reshape(n_prompt_tokens, -1).repeat(len(self.dev_data['input_ids']), 1, 1)
            self.model.set_prompt_embedding(prompt_embedding)
            for k, v in self.dev_data.items():
                self.dev_data[k] = v.to(device)
            with torch.no_grad():
                logits = self.model(
                    input_ids=self.dev_data['input_ids'],
                    attention_mask=self.dev_data['attention_mask'],
                    mask_pos=self.dev_data['mask_pos'],
                )['logits']

            loss, perf, y_statistic, converted_target = self.calc_metric(logits, self.dev_data['labels'])
            if flag:
                prob_all = y_statistic*weights[count]
                pred_all = y_statistic.argmax(dim=-1).view(-1, 1)
                flag=False
            else:
                prob_all += y_statistic*weights[count]
                pred_all = torch.cat((pred_all, y_statistic.argmax(dim=-1).view(-1, 1)), 1)

            count += 1

        emperical_distribution = torch.zeros((pred_all.size()[0],num_labels)).cuda()
        for k in range(pred_all.size()[0]):
            emperical_distribution[k] = torch.histc(pred_all[k], bins=num_labels, min=0, max=num_labels-1)
        emperical_distribution = emperical_distribution/count

        if args.alg_name == 'ABC_SMC' or args.alg_name == 'SBI_neural':
            prob_avg = emperical_distribution
        else:
            prob_avg = prob_all

        pred = prob_avg.argmax(dim=-1)
        # cross-entropy loss
        loss = torch.mean(-torch.log(prob_avg[torch.arange(converted_target.size()[0]), converted_target])).item()
        if self.metric_key == 'acc':
            perf = (pred == converted_target).sum() / len(self.dev_data['labels'])
        elif self.metric_key == 'f1':
            perf = f1_score(converted_target.detach().cpu().numpy().tolist(),pred.detach().cpu().numpy().tolist())
        else:
            raise KeyError(f'[Metric] Only support [acc, f1], got {self.metric_key} instead.')
        #print('validation accuracy', perf.item())

        return loss, perf



if model_name in ['roberta-base', 'roberta-large']:
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
    tokenizer = BertTokenizer.from_pretrained(model_name)
else:
    raise NotImplementedError

cache_fn = f"caches/data_{model_name.replace('/', '-')}_{task_name}_{n_prompt_tokens}_{seed}.pt"
DataLoader = {
    'sst2': SST2Loader,
    'agnews': AGNewsLoader,
    'yelpp': YelpPLoader,
    'dbpedia': DBPediaLoader,
    'rte': RTELoader,
    'mrpc': MRPCLoader,
    'snli': SNLILoader,
    'mnli': MNLILoader,
    'imdb': IMDBLoader,
    }

print('cache_fn',cache_fn)
@cache_results(cache_fn, _refresh=False)
def get_data(task_name, ood_name_list, tokenizer):
    print(task_name)
    if task_name in ['agnews', 'yelpp', 'dbpedia', 'snli']:
        splits = ['train', 'test']
    else:  # for datasets without test set, we use dev set
        splits = ['train', 'validation']
    if args.cat_or_add == 'cat':
        data_bundle = DataLoader[task_name](tokenizer=tokenizer, n_prompt_tokens=0).my_load(splits)
    else:
        data_bundle = DataLoader[task_name](tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens).my_load(splits)

    data_bundle_ood = []
    for ood_name in ood_name_list:
        print(ood_name)
        if ood_name in ['agnews', 'yelpp', 'dbpedia', 'snli', 'imdb']:
            splits = ['train', 'test']
        else:  # for datasets without test set, we use dev set
            splits = ['train', 'validation']
        if args.cat_or_add == 'cat':
            test_dataloader = DataLoader[task_name](tokenizer=tokenizer, n_prompt_tokens=0)
            ood_dataloader = DataLoader[ood_name](tokenizer=tokenizer, n_prompt_tokens=0) 
            ood_dataloader.set_label2text(test_dataloader.get_label2text())
            data_bundle_ood.append(ood_dataloader.my_load(splits))
        else:
            test_dataloader = DataLoader[task_name](tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens)
            ood_dataloader = DataLoader[ood_name](tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens)
            ood_dataloader.set_label2text(test_dataloader.get_label2text())
            data_bundle_ood.append(ood_dataloader.my_load(splits))

    return data_bundle, data_bundle_ood


def construct_true_few_shot_data(train_data, k_shot):
    train_label_count = {}
    dev_label_count = {}
    new_train_data = DataSet()
    new_dev_data = DataSet()
    all_indices = [_ for _ in range(len(train_data))]
    np.random.shuffle(all_indices)

    for index in all_indices:
        label = train_data[index]['labels']
        if label < 0:
            continue

        if label not in train_label_count:
            train_label_count[label] = 0
        if label not in dev_label_count:
            dev_label_count[label] = 0

        if train_label_count[label] < k_shot:
            new_train_data.append(train_data[index])
            train_label_count[label] += 1
        elif dev_label_count[label] < k_shot:
            new_dev_data.append(train_data[index])
            dev_label_count[label] += 1

    new_train_data.set_input("input_ids", "attention_mask", "mask_pos")
    new_dev_data.set_input("input_ids", "attention_mask", "mask_pos")
    new_train_data.set_target("labels")
    new_dev_data.set_target("labels")
    return new_train_data, new_dev_data


data_bundle, data_bundle_ood = get_data(task_name=task_name, ood_name_list=ood_name_list, tokenizer=tokenizer)
if task_name in ['agnews', 'yelpp', 'dbpedia', 'snli']:
    train_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('test')
else:
    train_data, test_data = data_bundle.get_dataset('train'), data_bundle.get_dataset('validation')

ood_data = []
for i in range(len(ood_name_list)):
    ood_name = ood_name_list[i]
    if ood_name in ['agnews', 'yelpp', 'dbpedia', 'snli', 'imdb']:
        ood_data.append(data_bundle_ood[i].get_dataset('test'))
    else:
        ood_data.append(data_bundle_ood[i].get_dataset('validation'))

train_data, dev_data = construct_true_few_shot_data(train_data, k_shot)
for ds in [train_data, dev_data, test_data]:
    ds.set_pad_val('input_ids', tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    ds.set_pad_val('attention_mask', 0)
for ds in ood_data:
    ds.set_pad_val('input_ids', tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    ds.set_pad_val('attention_mask', 0)
print('# of train data: {}'.format(len(train_data)))
print('Example:')
print(train_data[0])
print('\n# of dev data: {}'.format(len(dev_data)))
print('Example:')
print(dev_data[0])
print('\n# of test data: {}'.format(len(test_data)))
print('Example:')
print(test_data[0])
for ood in ood_data:
    print('\n# of ood data: {}'.format(len(ood)))
    print('Example:')
    print(ood[0])


# Train, validation, test, and OOD data
train_data = {
    'input_ids': torch.tensor(train_data['input_ids'].get(list(range(len(train_data))))),
    'attention_mask': torch.tensor(train_data['attention_mask'].get(list(range(len(train_data))))),
    'mask_pos': torch.tensor(train_data['mask_pos'].get(list(range(len(train_data))))),
    'labels': torch.tensor(train_data['labels'].get(list(range(len(train_data))))),
    }
dev_data = {
    'input_ids': torch.tensor(dev_data['input_ids'].get(list(range(len(dev_data))))),
    'attention_mask': torch.tensor(dev_data['attention_mask'].get(list(range(len(dev_data))))),
    'mask_pos': torch.tensor(dev_data['mask_pos'].get(list(range(len(dev_data))))),
    'labels': torch.tensor(dev_data['labels'].get(list(range(len(dev_data))))),
}

test_data = {
    'input_ids': torch.tensor(test_data['input_ids'].get(list(range(len(test_data))))),
    'attention_mask': torch.tensor(test_data['attention_mask'].get(list(range(len(test_data))))),
    'mask_pos': torch.tensor(test_data['mask_pos'].get(list(range(len(test_data))))),
    'labels': torch.tensor(test_data['labels'].get(list(range(len(test_data))))),
}

for i in range(len(ood_data)):
    ood_data[i] = {
        'input_ids': torch.tensor(ood_data[i]['input_ids'].get(list(range(len(ood_data[i]))))),
        'attention_mask': torch.tensor(ood_data[i]['attention_mask'].get(list(range(len(ood_data[i]))))),
        'mask_pos': torch.tensor(ood_data[i]['mask_pos'].get(list(range(len(ood_data[i]))))),
        'labels': torch.tensor(ood_data[i]['labels'].get(list(range(len(ood_data[i]))))),
    }
    

model_forward_api = LMForwardAPI(
    train_data = train_data,
    model_name=model_name,
    n_prompt_tokens=n_prompt_tokens,
    task_name=task_name,
    loss_type=loss_type,
    init_prompt_path= None
)

# Sample the prompts
if args.alg_name=='Ensembles':
    sampler = Ensembles(model_forward_api, intrinsic_dim, popsize, budget,num_samples)
    sample_collections, weights = sampler.sampling()
elif args.alg_name=='CMA_ELBO':
    sampler = CMA_ELBO(model_forward_api, intrinsic_dim, num_samples, args.variance, seed, popsize, budget, bound, sigma)
    sample_collections, weights = sampler.sampling()
elif args.alg_name=='ABC_SMC':
    sampler = ABC_SMC(model_forward_api, intrinsic_dim, num_samples, args.variance, args.popsize, weighted=False)
    sample_collections, weights = sampler.sampling()
elif args.alg_name=='SBI_neural':
    sampler = SBI_neural(model_forward_api, intrinsic_dim, num_samples, args.variance, num_labels, args.popsize)
    sample_collections, weights = sampler.sampling()
elif args.alg_name=='BBT':
    cma_opts = {
        'seed': seed,
        'popsize': popsize,
        'maxiter': budget,
        'verbose': -1,
    }
    if bound > 0:
        cma_opts['bounds'] = [-1 * bound, 1 * bound]
    es = cma.CMAEvolutionStrategy(intrinsic_dim * [0], 1, inopts=cma_opts)
    while not es.stop():
        sample_collections = es.ask()
        all_loss, _, _,_ = model_forward_api.eval(sample_collections, parallel=True)
        es.tell(sample_collections, all_loss)
    sample_collections = [model_forward_api.best_prompt]
    weights = np.ones(1)
else:
    raise NotImplementedError


# Save the collection of optimized prompt embedding and sampling weights
dir_path = './pretrained_prompt/'
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
prompt_path = dir_path + '_'.join([str(model_name), str(args.task_name), str(args.alg_name), str(args.seed)]) + '_prompt.pt'
weights_path = dir_path + '_'.join([str(model_name), str(args.task_name), str(args.alg_name), str(args.seed)]) + '_weights.pt'

flag = True
for sample in sample_collections:
    sample =  torch.from_numpy(sample.astype(np.float32))
    if flag:
        sample_all = sample.view(1,-1)
        flag = False
    else:
        sample_all = torch.cat((sample_all, sample.view(1,-1)),0)
torch.save(sample_all, prompt_path)
torch.save( torch.from_numpy(weights.astype(np.float32)), weights_path)


# Evaluation on downstream tasks
sample_tensor = torch.load(prompt_path)
weights = torch.load(weights_path)
weights = weights.numpy()
sample_collections = []
for i in range(sample_tensor.size()[0]):
    sample_collections.append(sample_tensor[i].numpy())

print('Evaluate on test data...')
test_acc, index, uncertainty_test, ECE_score = model_forward_api.test(sample_collections, test_data, weights)
print('Test acc: {}'.format(test_acc))
print('ECE Score: {}'.format(ECE_score))
aurrrc_selective = utils.ROC_selective(uncertainty_test, index)
aurrrc_ood = []
for i in range(len(ood_data)):
    print('OOD dataset:', ood_name_list[i])
    uncertainty_ood = model_forward_api.ood(sample_collections, ood_data[i], weights)
    aurrrc_ood.append(utils.ROC_OOD(uncertainty_test, uncertainty_ood))

# Save results of downstream tasks
results_content = ['Test_Accuracy', 'ECE_Score', 'Selective_Classification'+'_Entropy','Selective_Classification'+'_Confidence']
for item in ood_name_list:
    results_content.append('OOD_detection_'+item+'_Entropy')
    results_content.append('OOD_detection_' + item + '_Confidence')
all_results = {'Test_Accuracy': test_acc, 'ECE_Score': ECE_score, 'Selective_Classification'+'_Entropy':aurrrc_selective[0], 'Selective_Classification'+'_Confidence':aurrrc_selective[1]}
for i,item in enumerate(ood_name_list):
    all_results['OOD_detection_'+item+'_Entropy'] = aurrrc_ood[i][0]
    all_results['OOD_detection_' + item + '_Confidence'] = aurrrc_ood[i][1]

dir_path = './results/'
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
results_path = dir_path + '_'.join([str(model_name), str(args.task_name), str(args.alg_name), str(args.seed)]) + '.csv'
try:
    with open(results_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results_content)
        writer.writeheader()
        writer.writerow(all_results)
except IOError:
    print("I/O error")