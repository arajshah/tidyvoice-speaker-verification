# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from pprint import pformat
import copy
import hashlib
from pathlib import Path

import fire
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader

import wespeaker.utils.schedulers as schedulers
from wespeaker.dataset.dataset import Dataset
from wespeaker.frontend import *
from wespeaker.models.projections import get_projection
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint, save_checkpoint
from wespeaker.utils.executor import run_epoch
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import get_logger, parse_config_or_kwargs, set_seed, \
    spk2id
from wespeaker.models.grl import GRL

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def _load_utt2lang(path: str) -> dict:
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            utt, lang = parts[0], parts[1]
            m[utt] = lang
    return m

def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get('checkpoint', None)

    lang_adv = configs.get("lang_adv", {}) or {}
    lang_enabled = bool(lang_adv.get("enabled", False))

    if lang_enabled:
        utt2lang_path = lang_adv.get("utt2lang_path", None)
        if not utt2lang_path:
            raise ValueError("lang_adv.enabled=true but lang_adv.utt2lang_path is missing")
        utt2lang_path = str(Path(utt2lang_path))
        if not Path(utt2lang_path).is_file():
            raise FileNotFoundError(f"utt2lang not found: {utt2lang_path}")

    # dist configs
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(configs['gpus'][local_rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    model_dir = os.path.join(configs['exp_dir'], "models")
    if rank == 0:
        try:
            os.makedirs(model_dir)
        except IOError:
            print("[warning] " + model_dir + " already exists !!!")
            if checkpoint is None:
                print("[error] checkpoint is null !")
                exit(1)
    dist.barrier(device_ids=[gpu])  # let the rank 0 mkdir first

    logger = get_logger(configs['exp_dir'], 'train.log')
    if world_size > 1:
        logger.info('training on multiple gpus, this gpu {}'.format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs['exp_dir']))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split('\n'):
            logger.info(line)

    # seed
    set_seed(configs['seed'] + rank)

    # train data
    train_label = configs['train_label']
    train_utt_spk_list = read_table(train_label)
    spk2id_dict = spk2id(train_utt_spk_list)

    lang2id = None
    utt2lang_map = None

    if lang_enabled:
        utt2lang_str = _load_utt2lang(utt2lang_path)
        train_utts = [row[0] for row in train_utt_spk_list]

        missing = [u for u in train_utts if u not in utt2lang_str]
        if missing:
            preview = missing[:20]
            raise ValueError(
                f"utt2lang incomplete: missing {len(missing)}/{len(train_utts)} train utts. "
                f"Examples: {preview}"
            )

        langs = sorted(set(utt2lang_str[u] for u in train_utts))
        lang2id = {l: i for i, l in enumerate(langs)}
        utt2lang_map = {u: lang2id[utt2lang_str[u]] for u in train_utts}

        lang_adv_meta = dict(lang_adv)
        lang_adv_meta["utt2lang_path"] = utt2lang_path
        lang_adv_meta["utt2lang_sha256"] = _sha256_file(utt2lang_path)
        lang_adv_meta["num_langs"] = len(lang2id)
        lang_adv_meta["lang2id"] = lang2id
        configs["lang_adv"] = lang_adv_meta

        configs.setdefault("dataset_args", {})
        configs["dataset_args"]["lang_adv"] = dict(configs["lang_adv"])
        configs["dataset_args"]["utt2lang_map"] = utt2lang_map

    if rank == 0:
        logger.info("<== Data statistics ==>")
        logger.info("train data num: {}, spk num: {}".format(
            len(train_utt_spk_list), len(spk2id_dict)))

    # dataset and dataloader
    train_dataset = Dataset(configs['data_type'],
                            configs['train_data'],
                            configs['dataset_args'],
                            spk2id_dict,
                            reverb_lmdb_file=configs.get('reverb_data', None),
                            noise_lmdb_file=configs.get('noise_data', None))
    train_dataloader = DataLoader(train_dataset, **configs['dataloader_args'])
    batch_size = configs['dataloader_args']['batch_size']
    if configs['dataset_args'].get('sample_num_per_epoch', 0) > 0:
        sample_num_per_epoch = configs['dataset_args']['sample_num_per_epoch']
    else:
        sample_num_per_epoch = len(train_utt_spk_list)
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info('epoch iteration number: {}'.format(epoch_iter))

    # model: frontend (optional) => speaker model => projection layer
    logger.info("<== Model ==>")
    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    if frontend_type != "fbank":
        frontend_args = frontend_type + "_args"
        frontend = frontend_class_dict[frontend_type](
            **configs['dataset_args'][frontend_args],
            sample_rate=configs['dataset_args']['resample_rate'])
        configs['model_args']['feat_dim'] = frontend.output_size()
        model = get_speaker_model(configs['model'])(**configs['model_args'])
        model.add_module("frontend", frontend)
    else:
        model = get_speaker_model(configs['model'])(**configs['model_args'])
    if rank == 0:
        num_params = sum(param.numel() for param in model.parameters())
        logger.info('speaker_model size: {}'.format(num_params))
    # For model_init, only frontend and speaker model are needed !!!
    if configs['model_init'] is not None:
        logger.info('Load initial model from {}'.format(configs['model_init']))
        load_checkpoint(model, configs['model_init'])
    elif checkpoint is None:
        logger.info('Train model from scratch ...')
    # projection layer
    configs['projection_args']['embed_dim'] = configs['model_args'][
        'embed_dim']
    configs['projection_args']['num_class'] = len(spk2id_dict)
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)
    if configs['data_type'] != 'feat' and configs['dataset_args'][
            'speed_perturb']:
        # diff speed is regarded as diff spk
        configs['projection_args']['num_class'] *= 3
        if configs.get('do_lm', False):
            logger.info(
                'No speed perturb while doing large margin fine-tuning')
            configs['dataset_args']['speed_perturb'] = False
    projection = get_projection(configs['projection_args'])
    model.add_module("projection", projection)
    if lang_enabled:
        embed_dim = configs["model_args"]["embed_dim"]
        num_langs = configs["lang_adv"]["num_langs"]

        grl_lambda = float(configs["lang_adv"].get("grl_lambda", 1.0))
        model.add_module("grl", GRL(lambda_=grl_lambda))
        model.add_module("lang_head", torch.nn.Linear(embed_dim, num_langs))

    if rank == 0:
        # print model
        for line in pformat(model).split('\n'):
            logger.info(line)
        # !!!IMPORTANT!!!
        # Try to export the model by script, if fails, we should refine
        # the code to satisfy the script export requirements
        if frontend_type == 'fbank':
            script_model = torch.jit.script(model)
            script_model.save(os.path.join(model_dir, 'init.zip'))

    # If specify checkpoint, load some info from checkpoint.
    # For checkpoint, frontend, speaker model, and projection layer
    # are all needed !!!
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",
                                     checkpoint)[0]) + 1
        logger.info('Load checkpoint: {}'.format(checkpoint))
    else:
        start_epoch = 1
    logger.info('start_epoch: {}'.format(start_epoch))

    # ddp_model
    model.cuda()
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    device = torch.device("cuda")

    criterion = getattr(torch.nn, configs['loss'])(**configs['loss_args'])
    if rank == 0:
        logger.info("<== Loss ==>")
        logger.info("loss criterion is: " + configs['loss'])

    configs['optimizer_args']['lr'] = configs['scheduler_args']['initial_lr']
    optimizer = getattr(torch.optim,
                        configs['optimizer'])(ddp_model.parameters(),
                                              **configs['optimizer_args'])
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = epoch_iter
    # here, we consider the batch_size 64 as the base, the learning rate will be
    # adjusted according to the batchsize and world_size used in different setup
    configs['scheduler_args']['scale_ratio'] = 1.0 * world_size * configs[
        'dataloader_args']['batch_size'] / 64
    scheduler = getattr(schedulers,
                        configs['scheduler'])(optimizer,
                                              **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # margin scheduler
    configs['margin_update']['epoch_iter'] = epoch_iter
    margin_scheduler = getattr(schedulers, configs['margin_scheduler'])(
        model=model, **configs['margin_update'])
    if rank == 0:
        logger.info("<== MarginScheduler ==>")

    # save config.yaml
    if rank == 0:
        saved_config_path = os.path.join(configs["exp_dir"], "config.yaml")
        configs_to_save = copy.deepcopy(configs)
        if "dataset_args" in configs_to_save:
            configs_to_save["dataset_args"].pop("utt2lang_map", None)
        with open(saved_config_path, "w") as fout:
            fout.write(yaml.dump(configs_to_save))

    # training
    dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ['Epoch', 'Batch', 'Lr', 'Margin', 'Loss', "Acc"]
        for line in tp.header(header, width=10, style='grid').split('\n'):
            logger.info(line)
    dist.barrier(device_ids=[gpu])  # synchronize here

    scaler = torch.cuda.amp.GradScaler(enabled=configs['enable_amp'])
    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        train_dataset.set_epoch(epoch)

        run_epoch(train_dataloader,
                  epoch_iter,
                  ddp_model,
                  criterion,
                  optimizer,
                  scheduler,
                  margin_scheduler,
                  epoch,
                  logger,
                  scaler,
                  device=device,
                  configs=configs)

        if rank == 0:
            if epoch % configs['save_epoch_interval'] == 0 or epoch > configs[
                    'num_epochs'] - configs['num_avg']:
                save_checkpoint(
                    model, os.path.join(model_dir,
                                        'model_{}.pt'.format(epoch)))

    if rank == 0:
        os.symlink('model_{}.pt'.format(configs['num_epochs']),
                   os.path.join(model_dir, 'final_model.pt'))
        logger.info(tp.bottom(len(header), width=10, style='grid'))


if __name__ == '__main__':
    fire.Fire(train)
