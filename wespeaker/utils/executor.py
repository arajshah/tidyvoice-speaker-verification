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

import tableprint as tp

import torch

try:
    import torchnet as tnt
except Exception:
    tnt = None

    class _AvgMeter:
        def __init__(self):
            self.s = 0.0
            self.n = 0
        def add(self, v):
            self.s += float(v)
            self.n += 1
        def value(self):
            return (self.s / self.n if self.n else 0.0,)

    class _AccMeter:
        def __init__(self):
            self.correct = 0
            self.total = 0
        def add(self, outputs, targets):
            # outputs, targets are numpy arrays
            import numpy as np
            preds = np.argmax(outputs, axis=1)
            self.correct += int((preds == targets).sum())
            self.total += int(len(targets))
        def value(self):
            return (100.0 * self.correct / self.total if self.total else 0.0,)

from wespeaker.dataset.dataset_utils import apply_cmvn, spec_aug

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x



def run_epoch(dataloader, epoch_iter, model, criterion, optimizer, scheduler,
              margin_scheduler, epoch, logger, scaler, device, configs):
    model.train()
    # By default use average pooling
    loss_meter = tnt.meter.AverageValueMeter() if tnt is not None else _AvgMeter()
    acc_meter  = tnt.meter.ClassErrorMeter(accuracy=True) if tnt is not None else _AccMeter()

    frontend_type = configs['dataset_args'].get('frontend', 'fbank')
    lang_adv_cfg = configs.get("lang_adv", {})
    if isinstance(lang_adv_cfg, bool):
        lang_adv_cfg = {"enabled": bool(lang_adv_cfg)}
    lang_adv_enabled = bool(lang_adv_cfg.get("enabled", False))
    lang_loss_weight = float(lang_adv_cfg.get("loss_weight", 0.0))

    lang_ce = None
    if lang_adv_enabled:
        lang_ce = torch.nn.CrossEntropyLoss()

    def _compute_grl_lambda(step: int, cfg: dict) -> float:
        grl = cfg.get("grl", {})
        if not isinstance(grl, dict):
            grl = {}
        schedule = str(grl.get("schedule", "constant")).lower()
        max_lambda = float(grl.get("max_lambda", cfg.get("max_lambda", 1.0)))
        if schedule == "constant":
            return max_lambda
        if schedule == "linear_warmup":
            warmup_steps = int(grl.get("warmup_steps", 10000))
            if warmup_steps <= 0:
                return max_lambda
            return max_lambda * min(1.0, step / float(warmup_steps))
        return max_lambda

    for i, batch in enumerate(tqdm(dataloader, total=epoch_iter, desc=f"epoch {epoch}", leave=False)):
        cur_iter = (epoch - 1) * epoch_iter + i
        scheduler.step(cur_iter)
        margin_scheduler.step(cur_iter)

        utts = batch['key']
        targets = batch['label']
        targets = targets.long().to(device)  # (B)
        if frontend_type == 'fbank':
            features = batch['feat']  # (B,T,F)
            features = features.float().to(device)
        else:  # 's3prl'
            wavs = batch['wav']  # (B,1,W)
            wavs = wavs.squeeze(1).float().to(device)  # (B,W)
            wavs_len = torch.LongTensor([wavs.shape[1]]).repeat(
                wavs.shape[0]).to(device)  # (B)
            with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
                features, _ = model.module.frontend(wavs, wavs_len)

        with torch.cuda.amp.autocast(enabled=configs['enable_amp']):
            # apply cmvn
            if configs['dataset_args'].get('cmvn', True):
                features = apply_cmvn(
                    features, **configs['dataset_args'].get('cmvn_args', {}))
            # spec augmentation
            if configs['dataset_args'].get('spec_aug', False):
                features = spec_aug(features,
                                    **configs['dataset_args']['spec_aug_args'])

            outputs = model(features)  # (embed_a,embed_b) in most cases
            embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
            outputs = model.module.projection(embeds, targets)
            if isinstance(outputs, tuple):
                outputs, loss = outputs
            else:
                loss = criterion(outputs, targets)
            
            if lang_adv_enabled:
                if "lang" not in batch:
                    raise KeyError("lang_adv.enabled=true but batch has no 'lang'. Did you enable lang injection in the dataset pipeline?")
                if not hasattr(model.module, "grl") or not hasattr(model.module, "lang_head"):
                    raise AttributeError("lang_adv.enabled=true but model is missing 'grl' and/or 'lang_head'. Did train.py attach them?")

                lang_targets = batch["lang"].long().to(device)

                grl_lambda = _compute_grl_lambda(cur_iter, lang_adv_cfg)
                if hasattr(model.module.grl, "set_lambda"):
                    model.module.grl.set_lambda(grl_lambda)

                lang_logits = model.module.lang_head(model.module.grl(embeds))
                loss_lang = lang_ce(lang_logits, lang_targets)

                loss = loss + (lang_loss_weight * loss_lang)


        # loss, acc
        loss_meter.add(loss.item())
        acc_meter.add(outputs.cpu().detach().numpy(), targets.cpu().numpy())

        # updata the model
        optimizer.zero_grad()
        # scaler does nothing here if enable_amp=False
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # log
        if (i + 1) % configs['log_batch_interval'] == 0:
            logger.info(
                tp.row((epoch, i + 1, scheduler.get_lr(),
                        margin_scheduler.get_margin()) +
                       (loss_meter.value()[0], acc_meter.value()[0]),
                       width=10,
                       style='grid'))

        if (i + 1) == epoch_iter:
            break

    logger.info(
        tp.row(
            (epoch, i + 1, scheduler.get_lr(), margin_scheduler.get_margin()) +
            (loss_meter.value()[0], acc_meter.value()[0]),
            width=10,
            style='grid'))
