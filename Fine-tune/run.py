import os
import csv

import torch
from torch.nn import L1Loss
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from transformers import AdamW, get_linear_schedule_with_warmup

from model import Model
from metrics import val_metrics
from params import hyper_param
from constants import SAVE_PATH, PATIENCE, MAX_EPOCH, MAX_GRAD_NORM, DEVICE
from dataset import train_loader, dev_loader, test_loader


checkpoint = False


def train(param, device):
    model = Model(param)
    optimizer = AdamW(model.parameters(), lr=param.lr, eps=1e-8)
    update_steps = MAX_EPOCH * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=update_steps)
    loss_fn = L1Loss()
    trainer = create_trainer(model, optimizer, scheduler, loss_fn, MAX_GRAD_NORM, device)
    dev_evaluator = create_evaluator(model, val_metrics, device)
    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=10), log_training_loss)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_results, *[dev_evaluator, dev_loader, 'Dev'])
    es_handler = EarlyStopping(patience=PATIENCE, score_function=score_fn, trainer=trainer)
    dev_evaluator.add_event_handler(Events.COMPLETED, es_handler)
    ckpt_handler = ModelCheckpoint(SAVE_PATH, f'lr_{param.lr}', score_function=score_fn,
                                   score_name='score', require_empty=False)
    dev_evaluator.add_event_handler(Events.COMPLETED, ckpt_handler, {SAVE_PATH.split("/")[-1]: model})
    print(f'Start running {SAVE_PATH.split("/")[-1]} at device: {DEVICE}\tlr: {param.lr}')
    trainer.run(train_loader, max_epochs=MAX_EPOCH)


def load(param, device):
    model = Model(param)
    checkpoints = os.listdir(SAVE_PATH)
    res_file = f'{SAVE_PATH.split("/")[-1]}_results.tsv'
    if res_file in checkpoints:
        checkpoints.remove(res_file)
    checkpoints.sort(key=lambda check: float(check.split('=')[1].rsplit('.', 1)[0]), reverse=True)
    res_path = os.path.join(SAVE_PATH, res_file)
    with open(res_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Acc-2', 'F1', 'MAE', 'Corr', 'Acc-7', 'Acc-5'])
        for ckpt in checkpoints:
            state_dicts = torch.load(os.path.join(SAVE_PATH, ckpt))
            model.load_state_dict(state_dicts)
            test_evaluator = create_evaluator(model, val_metrics, device)
            log_ckpt(test_evaluator, test_loader, 'Test', ckpt, writer)


def create_trainer(model, optimizer, scheduler, loss_fn, max_grad_norm, device):
    model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device)
        y_pred = model(*x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        return loss

    return Engine(_update)


def create_evaluator(model, metrics, device):
    metrics = metrics or {}
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device)
            y_pred = model(*x)
            return y_pred, y

    evaluator = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    return evaluator


def log_training_loss(trainer):
    print(f'Epoch[{trainer.state.epoch}]\tIteration: {trainer.state.iteration}\tLoss: {trainer.state.output:.4f}')


def log_results(evaluator, loader, split):
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    print(f"{split} Results:\t"
          f"{metrics['acc2']:.3f}\t{metrics['f1']:.3f}\t"
          f"{metrics['mae']:.3f}\t{metrics['corr']:.3f}\t"
          f"{metrics['acc7']:.3f}\t{metrics['acc5']:.3f}")


def log_ckpt(evaluator, loader, split, ckpt, writer):
    evaluator.run(loader)
    metrics = evaluator.state.metrics
    writer.writerow([f"{metrics['acc2']:.3f}", f"{metrics['f1']:.3f}",
                     f"{metrics['mae']:.3f}", f"{metrics['corr']:.3f}",
                     f"{metrics['acc7']:.3f}", f"{metrics['acc5']:.3f}"])

    print(f"ckpt: {ckpt.split('=')[-1]}\t{split} Results:\t"
          f"{metrics['acc2']:.3f}\t{metrics['f1']:.3f}\t"
          f"{metrics['mae']:.3f}\t{metrics['corr']:.3f}\t"
          f"{metrics['acc7']:.3f}\t{metrics['acc5']:.3f}")


def score_fn(evaluator):
    score = sum(evaluator.state.metrics.values()) - 2 * evaluator.state.metrics['mae']
    return score


def prepare_batch(batch, device):
    for i in range(len(batch)):
        batch[i] = batch[i].to(device)
    return batch[:-1], batch[-1]


if __name__ == '__main__':
    if checkpoint:
        load(hyper_param, torch.device(f'cuda:{DEVICE}'))
    else:
        train(hyper_param, torch.device(f'cuda:{DEVICE}'))
