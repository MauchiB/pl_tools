import pytorch_lightning as pl
import torch
import random
import gc
from typing import Union


def clean_zero_in_labels(outputs:Union[list[dict], dict], pad_token_id=0) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = [f for f in outputs if f['labels'].numel() > 0]
        logits = torch.cat([f['logits'].argmax(-1) for f in outputs], 0)
        labels = torch.cat([f['labels'] for f in outputs], 0)

        labels = torch.where(labels == -100, torch.tensor(pad_token_id, device=labels.device), labels)
        return logits, labels



class VisualizationConsoleCallback(pl.Callback):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg


    def _get_metric_dict(self, trainer, name):
        metric_dict = {}

        metric_dict[f'LOSS'] = trainer.callback_metrics[f'{name}_loss']

        for name_metric, metric in self.cfg.metrics_dict.items():
            if getattr(metric, 'requires_text', False):
                continue

            mean_metric = trainer.callback_metrics[f'{name}_{name_metric}']
            metric_dict[name_metric] = mean_metric

        return metric_dict


    def _printing(self, trainer, pl_module, name):
        metric_dict = self._get_metric_dict(trainer=trainer, pl_module=pl_module, name=name)

        metric_str = ' '.join([f'{n} - {v} | ' for n, v in metric_dict.items()])
        output_str =  f'{name} | ' + metric_str

        print(output_str)


    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return  # пропускаем печать во время sanity check
        self._printing(trainer=trainer, pl_module=pl_module, name='TRAIN')

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return  # пропускаем печать во время sanity check
        self._printing(trainer=trainer, pl_module=pl_module, name='VALID')

    def on_test_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return  # пропускаем печать во время sanity check
        self._printing(trainer=trainer, pl_module=pl_module, name='TEST')






class VisualizationTextCallback(pl.Callback):
    def __init__(self, cfg, num_texts=3):
        super().__init__()

        self.cfg = cfg
        self.num_texts = num_texts

        self.wandb_table = []
        self.wandb_colums = ['epoch', 'step', 'generate', 'target']


    def _decode_text(self, outputs:list[dict]) -> tuple[list[str], list[str]]:
        logits, labels = clean_zero_in_labels(outputs=outputs, pad_token_id=self.cfg.tokenizer.pad_token_id)
    
        decode_labels = self.cfg.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decode_logits = self.cfg.tokenizer.batch_decode(logits, skip_special_tokens=True)

        return decode_logits, decode_labels
    
    def _get_outputs(self, pl_module, name:str, sample:bool=False) -> dict:
        outputs = pl_module.valid_outputs if name == 'VALID' else pl_module.test_outputs

        if sample and outputs:
            selected_samples = []
            
            for f in outputs:
                
                batch_size = f['labels'].size(0)

                for i in range(batch_size):
                    selected_samples.append({
                        'logits':f['logits'][i:i+1],
                        'labels':f['labels'][i:i+1],
                    })

                    if len(selected_samples) > self.num_texts:
                        break
                
                if len(selected_samples) > self.num_texts:
                    break

            outputs = selected_samples
                    
        return outputs
    
    
    def _text_metric_compute(self, pl_module, name:str) -> None:
        outputs = self._get_outputs(pl_module=pl_module, name=name, sample=False)
        decode_logits, decode_labels = self._decode_text(outputs=outputs)
        pl_module._log_step(decode_logits, decode_labels, name=name)


    def _log_sample_text(self, trainer, pl_module, name:str):
        outputs = self._get_outputs(pl_module=pl_module, name=name, sample=True)
        decode_logits, decode_labels = self._decode_text(outputs=outputs)
        self._log_text(decode_logits=decode_logits, decode_labels=decode_labels, trainer=trainer, name=name)
    
    
    def _log_text(self, decode_logits:list[str], decode_labels:list[str], trainer, name) -> None:
        loggers = trainer.logger
        if not isinstance(loggers, list):
            loggers = [loggers]
            

        for i, (text, target) in enumerate(zip(decode_logits, decode_labels)):
            msg = f'GENERATE | {text}\nTARGET | {target}'

            for logger in loggers:
                logger_str = str(type(logger))

                if 'TensorBoardLogger' in logger_str:
                    logger.experiment.add_text(f'{name}_text_epoch_{trainer.current_epoch}', msg, global_step=trainer.global_step)
                
                elif 'CometLogger' in logger_str:
                    logger.experiment.log_text(msg, step=trainer.global_step, metadata={'epoch':trainer.current_epoch, 'name':name})

                elif 'WandbLogger' in logger_str:
                    self.wandb_table.append([trainer.current_epoch, trainer.global_step, text, target])

                    if i == len(decode_logits) - 1:
                        logger.log_text(key=f'{name}_texts', columns=self.wandb_colums, data=self.wandb_table)
                        
                        self.wandb_table = []
            


    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return

        try:
            self._text_metric_compute(pl_module, 'VALID')
            self._log_sample_text(trainer=trainer, pl_module=pl_module, name='VALID')
        finally:
            pl_module.valid_outputs.clear()
            torch.cuda.empty_cache()
            gc.collect()
        


    def on_test_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        
        try:
            self._text_metric_compute(pl_module, 'TEST')
            self._log_sample_text(trainer=trainer, pl_module=pl_module, name='TEST')
        finally:
            pl_module.valid_outputs.clear()
            torch.cuda.empty_cache()
            gc.collect()

