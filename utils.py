import pytorch_lightning as pl
import torch
import random
import gc
from typing import Union
from pytorch_lightning.utilities.rank_zero import rank_zero_only




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
    def __init__(self, cfg, num_step_to_log=100, num_texts=3):
        super().__init__()

        self.cfg = cfg
        self.num_step_to_log = num_step_to_log
        self.num_texts = num_texts



    def decode_text(self, outputs) -> tuple[list[str], list[str]]:
        decode_logits = self.cfg.tokenizer.batch_decode(outputs['logits'], skip_special_tokens=True)
        decode_labels = self.cfg.tokenizer.batch_decode(outputs['labels'], skip_special_tokens=True)
        return decode_logits, decode_labels
    


    def sampling(self, outputs):
        random_numbers = random.sample(range(outputs['labels'].size(0)), self.num_texts)
        outputs['logits'] = torch.stack([outputs['logits'][i] for i in random_numbers])
        outputs['labels'] = torch.stack([outputs['labels'][i] for i in random_numbers])
        return outputs
    
    
    
    def prepare_outputs(self, outputs:Union[list[dict], dict]) -> tuple[torch.Tensor, torch.Tensor]:
        logits = outputs['logits'].argmax(-1)
        labels = torch.where(outputs['labels'] == -100,
                             torch.tensor(self.cfg.tokenizer.pad_token_id, device=outputs['labels'].device),
                             outputs['labels'])
        
        outputs['labels'] = labels
        outputs['logits'] = logits
        return outputs
    
    
    def get_outputs(self, outputs, sample:bool) -> tuple:
        outputs = self.sampling(outputs=outputs) if sample else outputs
        outputs = self.prepare_outputs(outputs=outputs)    
        return outputs
    
    
    def get_texts(self, outputs, sample):
        outputs = self.get_outputs(outputs=outputs, sample=sample)
        decode_logits, decode_labels = self.decode_text(outputs)
        return decode_logits, decode_labels
        
    
    def text_metric_compute(self, pl_module, outputs, name:str) -> None:
        decode_logits, decode_labels = self.get_texts(outputs=outputs, sample=False)
        pl_module._log_step(decode_logits, decode_labels, name=name)


    def log_sample_text(self, trainer, outputs, name:str):
        decode_logits, decode_labels = self.get_texts(outputs=outputs, sample=True)
        self.log_text(decode_logits=decode_logits, decode_labels=decode_labels, trainer=trainer, name=name)
    
    
    
    def log_text(self, decode_logits:list[str], decode_labels:list[str], trainer, name) -> None:
        pass

            

    
    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.text_metric_compute(pl_module, outputs, name='TRAIN')
        if (batch_idx + 1) % self.num_step_to_log == 0:
            self.log_sample_text(trainer, outputs, name='TRAIN')

    
    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self.text_metric_compute(pl_module, outputs, name='VALID')
        if (batch_idx + 1) % self.num_step_to_log == 0:
            self.log_sample_text(trainer, outputs, name='VALID')

    
    @rank_zero_only
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self.text_metric_compute(pl_module, outputs, name='TEST')
        if (batch_idx + 1) % self.num_step_to_log == 0:
            self.log_sample_text(trainer, outputs, name='TEST')

