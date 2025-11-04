import pytorch_lightning as pl
import torch
import random
import gc
from typing import Union
from pytorch_lightning.utilities.rank_zero import rank_zero_only

class BaseCallback(pl.Callback):
    def __init__(self, num_obj):
        super().__init__()
        self.num_obj = num_obj

    def sampling(self, outputs):
        column = list(outputs.keys())[0]
        random_numbers = random.sample(range(outputs[column].size(0)), self.num_obj)
        for k,v in outputs.items():
            if v.numel() > self.num_obj:
                outputs[k] = v[random_numbers]

        return outputs
    
    def switch_device(self, outputs):
        for k,v in outputs.items():
            outputs[k] = v.cpu()
        return outputs
    


class VisualizationTextCallback(BaseCallback):
    def __init__(self, tokenizer, num_step_to_log=100, num_obj=3):
        super().__init__(num_obj)

        self.tokenizer = tokenizer
        self.num_step_to_log = num_step_to_log
        self.num_texts = num_obj



    def decode_text(self, outputs) -> tuple[list[str], list[str]]:
        decode_logits = self.tokenizer.batch_decode(outputs['outputs'], skip_special_tokens=True)
        decode_labels = self.tokenizer.batch_decode(outputs['labels'], skip_special_tokens=True)
        return decode_logits, decode_labels
    
    
    
    
    def prepare_outputs(self, outputs:Union[list[dict], dict]) -> tuple[torch.Tensor, torch.Tensor]:
        logits = outputs['outputs'].argmax(-1)
        labels = torch.where(outputs['labels'] == -100,
                             torch.tensor(self.tokenizer.pad_token_id, device=outputs['labels'].device),
                             outputs['labels'])
        
        outputs['labels'] = labels
        outputs['outputs'] = logits
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
        raise ValueError('log_text wasn`t determined')


    def _log(self, pl_module, trainer, outputs, name):
        self.text_metric_compute(pl_module, outputs, name=name)
        if (trainer.global_step + 1) % self.num_step_to_log == 0:
            self.log_sample_text(trainer, outputs, name=name)
        

    
    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log(pl_module=pl_module, trainer=trainer, outputs=outputs, name='TRAIN')

    
    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self._log(pl_module=pl_module, trainer=trainer, outputs=outputs, name='VALID')

    
    @rank_zero_only
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self._log(pl_module=pl_module, trainer=trainer, outputs=outputs, name='TEST')






class VisualizationImageCallback(BaseCallback):
    def __init__(self, num_step_to_log=100, num_obj=3):
        super().__init__(num_obj)


    def switch_device(self, outputs):
        for k,v in outputs.items():
            outputs[k] = v.cpu()
        return outputs

    
    def prepare_images(self, batch, outputs):
        common = {**batch, **outputs}
        outputs = self.switch_device(common)
        outputs = self.sampling(outputs)
        return outputs

    def log_image(self, outputs, trainer, name):
        pass


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        outputs = self.prepare_images(batch, outputs)
        self.log_image(outputs, trainer, 'TRAIN')

    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        outputs = self.prepare_images(batch, outputs)
        self.log_image(outputs, trainer, 'VALID')
    

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        outputs = self.prepare_images(batch, outputs)
        self.log_image(outputs, trainer, 'TEST')
    


