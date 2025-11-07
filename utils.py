import pytorch_lightning as pl
import torch
import random
import gc
from typing import Union, List, Dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchvision.utils import make_grid, save_image
import os

class BaseCallback(pl.Callback):
    def __init__(self, num_step_to_log, num_obj):
        super().__init__()
        self.num_obj = num_obj
        self.num_step_to_log = num_step_to_log


    def _map_dicts(self, *dicts, func, **kwargs) -> List[Dict]:
        results = []
        for d in dicts:
            results.append(func(outputs=d, **kwargs))
        return results
    
    
    def get_batch(self, outputs):
        tensors = [v.size(0) for v in outputs.values() if isinstance(v, torch.Tensor) and v.dim() > 0]
        if not tensors:
            raise ValueError('No batched tensors (dim > 0) found in outputs')
        batch_size = max(tensors)
        return batch_size
            
            
            

    def sampling(self, outputs, seed:int):
        
        batch_size = self.get_batch(outputs=outputs)

        if self.num_obj > batch_size:
            raise ValueError('num_obj > batch_size')
        
        gen = torch.Generator()
        if seed:
            gen.manual_seed(seed)
        
        random_numbers = torch.randperm(batch_size, generator=gen)[:self.num_obj]

        sampled_dict = {k:v[random_numbers] if isinstance(v, torch.Tensor) and v.dim() > 0 else v
                        for k,v in outputs.items()}
                
        return sampled_dict
        
    
    def switch_device(self, outputs):
        device_dict = {k:v.detach().cpu()
                            for k,v in outputs.items()}
        return device_dict
    

    


class VisualizationTextCallback(BaseCallback):
    def __init__(self, tokenizer, num_step_to_log=100, num_obj=3):
        super().__init__(num_step_to_log, num_obj)

        self.tokenizer = tokenizer
        self.num_step_to_log = num_step_to_log
        self.num_texts = num_obj



    def decode_text(self, outputs) -> tuple[list[str], list[str]]:
        decode_logits = self.tokenizer.batch_decode(outputs['outputs'], skip_special_tokens=True)
        decode_labels = self.tokenizer.batch_decode(outputs['labels'], skip_special_tokens=True)
        return decode_logits, decode_labels
    
    
    
    
    def prepare_outputs(self, outputs) -> tuple[torch.Tensor, torch.Tensor]:
        logits = outputs['outputs'].argmax(-1)
        labels = torch.where(outputs['labels'] == -100,
                             torch.tensor(self.tokenizer.pad_token_id, device=outputs['labels'].device),
                             outputs['labels'])
        outputs['labels'] = labels
        outputs['outputs'] = logits
        return outputs
    
    
    def get_outputs(self, outputs, sample:bool) -> tuple:
        outputs = self.switch_device(outputs=outputs)
        outputs = self.prepare_outputs(outputs=outputs)  
        outputs = self.sampling(outputs=outputs) if sample else outputs  
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
    
    
    
    def log_text(self, decode_logits:List[str], decode_labels:List[str], trainer, name) -> None:
        raise ValueError('log_text wasn`t determined')


    def _log(self, pl_module, trainer, outputs, name):
        self.text_metric_compute(pl_module, outputs, name=name)
        if (trainer.global_step) % self.num_step_to_log == 0:
            self.log_sample_text(trainer, outputs, name=name)
        

    
    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._map_dicts(outputs, func=self._log, pl_module=pl_module, trainer=trainer, name='TRAIN')

    
    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self._map_dicts(outputs, func=self._log, pl_module=pl_module, trainer=trainer, name='VALID')

    
    @rank_zero_only
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self._map_dicts(outputs, func=self._log, pl_module=pl_module, trainer=trainer, name='TEST')






class VisualizationImageCallback(BaseCallback):
    def __init__(self, image_column, num_step_to_log=100, num_obj=3, folder_to_save=None):
        super().__init__(num_step_to_log, num_obj)
        self.image_column = image_column
        self.folder_to_save = folder_to_save



    def prepare_outputs(self, outputs, seed):
        outputs = self.switch_device(outputs)
        outputs = self.sampling(outputs, seed=seed)
        return outputs
    

    def make_image(self, batch, outputs, trainer, name):
        return batch[self.image_column]


    def annotate_image(self, image, batch, outputs, trainer, name):
        return image
        

    def save_image(self, image):
        if not os.path.exists(self.folder_to_save):
            os.makedirs(self.folder_to_save, exist_ok=True)
        save_image(image, self.folder_to_save)


    def prepare_image(self, batch, outputs, trainer, name):
        image = self.make_image(batch, outputs, trainer, name)
        image = self.annotate_image(image, batch, outputs, trainer, name)
        
        if self.folder_to_save:
            self.save_image(image=image)
        
        return image
    
        

    def log_image(self, image, batch, outputs, trainer, name):
        pass


    def _log_image(self, batch, outputs, trainer, name):
        batch, outputs = self._map_dicts(batch, outputs, func=self.prepare_outputs, seed=trainer.global_step)
        image = self.prepare_image(batch=batch, outputs=outputs, trainer=trainer, name=name)
        self.log_image(image=image, batch=batch, outputs=outputs, trainer=trainer, name=name)
        

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step) % self.num_step_to_log == 0:
            self._log_image(batch=batch, outputs=outputs, trainer=trainer, name='TRAIN')


    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        if (trainer.global_step) % self.num_step_to_log == 0:
            self._log_image(batch=batch, outputs=outputs, trainer=trainer, name='VALID')

    
    @rank_zero_only
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        if (trainer.global_step) % self.num_step_to_log == 0:
            self._log_image(batch=batch, outputs=outputs, trainer=trainer, name='TEST')
        


