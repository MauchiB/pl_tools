import pytorch_lightning as pl
import torch
from typing import Union, List, Dict
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torchvision.utils import make_grid, save_image
import os
import inspect

class BaseCallback(pl.Callback):
    def __init__(self, num_step_to_log:int, num_obj:int) -> None:
        super().__init__()
        self.num_obj = num_obj
        self.num_step_to_log = num_step_to_log


        self._val_step = 0
        self._test_step = 0
        self._pred_step = 0



    def _map_dicts(self, *dicts, func, **kwargs) -> List[Dict]:
        results = []
        for d in dicts:
            results.append(func(outputs=d, **kwargs))
        return results
    
    
    def get_batch(self, outputs:Dict) -> int:
        tensors = [v.size(0) for v in outputs.values() if isinstance(v, torch.Tensor) and v.dim() > 0]
        if not tensors:
            raise ValueError('No batched tensors (dim > 0) found in outputs')
        batch_size = max(tensors)
        return batch_size
            
            
            

    def sampling(self, outputs:Dict) -> Dict:
        
        batch_size = self.get_batch(outputs=outputs)

        if self.num_obj > batch_size:
            raise ValueError('num_obj > batch_size')
        
        gen = torch.Generator()
        gen.manual_seed(self._step)
        
        random_numbers = torch.randperm(batch_size, generator=gen)[:self.num_obj]

        sampled_dict = {k:v[random_numbers] if isinstance(v, torch.Tensor) and v.dim() > 0 else v
                        for k,v in outputs.items()}
                
        return sampled_dict
        
    
    def switch_device(self, outputs):
        device_dict = {}

        for k,v in outputs.items():
            if isinstance(v, torch.Tensor):
                device_dict[k] = v.detach().cpu()
            else:
                device_dict[k] = v
                
        return device_dict
    

    def _set_step(self) -> int:
        if not self._trainer.sanity_checking:
            if self._phase == 'VALIDATION':
                self._val_step += 1

            if self._phase == 'TEST':
                self._test_step += 1

            if self._phase == 'PREDICT':
                self._pred_step += 1
        
    @property
    def _step(self):
        if self._phase == 'TRAIN':
            return self._trainer.global_step
        if self._phase == 'VALIDATION':
            return self._val_step
        if self._phase == 'TEST':
            return self._test_step
        if self._phase == 'PREDICT':
            return self._pred_step
        
    
    
    def _setuping(self):
        self.phase = self._pl_module.phase
        self._phase = self._pl_module._phase
        self._set_step()
        

    def setup(self, trainer, pl_module, stage):
        self.device = next(pl_module.parameters()).device
        self._trainer = trainer
        self._pl_module = pl_module
        self.logger = trainer.logger


    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._setuping()


    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._setuping()


    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._setuping()




class VisualizationTextCallback(BaseCallback):
    def __init__(self, tokenizer, num_step_to_log:int=100, num_obj:int=3):
        super().__init__(num_step_to_log, num_obj)

        self.tokenizer = tokenizer
        self.num_step_to_log = num_step_to_log
        self.num_texts = num_obj



    def decode_text(self, outputs:Dict) -> tuple[List[str], List[str]]:
        decode_logits = self.tokenizer.batch_decode(outputs['outputs'], skip_special_tokens=True)
        decode_labels = self.tokenizer.batch_decode(outputs['labels'], skip_special_tokens=True)
        return decode_logits, decode_labels
    
    
    def prepare_outputs(self, outputs:Dict) -> Dict:
        prepared_outputs = outputs.copy()

        logits = prepared_outputs['outputs'].argmax(-1)
        labels = torch.where(prepared_outputs['labels'] == -100,
                             self._pad_token_id,
                             prepared_outputs['labels'])
        
        prepared_outputs['labels'] = labels
        prepared_outputs['outputs'] = logits
        return prepared_outputs
    
    
    def get_outputs(self, outputs:Dict, sample:bool) -> tuple:
        outputs = self.switch_device(outputs=outputs)
        outputs = self.prepare_outputs(outputs=outputs)  
        outputs = self.sampling(outputs=outputs) if sample else outputs  
        return outputs
    
    
    def get_texts(self, outputs:Dict, sample:bool) -> tuple[List[str], List[str]]:
        outputs = self.get_outputs(outputs=outputs, sample=sample)
        decode_logits, decode_labels = self.decode_text(outputs)
        return decode_logits, decode_labels
        
    
    def text_metric_compute(self, outputs:Dict) -> None:
        decode_logits, decode_labels = self.get_texts(outputs=outputs, sample=False)
        self._pl_module._log_step(decode_logits, decode_labels)


    def log_sample_text(self, outputs:Dict) -> None:
        decode_logits, decode_labels = self.get_texts(outputs=outputs, sample=True)
        self.log_text(decode_logits=decode_logits, decode_labels=decode_labels)
    
    
    def log_text(self, decode_logits:List[str], decode_labels:List[str]) -> None:
        '''use self.logger for getting logger'''
        raise ValueError('log_text wasn`t determined')


    def _log(self, outputs):
        self.text_metric_compute(outputs)
        if (self._step) % self.num_step_to_log == 0:
            self.log_sample_text(outputs)



    def on_train_start(self, trainer, pl_module):
        self._pad_token_id = torch.tensor(self.tokenizer.pad_token_id, device=self.device)



    
    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log(outputs=outputs)
  

    
    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self._log(outputs=outputs)
  

    
    @rank_zero_only
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        self._log(outputs=outputs)








class VisualizationImageCallback(BaseCallback):
    def __init__(self, image_column, num_step_to_log:int=100, num_obj:int=3, folder_to_save:str=None):
        super().__init__(num_step_to_log, num_obj)

        self.image_column = image_column
        self.folder_to_save = folder_to_save



    def prepare_outputs(self, outputs:Dict):
        outputs = self.switch_device(outputs)
        outputs = self.sampling(outputs)
        return outputs
    

    def make_image(self, batch, outputs:Dict):
        return make_grid(batch[self.image_column])


    def annotate_image(self, image, batch:Dict, outputs:Dict):
        return image
        

    def save_image(self, image):
        if not os.path.exists(self.folder_to_save):
            os.makedirs(self.folder_to_save, exist_ok=True)
        path = os.path.join(self.folder_to_save, f'{self.phase}_step_{self.step}.jpg')
        save_image(image, path)


    def prepare_image(self, batch:Dict, outputs:Dict):
        image = self.make_image(batch, outputs)
        image = self.annotate_image(image, batch, outputs)
        
        if self.folder_to_save:
            self.save_image(image=image)
        
        return image
    
        

    def log_image(self, image, batch:Dict, outputs:Dict):
        pass


    def _log_image(self, batch, outputs):
        batch, outputs = self._map_dicts(batch, outputs, func=self.prepare_outputs)
        image = self.prepare_image(batch=batch, outputs=outputs)
        self.log_image(image=image, batch=batch, outputs=outputs)
        


    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (self._step) % self.num_step_to_log == 0:
            self._log_image(batch=batch, outputs=outputs)


    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        if (self._step) % self.num_step_to_log == 0:
            self._log_image(batch=batch, outputs=outputs)

    
    @rank_zero_only
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx = 0):
        if (self._step) % self.num_step_to_log == 0:
            self._log_image(batch=batch, outputs=outputs)


        


