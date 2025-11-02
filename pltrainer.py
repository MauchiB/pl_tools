import pytorch_lightning as pl
import torch
from typing import Union, Any, List, Dict
import inspect


class Config:
    def __init__(self, 
                model, 
                tokenizer,
                optimizer_dict:Dict,
                loss_fn: Any = None,
                metrics_dict:Dict={},
                freeze:Union[None, int, list, tuple]=None, 
                label_name:str='label',
                num_batch_to_save=1,
                input_names:List=None
                ):
        
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.freeze = freeze
        self.label_name = label_name
        self.loss_fn = loss_fn
        self.optimizer_dict = optimizer_dict
        self.metrics_dict = metrics_dict
        self.num_batch_to_save = num_batch_to_save
        self.input_names=input_names


        self._requires_text()

    def _requires_text(self):
        if any([i.requires_text for i in self.metrics_dict.values()]):
            self.requires_text = True
            
            
    
    



class CustomModel(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
        self.model = cfg.model

        self.metrics_train = torch.nn.ModuleDict({k:v.clone() for k, v in cfg.metrics_dict.items()})
        self.metrics_valid = torch.nn.ModuleDict({k:v.clone() for k, v in cfg.metrics_dict.items()})
        self.metrics_test = torch.nn.ModuleDict({k:v.clone() for k, v in cfg.metrics_dict.items()})
        
        self.train_outputs = []
        self.valid_outputs = []
        self.test_outputs = []

        self.save_hyperparameters()

        self.custom_freeze(self.model, freeze=cfg.freeze)


    def custom_freeze(self, model, freeze):
        self.model.train()
        layer_counter = 0
        
        if isinstance(freeze, int):
            list_layers = list(model.named_parameters())[::-1]
        else:
            list_layers = list(model.named_parameters())


        for name, param in list_layers:

            if freeze == None:
                param.requires_grad_(True)

            elif isinstance(freeze, int):
                is_bias = 'bias' in name.lower()
                if not is_bias:
                    layer_counter += 1
                param.requires_grad_(layer_counter < freeze)

            elif isinstance(freeze, (list, tuple)):
                param.requires_grad_(any(f in name.lower() for f in freeze))

            else:
                raise ValueError(f'unsupported type for freeze: {type(freeze)}')
            

    def _get_metrics_dict(self, name:str) -> torch.nn.ModuleDict:
        if name == 'TRAIN':
            metric_dict = self.metrics_train
        if name == 'VALID':
            metric_dict = self.metrics_valid
        if name == 'TEST':
            metric_dict = self.metrics_test

        return metric_dict
    
            


    def _log_step(self, outputs:list[Union[str, torch.Tensor]], labels:list[Union[str, torch.Tensor]], name:str, loss=None) -> None:
        metric_dict = self._get_metrics_dict(name=name)

        is_str = isinstance(outputs[0], str)

        if not is_str:
            outputs = outputs.argmax(-1)

        if loss:
            self.log(f'{name}_loss', loss, on_epoch=True)

            
        for name_metric, metric in metric_dict.items():
            if getattr(metric, 'requires_text', False) and not is_str:
                continue
    

            metric.update(outputs, labels)
            self.log(f'{name}_{name_metric}', metric, on_epoch=True)


    def _get_param(self, batch:dict) -> tuple:
        if self.cfg.input_names:
            annot_keys = self.cfg.input_names
        else:
            forward = self.model.__class__.forward
            annot_keys = list(inspect.signature(forward).parameters.keys())
   

        param = {k:v for k,v in batch.items() if k in annot_keys}
        labels = batch[self.cfg.label_name]

        return param, labels
    


    def _get_outputs(self, outputs:Any):

        if hasattr(outputs, 'logits'):
            logits = outputs.logits

        elif isinstance(outputs, dict):
            if 'logits' in outputs:
                logits = outputs['logits']
    
            else:
                logits = list(outputs.values())[0]
                
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            raise ValueError(f'unsupported type for outputs: {type(outputs)}')

        return outputs, logits
    
    

    def _get_loss(self, outputs, logits, labels):
        if hasattr(outputs, 'loss'):
            loss = outputs.loss

            if loss is None:
                raise ValueError(f'outputs.logits return {type(loss)} / type of outputs is {type(outputs)}')

        elif hasattr(self.cfg, 'loss_fn'):
            loss = self.cfg.loss_fn(logits, labels)

            if loss is None:
                raise ValueError(f'cfg.loss_fn return {type(loss)}')

        else:
            raise ValueError(f'model didn`t return loss and cfg.loss_fn is None')
        

        return loss
    
    def _save_results(self, name, **kwargs):
        if name == 'TEST':
            self.test_outputs.append(kwargs)
        elif name == 'VALID':
            self.valid_outputs.append(kwargs)
        elif name == 'TRAIN':
            self.train_outputs.append(kwargs)
        else:
            raise ValueError(f'name is {name}')
        
            


    def forward(self, x):
        x = self.model(**x)
        return x

    def _step(self, batch, name:str, batch_idx:int):
        param, labels = self._get_param(batch)
          
        outputs = self(param)

        outputs, logits = self._get_outputs(outputs)

        loss = self._get_loss(outputs=outputs, logits=logits, labels=labels)


        if self.cfg.num_batch_to_save < batch_idx:
            self._save_results(name=name,
                               logits=logits.argmax(-1).detach().cpu(),
                               labels=labels.detach().cpu()
                               )

        return loss, logits, labels


    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch, name='TRAIN', batch_idx=batch_idx)

        self._log_step(outputs=outputs, labels=labels, name='TRAIN', loss=loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch, name='VALID', batch_idx=batch_idx)

        self._log_step(outputs=outputs, labels=labels, name='VALID', loss=loss)

        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch, name='TEST', batch_idx=batch_idx)

        self._log_step(outputs=outputs, labels=labels, name='TEST', loss=loss)

        return loss

    def predict_step(self, batch, batch_idx):
        labels = batch.pop(self.cfg.label_name)
        outputs = self(batch)
        return outputs
    
    def configure_optimizers(self):
        return self.cfg.optimizer_dict