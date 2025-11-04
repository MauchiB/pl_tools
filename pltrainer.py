import pytorch_lightning as pl
import torch
from typing import Union, Any, List, Dict
import inspect


class Config:
    def __init__(self, 
                model, 
                optimizer_dict:Dict,
                loss_fn: Any = None,
                metrics_dict:Dict={},
                freeze:Union[None, int, list, tuple]=None, 
                label_names:List=['labels'],
                input_names:List=None
                ):
        
        super().__init__()

        self.model = model
        self.freeze = freeze
        self.label_names = label_names
        self.loss_fn = loss_fn
        self.optimizer_dict = optimizer_dict
        self.metrics_dict = metrics_dict
        self.input_names=input_names
            
            

    



class CustomModel(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
        self.model = cfg.model

        self.metrics_train = torch.nn.ModuleDict({k:v.clone() for k, v in cfg.metrics_dict.items()})
        self.metrics_valid = torch.nn.ModuleDict({k:v.clone() for k, v in cfg.metrics_dict.items()})
        self.metrics_test = torch.nn.ModuleDict({k:v.clone() for k, v in cfg.metrics_dict.items()})
    

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
    
            

    def _log_step(self, 
                  outputs:list[Union[str, torch.Tensor]],
                  labels:list[Union[str, torch.Tensor]],
                  name:str,
                  loss=None) -> None:
        
        is_str = isinstance(outputs[0], str)
        metric_dict = self._get_metrics_dict(name=name)
        
        if loss: self.log(f'{name}_loss', loss, on_epoch=True)

        for name_metric, metric in metric_dict.items():
            if getattr(metric, 'requires_text', False):
                if not is_str:
                    continue
    
            metric.update(outputs if is_str else outputs.argmax(-1),
                          labels)
            
            self.log(f'{name}_{name_metric}', metric, on_epoch=True)



    def get_param(self, batch:dict) -> tuple:
        if self.cfg.input_names:
            annot_keys = self.cfg.input_names
        else:
            annot_keys = list(inspect.signature(self.model.forward).parameters.keys())
   

        param = {k:v for k,v in batch.items() if k in annot_keys}
        labels = {k:v for k,v in batch.items() if k in self.cfg.label_names}

        if len(labels) == 1:
            labels = labels[self.cfg.label_names[0]]

        return param, labels
    


    def get_outputs(self, outputs:Any):

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

        return logits
    
    

    def get_loss(self, outputs, logits, labels):
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss

        elif self.cfg.loss_fn:
            loss = self.cfg.loss_fn(logits, labels)

            if loss is None:
                raise ValueError(f'cfg.loss_fn return {type(loss)}')

        else:
            raise ValueError(f'model didn`t return loss and cfg.loss_fn is None')
        
        return loss
        


    def forward(self, x):
        x = self.model(**x)
        return x
    

    def _step(self, batch, name:str, batch_idx:int):
        param, labels = self.get_param(batch)
          
        outputs = self(param)

        outputs = self.get_outputs(outputs)

        loss = self.get_loss(outputs=outputs, logits=outputs, labels=labels)
            
        self._log_step(outputs=outputs, labels=labels, name=name, loss=loss)

        return loss, outputs, labels
    


    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch, name='TRAIN', batch_idx=batch_idx)
        return {'loss':loss, 'outputs':outputs, 'labels':labels}

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch, name='VALID', batch_idx=batch_idx)
        return {'loss':loss, 'outputs':outputs, 'labels':labels}

    def test_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch, name='TEST', batch_idx=batch_idx)
        return {'loss':loss, 'outputs':outputs, 'labels':labels}

    def predict_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        return self.cfg.optimizer_dict