import pytorch_lightning as pl
import torch
from typing import Union, Any


class Config:
    def __init__(self, model, tokenizer, metrics_dict:dict={},
                freeze:Union[None, int, list, tuple]=None, label_name:str='label',
                loss_fn=None, optimizer=None,
                lr_scheduler=None,
                lr_interval:str='step',
                ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.freeze = freeze
        self.label_name = label_name
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.interval = lr_interval
        self.metrics_dict = metrics_dict

        self._init_param()
        self._requires_text()


    def _init_param(self):
        self.optimizer_dict = {'optimizer':self.optimizer}
        if self.lr_scheduler:
            self.optimizer_dict['lr_scheduler'] = {
                'scheduler':self.lr_scheduler,
                'interval':self.interval
                }
            
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

        annot_keys = self.model.forward.__annotations__.keys()
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
                raise ValueError(f'outputs.logits return {type(loss)}')

        elif hasattr(self.cfg, 'loss_fn'):
            loss = self.cfg.loss_fn(logits, labels)

            if loss is None:
                raise ValueError(f'cfg.loss_fn return {type(loss)}')

        else:
            raise ValueError(f'model didn`t return loss and cfg.loss_fn is None')
        

        return loss
            


    def forward(self, x):
        x = self.model(**x)
        return x

    def _step(self, batch, name:str):
        param, labels = self._get_param(batch)

        outputs = self(param)

        outputs, logits = self._get_outputs(outputs)

        loss = self._get_loss(outputs=outputs, logits=logits, labels=labels)


        if self.cfg.requires_text:
    
            if name == 'VALID':
                self.valid_outputs.append({'logits':logits.detach().cpu(), 'labels':labels.detach().cpu()})
            if name == 'TEST':
                self.test_outputs.append({'logits':logits.detach().cpu(), 'labels':labels.detach().cpu()})

        return loss, logits, labels


    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch, name='TRAIN')
        self._log_step(outputs=outputs, labels=labels, name='TRAIN', loss=loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch, name='VALID')
        self._log_step(outputs=outputs, labels=labels, name='VALID', loss=loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch, name='TEST')
        self._log_step(outputs=outputs, labels=labels, name='TEST', loss=loss)
        return loss

    def predict_step(self, batch, batch_idx):
        labels = batch.pop(self.cfg.label_name)
        outputs = self(batch)
        return outputs
    
    def configure_optimizers(self):
        return self.cfg.optimizer_dict