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
                input_names:List=None,
                phases:Dict[str, str] = None
                ):
        
        super().__init__()

        self.model = model
        self.freeze = freeze
        self.label_names = label_names
        self.loss_fn = loss_fn
        self.optimizer_dict = optimizer_dict
        self.metrics_dict = metrics_dict
        self.input_names=input_names
        self.phases = phases


        self.set_phases()


    def set_phases(self):
        base_phases = {
        'TRAIN':'TRAINING',
        'VALIDATION':'VALID',
        'TEST':'TESTING',
        'PREDICT':'PRED'
        }

        if self.phases:
            for k,v in self.phases.items():
                base_phases[k] = v

        self.phases = base_phases

        print(self.phases)
            
        
            
            



class CustomModel(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        self.cfg = cfg
        self.model = cfg.model



        self._train_metirc_dict = torch.nn.ModuleDict({k:v.clone() for k, v in cfg.metrics_dict.items()})
        self._valid_metirc_dict = torch.nn.ModuleDict({k:v.clone() for k, v in cfg.metrics_dict.items()})
        self._test_metirc_dict = torch.nn.ModuleDict({k:v.clone() for k, v in cfg.metrics_dict.items()})


        self._metrics_dict = {
            'TRAIN':self._train_metirc_dict,
            'VALIDATION':self._valid_metirc_dict,
            'TEST':self._test_metirc_dict
        }


        self.phase = 'UNKNOWN'
        self._phase = 'UNKNOWN'
    

        self.save_hyperparameters()

        self.custom_freeze(self.model)



    def _set_phase(self, phase_key) -> str:
        self.phase = self.cfg.phases[phase_key]
        self._phase = phase_key



    def custom_freeze(self, model):
        self.model.train()
        layer_counter = 0
        
        if isinstance(self.cfg.freeze, int):
            list_layers = list(model.named_parameters())[::-1]
        else:
            list_layers = list(model.named_parameters())


        for name, param in list_layers:

            if self.cfg.freeze == None:
                param.requires_grad_(True)

            elif isinstance(self.cfg.freeze, int):
                is_bias = 'bias' in name.lower()
                if not is_bias:
                    layer_counter += 1
                param.requires_grad_(layer_counter < self.cfg.freeze)

            elif isinstance(self.cfg.freeze, (list, tuple)):
                param.requires_grad_(any(f in name.lower() for f in self.cfg.freeze))

            else:
                raise ValueError(f'unsupported type for freeze: {type(self.cfg.freeze)}')
            
    
    @property
    def _get_metrics_dict(self) -> torch.nn.ModuleDict:
        return self._metrics_dict[self._phase]

    

    def _log_step(self, 
                  outputs:list[Union[str, torch.Tensor]],
                  labels:list[Union[str, torch.Tensor]],
                  loss=None) -> None:
        
        is_str = isinstance(outputs[0], str)
        metric_dict = self._get_metrics_dict
        
        if loss: self.log(f'{self.phase}_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


        if metric_dict:

            for name_metric, metric in metric_dict.items():
                if getattr(metric, 'requires_text', False):
                    if not is_str:
                        continue
        
                metric.update(outputs if is_str else outputs.argmax(-1),
                            labels)
                
                self.log(f'{self.phase}_{name_metric}', metric, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)



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
    

    def _step(self, batch):
        param, labels = self.get_param(batch)
          
        outputs = self(param)

        outputs = self.get_outputs(outputs)

        loss = self.get_loss(outputs=outputs, logits=outputs, labels=labels)
            
        self._log_step(outputs=outputs, labels=labels, loss=loss)

        return loss, outputs, labels
    


    def training_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch)
        return {'loss':loss, 'outputs':outputs, 'labels':labels}

    def validation_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch)
        return {'loss':loss, 'outputs':outputs, 'labels':labels}

    def test_step(self, batch, batch_idx):
        loss, outputs, labels = self._step(batch)
        return {'loss':loss, 'outputs':outputs, 'labels':labels}

    def predict_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        return self.cfg.optimizer_dict
    


    def on_train_epoch_start(self):
        self._set_phase(phase_key='TRAIN')


    def on_validation_epoch_start(self):
        self._set_phase(phase_key='VALIDATION')


    def on_test_epoch_start(self):
        self._set_phase(phase_key='TEST')