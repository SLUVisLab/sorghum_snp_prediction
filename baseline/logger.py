from abc import ABCMeta, abstractclassmethod
import warnings
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os, time, torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
try:
    import neptune
except ImportError:
    warnings.warn('Neptune is not installed. NeptuneLogger will not work.')

class SummaryWriter(SummaryWriter):
    def add_hparams(
        self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None, global_step=None
    ):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        if not run_name:
            run_name = str(time.time())
        logdir = os.path.join(self._get_file_writer().get_logdir(), run_name)
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v, global_step)


class _Logger(metaclass=ABCMeta):
    @abstractclassmethod
    def __init__(self):
        pass

    @abstractclassmethod
    def log_init(self, trainer):
        pass

    @abstractclassmethod
    def log_step(self):
        pass

    @abstractclassmethod
    def step(self, trainer):
        pass

    @abstractclassmethod
    def log_epoch(self):
        pass

def _find_record_params(instance):
    try:
        float(instance)
        is_float = True
    except (ValueError, TypeError):
        is_float = False
    if not hasattr(instance, 'record_params'):
        if is_float:
            return {'': float(instance)}
        else:
            return {}
    else:
        all_params = {}
        if is_float:
            all_params[type(instance).__name__: float]
        for rp in instance.record_params:
            param_list = _find_record_params(getattr(instance, rp))
            for key in list(param_list.keys()):
                if key == '':
                    param_list[rp] = param_list.pop(key)
                else:
                    param_list[rp + '.' + key] = param_list.pop(key)
            all_params.update(param_list)
        return all_params

class TensorBoardLogger(_Logger):
    def __init__(self, tb_log_dir, log_graph=False, comment='', log_interval=10, hparams={}):
        self.log_df = pd.DataFrame()
        self.log_interval = log_interval
        self.globa_step_counter = 0
        self.epoch_counter = 0
        self.tb_log_dir = tb_log_dir
        self.log_graph = log_graph
        self.tb_writer = SummaryWriter(self.tb_log_dir, comment=comment)
        self.hparams = hparams
        # trainer as member or parameter?

    def log_init(self, trainer):
        self.steps_per_epoch = trainer.steps_per_epoch
        if self.log_graph:
            example_input = trainer.dataloader.dataset[0][0].unsqueeze(0).cuda()
            self.tb_writer.add_graph(trainer.model, example_input)

    def step(self, trainer):
        record_params = _find_record_params(trainer)
        self.log_df = self.log_df.append(record_params, ignore_index=True)
        if (self.globa_step_counter + 1) % self.log_interval == 0:
            self.log_step()
        if (self.globa_step_counter + 1) % self.steps_per_epoch == 0:
            self.epoch_counter = trainer.epoch_num
            self.log_epoch()
        self.globa_step_counter += 1

    def log_step(self):
        step_mean = self.log_df[-self.log_interval:].mean()
        for k, v in step_mean.items():
            if v is not None:
                self.tb_writer.add_scalar('step/' + k, v, self.globa_step_counter)

    def log_epoch(self):
        epoch_mean = self.log_df[-self.steps_per_epoch:].mean()
        hparam_log_dict = {}
        for k, v in epoch_mean.items():
            hparam_log_dict[k] = v
            if v is not None:
                self.tb_writer.add_scalar('epoch/' + k, v, self.epoch_counter)
            

class NeptuneLogger(_Logger):
    def __init__(self, exp_name, proj_name, comment='', api_token=None, log_graph=False, log_interval=10, hparams={}):
        self.exp_name = exp_name
        self.proj_name = proj_name
        self.api_token = api_token
        self.log_graph = log_graph
        self.comment = comment
        self.hparams = hparams

        if self.api_token is None:
            try:  
                os.environ["NEPTUNE_API_TOKEN"]
            except KeyError: 
                print('Pass api_token or set env var NEPTUNE_API_TOKEN')
        self.run = neptune.init_run(api_token=self.api_token, project=self.proj_name,
                                     name=self.exp_name, description=self.comment)
        self.run['parameters'] = self.hparams
        
        self.log_df = pd.DataFrame()
        self.log_interval = log_interval
        self.globa_step_counter = 0
        self.epoch_counter = 0
        # trainer as member or parameter?

    def log_init(self, trainer):
        self.steps_per_epoch = trainer.steps_per_epoch
        # self.exp.set_property('train_timestamp', trainer.timestamp)

    def step(self, trainer):
        record_params = _find_record_params(trainer)
        self.log_df = self.log_df.append(record_params, ignore_index=True)
        if (self.globa_step_counter + 1) % self.log_interval == 0:
            self.log_step()
        if (self.globa_step_counter + 1) % self.steps_per_epoch == 0:
            self.epoch_counter = trainer.epoch_num
            self.log_epoch()
        self.globa_step_counter += 1

    def log_step(self):
        step_mean = self.log_df[-self.log_interval:].mean()
        for k, v in step_mean.items():
            if v is not None:
                # self.exp.log_metric('step/' + k, y=v, x=self.globa_step_counter)
                self.run['step/' + k].log(v, timestamp=self.globa_step_counter)
    def log_epoch(self):
        epoch_mean = self.log_df[-self.steps_per_epoch:].mean()
        hparam_log_dict = {}
        for k, v in epoch_mean.items():
            hparam_log_dict[k] = v
            if v is not None:
                # self.exp.log_metric('epoch/' + k, y=v, x=self.epoch_counter)
                self.run['epoch/' + k].log(v, timestamp=self.epoch_counter)  