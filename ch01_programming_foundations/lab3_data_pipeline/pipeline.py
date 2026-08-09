from abc import ABC, abstractmethod
import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO)

class PipelineStage(ABC):
    @abstractmethod
    def process(self, data):
        pass

class IngestStage(PipelineStage):
    def __init__(self, path, file_type='csv'):
        self.path = path
        self.file_type = file_type
        
    def process(self, data=None):
        logging.info(f"Ingesting {self.path}")
        if self.file_type == 'csv':
            return pd.read_csv(self.path)
        return pd.DataFrame()

class ValidateStage(PipelineStage):
    def process(self, data):
        logging.info("Validating data")
        if data.isnull().any().any():
            data = data.dropna()
        return data

class TransformStage(PipelineStage):
    def __init__(self, rename_cols=None):
        self.rename_cols = rename_cols or {}
        
    def process(self, data):
        logging.info("Transforming data")
        return data.rename(columns=self.rename_cols)

class ExportStage(PipelineStage):
    def __init__(self, path):
        self.path = path
        
    def process(self, data):
        logging.info(f"Exporting to {self.path}")
        data.to_parquet(self.path)
        return data

class DataPipeline:
    def __init__(self, stages):
        self.stages = stages
        
    def run(self):
        data = None
        for stage in self.stages:
            data = stage.process(data)
        return data
        
    @classmethod
    def from_yaml(cls, config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        stages = []
        for stage_cfg in config.get('stages', []):
            if stage_cfg['type'] == 'ingest':
                stages.append(IngestStage(stage_cfg['path']))
            elif stage_cfg['type'] == 'validate':
                stages.append(ValidateStage())
            elif stage_cfg['type'] == 'transform':
                stages.append(TransformStage(stage_cfg.get('rename', {})))
            elif stage_cfg['type'] == 'export':
                stages.append(ExportStage(stage_cfg['path']))
        return cls(stages)
