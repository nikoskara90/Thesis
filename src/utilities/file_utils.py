# Copyright 2023 Lawrence Livermore National Security, LLC and other
# LUAR Project Developers.
#
# SPDX-License-Identifier: Apache-2.0

import configparser
import os
from datetime import datetime

from tqdm import tqdm

from utilities import decorators as decorator

@decorator.singleton
class Utils(object):
    """Keeps track of all the paths required by the application and provides 
       basic utilities for reading JSON files.
    """
    def __init__(self):
        # Read project file configuration
        self._config_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.config_filename = os.path.join(self._config_dir, 'file_config.ini')
        self.config = configparser.ConfigParser()
        self.config.read(self.config_filename)

        # Set all the paths and create an experiment name
        self.set_paths()
        self._project_name = self.create_project_name()

    @staticmethod
    def normalize_path(path: str) -> str:
        """Normalize path to be platform-safe and absolute."""
        return os.path.abspath(os.path.normpath(path))

    @property
    def config_dir(self):
        return self._config_dir

    @property
    def project_path(self):
        return self._project_path

    @property
    def output_path(self):
        return self._output_path

    @property
    def data_path(self):
        return self._data_path

    @property
    def transformer_path(self):
        return self._transformer_path

    @property
    def embedding_path(self):
        return self._embeddings_path

    @property
    def query_path(self):
        return self._query_path

    def create_project_name(self):
        """Creates a project name."""
        name = datetime.now().strftime('Experiment_%B_%Y')
        return name

    def set_paths(self):
        """Sets all of the project path variables that are relevant for reading pre-trained models, 
           saving output, etc.
        """
        Experiment_Paths = self.config['Experiment_Paths']

        self._project_path = self.normalize_path(Experiment_Paths['project_root_path'])
        self._output_path = self.normalize_path(Experiment_Paths['output_path'])
        self._data_path = self.normalize_path(Experiment_Paths['data_path'])
        self._transformer_path = self.normalize_path(Experiment_Paths['transformer_path'])

        # Optional debug info
        print("[DEBUG] Normalized paths:")
        print("  project_path:", self._project_path)
        print("  output_path:", self._output_path)
        print("  data_path:", self._data_path)
        print("  transformer_path:", self._transformer_path)

        self.path_exists(self._output_path, create_path=True)
        self.path_exists(self._data_path, create_path=True)
        self.path_exists(self._transformer_path, create_path=True)

    def path_exists(self, path: str, create_path=False):
        """Checks if a path exists. If it does, returns it.

        Args:
            path (str): Path to check or create.
            create_path (bool): If True, creates the path if it doesn't exist.
        """
        norm_path = self.normalize_path(path)

        if os.path.exists(norm_path):
            return norm_path

        if create_path:
            os.makedirs(norm_path, exist_ok=True)
            return norm_path
        else:
            raise ValueError("The following path doesn't exist: {}".format(norm_path))

    def dict2string(self, dictionary: dict, title: str):
        """Turns a dictionary into something pretty to print or write."""
        message = '\n\t' + title + '\n'
        info = ['\t\t' + str(key) + '=' + str(value) + '\n' for key, value in dictionary.items()]
        return message + ''.join(info)
