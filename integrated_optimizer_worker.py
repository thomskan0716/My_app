# integrated_optimizer_worker.py

from PySide6.QtCore import QObject, Signal
from integrated_optimizer import run_integrated_optimizer


class IntegratedOptimizerWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, sample_file, existing_file=None, output_folder=".", num_points=15, 
                 sample_size=None, enable_hyperparameter_tuning=True, force_reoptimization=False, optimization_type="both"):
        super().__init__()
        self.sample_file = sample_file
        self.existing_file = existing_file
        self.output_folder = output_folder
        self.num_points = num_points
        self.sample_size = sample_size
        self.enable_hyperparameter_tuning = enable_hyperparameter_tuning
        self.force_reoptimization = force_reoptimization
        self.optimization_type = optimization_type

    def run(self):
        try:
            result = run_integrated_optimizer(
                sample_file=self.sample_file,
                existing_data_file=self.existing_file,
                output_folder=self.output_folder,
                num_experiments=self.num_points,
                sample_size=self.sample_size,
                enable_hyperparameter_tuning=self.enable_hyperparameter_tuning,
                force_reoptimization=self.force_reoptimization,
                optimization_type=self.optimization_type
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e)) 