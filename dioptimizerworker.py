# dioptimizerworker.py

from PySide6.QtCore import QObject, Signal
from d_i_optimizer import run_d_i_optimizer


class DIOptimizerWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, sample_file, existing_file, output_folder, num_points=15):
        super().__init__()
        self.sample_file = sample_file
        self.existing_file = existing_file
        self.output_folder = output_folder
        self.num_points = num_points

    def run(self):
        try:
            result = run_d_i_optimizer(
                sample_file=self.sample_file,
                existing_data_file=self.existing_file,
                output_folder=self.output_folder,
                num_experiments=self.num_points
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
