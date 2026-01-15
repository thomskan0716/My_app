from PySide6.QtCore import QObject, Signal
from Sample_Combiner import SampleCombiner

class SampleCombinerWorker(QObject):
    finished = Signal()
    error = Signal(str)

    def __init__(self, input_path, output_path):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path

    def run(self):
        try:
            SampleCombiner.generate_combinations(
                input_path=self.input_path,
                output_path=self.output_path
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
