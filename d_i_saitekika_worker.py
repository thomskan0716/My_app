from PySide6.QtCore import QObject, Signal
from d_i_saitekika import run_d_i_saitekika

class DISaitekikaWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, input_excel_path, output_excel_path, output_prefix, num_points):
        super().__init__()
        self.input_excel_path = input_excel_path
        self.output_excel_path = output_excel_path
        self.output_prefix = output_prefix
        self.num_points = num_points

    def run(self):
        try:
            results = run_d_i_saitekika(
                input_excel_path=self.input_excel_path,
                output_excel_path=self.output_excel_path,
                output_prefix=self.output_prefix,
                num_points=self.num_points,
            )
            self.dsaitekika_selected_df = results['selected_dataframe']
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
