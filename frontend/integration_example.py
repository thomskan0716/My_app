"""
Example integration of API client with existing GUI.
This shows how to modify the existing 0sec.py to use the backend API.
"""
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QMessageBox

from frontend.api_client import create_client, APIClientError, JobTimeoutError
from frontend.models import JobType


class RemoteAnalysisWorker(QThread):
    """
    Worker thread for remote analysis via API.
    Replaces the existing local workers (DsaitekikaWorker, NonlinearWorker, etc.)
    """
    
    # Signals
    progress_updated = Signal(int, str)  # (percent, message)
    finished = Signal(dict)  # results dict
    error = Signal(str)  # error message
    
    def __init__(
        self,
        api_base_url: str,
        input_file: Path,
        job_type: JobType,
        parameters: dict,
        output_dir: Path,
        parent=None
    ):
        super().__init__(parent)
        
        self.api_base_url = api_base_url
        self.input_file = input_file
        self.job_type = job_type
        self.parameters = parameters
        self.output_dir = output_dir
        
        self.downloaded_files = []
    
    def run(self):
        """Execute remote analysis"""
        try:
            # Create API client
            client = create_client(self.api_base_url)
            
            # Progress callback
            def progress_cb(message, percent):
                self.progress_updated.emit(percent, message)
            
            # Run complete workflow
            self.progress_updated.emit(0, "Connecting to server...")
            
            output_files = client.submit_and_wait(
                file_path=self.input_file,
                job_type=self.job_type,
                parameters=self.parameters,
                output_dir=self.output_dir,
                progress_callback=progress_cb
            )
            
            self.downloaded_files = output_files
            
            # Emit success
            result = {
                'status': 'completed',
                'output_files': [str(f) for f in output_files],
                'output_dir': str(self.output_dir)
            }
            
            self.finished.emit(result)
            
        except JobTimeoutError as e:
            self.error.emit(f"Job timed out: {e}")
        except APIClientError as e:
            self.error.emit(f"API error: {e}")
        except Exception as e:
            self.error.emit(f"Unexpected error: {e}")


# ===== Example: How to modify existing button handlers =====

def example_d_optimization_handler_OLD():
    """OLD VERSION - Local processing"""
    # This is how it currently works (from 0sec.py)
    from dsaitekikaworker import DsaitekikaWorker
    
    worker = DsaitekikaWorker(
        input_excel_path="input.xlsx",
        output_excel_path="output.xlsx",
        output_prefix="D_opt",
        num_points=15
    )
    worker.finished.connect(lambda result: print("Done!"))
    worker.error.connect(lambda error: print(f"Error: {error}"))
    worker.start()


def example_d_optimization_handler_NEW(self):
    """NEW VERSION - Remote processing via API"""
    
    # Check if input file is loaded
    if not hasattr(self, 'loaded_file_path'):
        QMessageBox.warning(self, "Error", "Please load a data file first")
        return
    
    # Get API base URL from config
    from frontend.api_client import create_client
    from frontend.models import JobType
    
    api_url = "http://ec2-instance-ip:8000/api/v1"  # From .env file
    
    # Create output directory
    output_dir = Path("./results/D_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set parameters
    parameters = {
        "objective": "minimize_wear",
        "num_points": 15,
        "iterations": 50,
        "random_seed": 42
    }
    
    # Create and start worker
    self.remote_worker = RemoteAnalysisWorker(
        api_base_url=api_url,
        input_file=Path(self.loaded_file_path),
        job_type=JobType.OPTIMIZATION,
        parameters=parameters,
        output_dir=output_dir
    )
    
    # Connect signals
    self.remote_worker.progress_updated.connect(self._on_progress_update)
    self.remote_worker.finished.connect(self._on_d_optimization_finished)
    self.remote_worker.error.connect(self._on_error)
    
    # Show progress dialog
    self._show_progress_dialog("Running D-optimization...")
    
    # Start worker
    self.remote_worker.start()


def example_nonlinear_handler_NEW(self):
    """NEW VERSION - Nonlinear analysis via API"""
    
    # Check input
    if not hasattr(self, 'filtered_df') or self.filtered_df.empty:
        QMessageBox.warning(self, "Error", "Please load and filter data first")
        return
    
    # Show configuration dialog (existing)
    from nonlinear_config_dialog import NonlinearConfigDialog
    
    dialog = NonlinearConfigDialog(parent=self)
    if dialog.exec() != dialog.Accepted:
        return
    
    config_values = dialog.get_values()
    
    # Save filtered data to temp file
    import tempfile
    temp_file = Path(tempfile.mktemp(suffix=".csv"))
    self.filtered_df.to_csv(temp_file, index=False)
    
    # Create output directory
    output_dir = Path("./results/nonlinear_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare parameters
    parameters = {
        "target_columns": config_values.get("TARGET_COLUMNS", []),
        "models_to_use": config_values.get("MODELS_TO_USE", []),
        "outer_splits": config_values.get("OUTER_SPLITS", 10),
        "n_trials": config_values.get("N_TRIALS", 50),
        "enable_shap": config_values.get("ENABLE_SHAP", True),
        "enable_pareto": config_values.get("ENABLE_PARETO", True),
    }
    
    # Create and start worker
    api_url = "http://ec2-instance-ip:8000/api/v1"
    
    self.remote_worker = RemoteAnalysisWorker(
        api_base_url=api_url,
        input_file=temp_file,
        job_type=JobType.NONLINEAR_ANALYSIS,
        parameters=parameters,
        output_dir=output_dir
    )
    
    # Connect signals
    self.remote_worker.progress_updated.connect(self._on_progress_update)
    self.remote_worker.finished.connect(self._on_nonlinear_finished)
    self.remote_worker.error.connect(self._on_error)
    
    # Show progress dialog
    self._show_progress_dialog("Running nonlinear analysis...")
    
    # Start worker
    self.remote_worker.start()


# ===== Example: Progress dialog helper =====

def _show_progress_dialog(self, title: str):
    """Show progress dialog for remote analysis"""
    from PySide6.QtWidgets import QProgressDialog
    
    self.progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self)
    self.progress_dialog.setWindowModality(Qt.WindowModal)
    self.progress_dialog.setAutoClose(False)
    self.progress_dialog.setAutoReset(False)
    self.progress_dialog.show()


def _on_progress_update(self, percent: int, message: str):
    """Handle progress updates from worker"""
    if hasattr(self, 'progress_dialog'):
        self.progress_dialog.setValue(percent)
        self.progress_dialog.setLabelText(message)


def _on_d_optimization_finished(self, result: dict):
    """Handle D-optimization completion"""
    # Close progress dialog
    if hasattr(self, 'progress_dialog'):
        self.progress_dialog.close()
    
    # Show success message
    output_dir = result['output_dir']
    num_files = len(result['output_files'])
    
    QMessageBox.information(
        self,
        "Completed",
        f"D-optimization completed successfully!\n\n"
        f"{num_files} files saved to:\n{output_dir}"
    )
    
    # Open output folder (optional)
    import os
    if os.name == 'nt':  # Windows
        os.startfile(output_dir)
    else:  # macOS/Linux
        import subprocess
        subprocess.call(['open' if os.name == 'darwin' else 'xdg-open', output_dir])


def _on_nonlinear_finished(self, result: dict):
    """Handle nonlinear analysis completion"""
    # Close progress dialog
    if hasattr(self, 'progress_dialog'):
        self.progress_dialog.close()
    
    # Show results dialog (existing)
    from pareto_results_dialog import ParetoResultsDialog
    
    output_dir = Path(result['output_dir'])
    
    # Find Pareto plots folder
    pareto_folder = output_dir / "05_Pareto分析" / "Pareto_Plots"
    
    # Find prediction output file
    prediction_file = output_dir / "04_予測値" / "prediction_output.xlsx"
    
    if pareto_folder.exists() and prediction_file.exists():
        dialog = ParetoResultsDialog(
            pareto_plots_folder=str(pareto_folder),
            prediction_output_file=str(prediction_file),
            parent=self
        )
        dialog.exec()
    else:
        QMessageBox.information(
            self,
            "Completed",
            f"Nonlinear analysis completed!\n\n"
            f"Results saved to:\n{result['output_dir']}"
        )


def _on_error(self, error_message: str):
    """Handle worker errors"""
    # Close progress dialog
    if hasattr(self, 'progress_dialog'):
        self.progress_dialog.close()
    
    # Show error message
    QMessageBox.critical(
        self,
        "Error",
        f"Analysis failed:\n\n{error_message}"
    )


# ===== Example: Configuration loader =====

def load_api_config():
    """Load API configuration from .env file"""
    from dotenv import load_dotenv
    import os
    
    # Load .env file
    load_dotenv("frontend/.env")
    
    # Get configuration
    config = {
        'api_base_url': os.getenv('API_BASE_URL', 'http://localhost:8000/api/v1'),
        'poll_interval': int(os.getenv('STATUS_POLL_INTERVAL_SECONDS', 2)),
        'max_wait_minutes': int(os.getenv('MAX_POLL_TIME_MINUTES', 120)),
    }
    
    return config


# ===== Example: Main window modification =====

class ModifiedMainWindow:
    
    def __init__(self):
        # ... existing __init__ code ...
        
        # NEW: Load API configuration
        from dotenv import load_dotenv
        import os
        load_dotenv("frontend/.env")
        self.api_base_url = os.getenv('API_BASE_URL')
        
        # ... rest of existing __init__ code ...
    
    def on_d_optimization_button_clicked(self):
        """MODIFIED: Use remote worker instead of local worker"""
        # OLD CODE (comment out):
        # from dsaitekikaworker import DsaitekikaWorker
        # self.worker = DsaitekikaWorker(...)
        # self.worker.start()
        
        # NEW CODE:
        example_d_optimization_handler_NEW(self)
    
    def on_nonlinear_button_clicked(self):
        """MODIFIED: Use remote worker instead of local worker"""
        # OLD CODE (comment out):
        # from nonlinear_worker import NonlinearWorker
        # self.worker = NonlinearWorker(...)
        # self.worker.start()
        
        # NEW CODE:
        example_nonlinear_handler_NEW(self)
    
    # ... all other existing methods remain unchanged ...


# ===== Summary of Changes =====

"""
To integrate the API client with existing 0sec.py:

1. Install frontend dependencies:
   pip install -r frontend/requirements.txt

2. Create frontend/.env file:
   API_BASE_URL=http://<EC2_IP>:8000/api/v1

3. Modify 0sec.py:
   - Add API config loading in __init__
   - Replace local workers with RemoteAnalysisWorker
   - Update signal handlers to work with new results format

4. Keep unchanged:
   - All UI code (buttons, layouts, dialogs)
   - Result display logic (graphs, tables)
   - Configuration dialogs
   - File loading/filtering logic

The UI looks and feels exactly the same to the user, but processing 
happens remotely on AWS instead of locally.
"""
