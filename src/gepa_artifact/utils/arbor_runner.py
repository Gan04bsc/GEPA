import os
import subprocess
import time
import uuid

class ArborRunner:
    """
    Context manager for running Arbor.
    """
    def __init__(self, config_filepath, portnum, rundirname):
        self.config_filepath = config_filepath
        self.portnum = portnum
        self.rundirname = rundirname
        self.session_name = None

    def __enter__(self):

        self.session_name = "arbor_" + uuid.uuid4().hex
        port_num = str(self.portnum)
        output_log_filename = os.path.join(self.rundirname, "arbor_logs.txt")

        # Construct the command to be run inside tmux
        if os.environ.get("WANDB_API_KEY") is None:
            command_to_run_in_tmux = (
                f"source env.sh && "
                f"uv run python -u -m arbor.cli serve --arbor-config {self.config_filepath} --port {port_num} "
                f"|& tee -a {output_log_filename}; bash"
            )
        else:
            command_to_run_in_tmux = (
                f"source env.sh  && export WANDB_API_KEY={os.environ.get('WANDB_API_KEY')} && "
                f"uv run python -u -m arbor.cli serve --arbor-config {self.config_filepath} --port {port_num} "
                f"|& tee -a {output_log_filename}; bash"
            )

        command_to_run_in_tmux = f"bash -i -c '{command_to_run_in_tmux}'"

        # Create the new detached tmux session and run the command
        try:
            # Using subprocess.run()
            # The check=True argument will raise a CalledProcessError if the command returns a non-zero exit code.

            # ExecStart=${TMUX_PATH} new-session -d -s ${SESSION_NAME} '${SCRIPT_PATH}'
            # ExecStop=${TMUX_PATH} kill-session -t ${SESSION_NAME}

            completed_process = subprocess.run(
                ["/usr/bin/tmux", "new-session", "-d", "-s", self.session_name, command_to_run_in_tmux],
                text=True,
                check=True  # Optional: raises an exception if tmux command fails
            )
            print(f"Tmux session '{self.session_name}' started successfully.")
            # completed_process.returncode will be 0 if successful
            # completed_process.stdout and completed_process.stderr would capture output from the tmux command itself,
            # but for 'tmux new-session -d', this is usually minimal or none.
        except FileNotFoundError:
            print("Error: tmux command not found. Please ensure tmux is installed and in your PATH.")
        except subprocess.CalledProcessError as e:
            print(f"Error starting tmux session: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        time.sleep(10)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            # Command to kill the specific pane
            kill_command = ["tmux", "kill-session", "-t", self.session_name]
            subprocess.run(kill_command, check=True, text=True)
            print(f"Tmux session '{self.session_name}' killed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error killing tmux pane: {e}")
            print("The pane or session might have already been closed or the target is incorrect.")
        except FileNotFoundError:
            print("Error: tmux command not found. Please ensure tmux is installed and in your PATH.")
        except Exception as e:
            print(f"An error occurred: {e}")