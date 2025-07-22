import subprocess
import sys
import re
import time
import os

def clear_terminal():
    os.system('clear' if os.name == 'posix' else 'cls')

def run_and_check():
    """
    Run resonator_work_test.py and check for multi-class classification accuracy > 90%.
    Print only the output from resonator_work_test.py live. Stop as soon as the target is reached.
    """
    target_accuracy = 0.90
    accuracy_pattern = re.compile(r"Final Test Accuracy:\s*([0-9.]+)")
    multiclass_section_pattern = re.compile(r"MULTI-CLASS ENSEMBLE SNN TRAINING PIPELINE")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    while True:
        clear_terminal()
        print("===== Running resonator_work_test.py =====\n")
        proc = subprocess.Popen(
            [sys.executable, "resonator_work_test.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=script_dir  # Ensure correct working directory
        )
        output_lines = []
        found_multiclass_section = False
        found_accuracy = False
        accuracy = None
        try:
            for line in proc.stdout:
                print(line, end='')  # Print each line as it is produced
                output_lines.append(line)
                if not found_multiclass_section and multiclass_section_pattern.search(line):
                    found_multiclass_section = True
                if found_multiclass_section:
                    match = accuracy_pattern.search(line)
                    if match:
                        try:
                            accuracy = float(match.group(1))
                            found_accuracy = True
                            if accuracy >= target_accuracy:
                                print("\n===== SUCCESS: Multi-class accuracy above 90%! =====\n")
                                # Print the rest of the output if any
                                for rest_line in proc.stdout:
                                    print(rest_line, end='')
                                proc.terminate()
                                return
                            else:
                                print(f"Accuracy {accuracy:.4f} < 0.90, rerunning...\n")
                                break
                        except Exception as e:
                            print(f"Error parsing accuracy: {e}")
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            print("\nInterrupted by user.")
            break
        if not found_accuracy:
            print("Could not find multi-class accuracy in output. Rerunning...\n")
        time.sleep(2)

if __name__ == "__main__":
    run_and_check() 