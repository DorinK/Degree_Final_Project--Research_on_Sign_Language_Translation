"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
  Extracting information from an FS-Detector model experiment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

FIRST_STAGE_PATH = './stage1'
SECOND_STAGE_PATH = './stage2'
OUTPUT_FILE = '_stats.txt'


def extract_experiment_info(path):
    """
        Function for extracting the training losses and development accuracies, over epochs and steps.
    """

    # Open the log file of the requested experiment.
    with open(path, 'r', encoding='utf-8') as file:

        # Ignore newline character (\n) at the end of each line.
        log_file = [x[:-2] if x.endswith("\n") else x for x in file.readlines()]

    epochs = []
    steps = []
    train_losses = []
    dev_ap = []

    for i, line in enumerate(log_file, 0):

        if not line:  # Ignore empty lines.
            continue
        else:
            if 'loss' in line:
                line = line.split(' ')
                epochs.append(int(line[1][:-1]))  # Save the epoch number.
                steps.append(int(line[3][:-1]))  # Save the step number.
                train_losses.append(float(line[6][:-1]))  # Save the train loss.
            if 'AP' in line:
                line = line.split(' ')
                dev_ap.append(float(line[6]))  # Save the dev AP.

        # Write the extracted information about the experiment to the output file.
        with open(path[:8] + '/' + path[2:8] + OUTPUT_FILE, 'w', encoding='utf-8') as f:

            f.write(f'Stats of FS-Detector Experiment on ChicagoFSWild Dataset - {path[2:8]}:\n\n')
            f.write(f'Epochs:\n{epochs}\n\n')
            f.write(f'Steps:\n{steps}\n\n')
            f.write(f'Train Losses:\n{train_losses}\n\n')
            f.write(f'Dev APs:\n{dev_ap}')


if __name__ == '__main__':

    # Extract the information of the first-stage of training in the experiment.
    extract_experiment_info(FIRST_STAGE_PATH + '/log')
    # Extract the information of the second-stage of training in the experiment.
    extract_experiment_info(SECOND_STAGE_PATH + '/log')
