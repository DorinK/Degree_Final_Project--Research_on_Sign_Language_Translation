"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
  Extracting information from an I3D model experiment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

EXP_PATH = './phoenix2014t_dec_bsl1k_pretraining_batch_4'
OUTPUT_FILE = EXP_PATH + '/experiment_stats.txt'


def extract_experiment_info(path):
    """
        Function for extracting the training and development losses and per-instance
        accuracies, over epochs.
    """

    # Open the log file of the requested experiment.
    with open(path, 'r', encoding='utf-8') as file:

        # Ignore newline character (\n) at the end of each line.
        log_file = [x[:-2] if x.endswith("\n") else x for x in file.readlines()]

    train_losses = []
    dev_losses = []
    train_accuracies = []
    dev_accuracies = []

    for i, line in enumerate(log_file, 0):

        if not line:  # Ignore empty lines.
            continue
        else:
            if i != 0:
                line = line.split('\t')
                train_losses.append(float(line[2]))  # Save the train loss.
                dev_losses.append(float(line[3]))  # Save the dev loss.
                train_accuracies.append(float(line[4]))  # Save the train per-instance accuracy.
                dev_accuracies.append(float(line[5]))  # Save the dev per-instance accuracy.

        # Write the extracted information about the experiment to the output file.
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:

            f.write(f'Stats of Experiment {EXP_PATH[2:]}:\n\n')
            f.write(f'Train losses:\n{train_losses}\n\n')
            f.write(f'Dev losses:\n{dev_losses}\n\n')
            f.write(f'Train per-instance accuracies:\n{train_accuracies}\n\n')
            f.write(f'Dev per-instance accuracies:\n{dev_accuracies}')


if __name__ == '__main__':
    # Extract the experiment information of the best model found for Phoenix14T dataset.
    extract_experiment_info(EXP_PATH + '/log.txt')
