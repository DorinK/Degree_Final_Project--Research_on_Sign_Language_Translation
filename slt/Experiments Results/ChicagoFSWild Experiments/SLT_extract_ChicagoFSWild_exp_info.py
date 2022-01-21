"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 Extracting information from SLT model experiment on ChicagoFSWild Dataset
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

EXP_PATH = 'Regular Experiments (threshold = 131 frames)/ChicagoFSWild_experiment_19 - Regular experiment -> One Full Experiment! (Good, Poor results)'
OUTPUT_FILE = EXP_PATH + '/experiment_stats.txt'


def extract_experiment_info(path):
    """
        Function for extracting the training batch translation loss, the development translation loss,
        WAcc and Sequence Accuracy scores.
    """

    # Open the log file of the requested experiment.
    with open(path, 'r', encoding='utf-8') as file:

        # Ignore newline character (\n) at the end of each line.
        log_file = [x[:-2] if x.endswith("\n") else x for x in file.readlines()]

    epochs = []
    steps = []
    train_batch_translation_loss = []
    dev_translation_loss = []
    wacc = []
    seq_accuracy = []

    for i, line in enumerate(log_file, 0):

        if 'Training ended' in line:  # Stop when you reach the end of the training process.
            break

        if not line:  # Ignore empty lines.
            continue

        else:
            if '[Epoch:' in line:

                line_split = line.split(' ')
                epochs.append(int(line_split[3]))  # Save the epoch number.
                steps.append(int(line_split[5][:-1]))  # Save the steps number.

                indexes = [index for index, ele in enumerate(line_split) if ele == "Loss:"]

                try:  # Save the training batch translation loss.
                    train_batch_translation_loss.append(float(line_split[indexes[0] + 2]))
                except:
                    try:
                        train_batch_translation_loss.append(float(line_split[indexes[0] + 1]))
                    except:
                        train_batch_translation_loss.append(float(line_split[indexes[0] + 3]))

            if line.startswith('\t'):

                if 'Recognition Loss:' in line:
                    line_split = line.split('\t')
                    for j, part in enumerate(line_split, 0):
                        line_split[j] = part.split(' ')
                    dev_translation_loss.append(float(line_split[2][2]))  # Save the dev translation loss.

                elif line[1:].startswith('WAcc'):
                    line_split = line.split('\t')
                    for j, part in enumerate(line_split, 0):
                        line_split[j] = part.split(' ')
                    wacc.append(float(line_split[1][4]))  # Save the WAcc score.

                elif line[1:].startswith('Sequence'):
                    line_split = line.split('\t')
                    for j, part in enumerate(line_split, 0):
                        line_split[j] = part.split(' ')
                    seq_accuracy.append(float(line_split[1][2]))  # Save the Sequence Accuracy score.

        # Write the extracted information about the experiment to the output file.
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:

            f.write(f'Stats of SLT Experiment on ChicagoFSWild dataset:\n\n')
            f.write(f'Epochs:\n{epochs}\n\n')
            f.write(f'Steps:\n{steps}\n\n')
            f.write(f'Train Batch Translation Losses:\n{train_batch_translation_loss}\n\n')
            f.write(f'Train Batch Translation Losses per 100:\n'
                    f'{[loss for i, loss in enumerate(train_batch_translation_loss) if (i + 1) % 100 == 0]}\n\n')
            f.write(f'Dev Translation Losses:\n{dev_translation_loss}\n\n')
            f.write(f'WAcc:\n{wacc}\n\n')
            f.write(f'Sequence Accuracy:\n{seq_accuracy}')


if __name__ == '__main__':
    # Extract the experiment information.
    extract_experiment_info(EXP_PATH + '/train.log')
