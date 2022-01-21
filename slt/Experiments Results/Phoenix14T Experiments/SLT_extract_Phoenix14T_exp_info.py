"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
  Extracting information from SLT model experiment on Phoenix14T Dataset
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

EXP_PATH = 'Phoenix14T_experiment_09 - WER 28.06 & BLEU-4 21.36'
OUTPUT_FILE = EXP_PATH + '/experiment_stats.txt'


def extract_experiment_info(path):
    """
        Function for extracting the training batch translation & recognition losses, the development
        translation & recognition losses, WER and BLEU-4 scores.
    """

    # Open the log file of the requested experiment.
    with open(path, 'r', encoding='utf-8') as file:

        # Ignore newline character (\n) at the end of each line.
        log_file = [x[:-2] if x.endswith("\n") else x for x in file.readlines()]

    epochs = []
    steps = []
    train_batch_recognition_loss = []
    train_batch_translation_loss = []
    dev_recognition_loss = []
    dev_translation_loss = []
    wer = []
    bleu_4 = []

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

                try:  # Save the training batch recognition loss.
                    train_batch_recognition_loss.append(float(line_split[indexes[0] + 2]))
                except:
                    try:
                        train_batch_recognition_loss.append(float(line_split[indexes[0] + 1]))
                    except:
                        train_batch_recognition_loss.append(float(line_split[indexes[0] + 3]))

                try:  # Save the training batch translation loss.
                    train_batch_translation_loss.append(float(line_split[indexes[1] + 2]))
                except:
                    try:
                        train_batch_translation_loss.append(float(line_split[indexes[1] + 1]))
                    except:
                        train_batch_translation_loss.append(float(line_split[indexes[1] + 3]))

            if line.startswith('\t'):

                if 'Recognition Loss:' in line:
                    line_split = line.split('\t')
                    for j, part in enumerate(line_split, 0):
                        line_split[j] = part.split(' ')
                    dev_recognition_loss.append(float(line_split[1][2]))  # Save the dev recognition loss.
                    dev_translation_loss.append(float(line_split[2][2]))  # Save the dev translation loss.

                elif line[1:].startswith('WER'):
                    line_split = line.split('\t')
                    for j, part in enumerate(line_split, 0):
                        line_split[j] = part.split(' ')
                    wer.append(float(line_split[1][1]))  # Save the WER score.

                elif line[1:].startswith('BLEU-4'):
                    line_split = line.split('\t')
                    for j, part in enumerate(line_split, 0):
                        line_split[j] = part.split(' ')
                    bleu_4.append(float(line_split[1][1]))  # Save the BLEU-4 score.

        # Write the extracted information about the experiment to the output file.
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:

            f.write(f'Stats of SLT Experiment on Phoenix14T dataset:\n\n')
            f.write(f'Epochs:\n{epochs}\n\n')
            f.write(f'Steps:\n{steps}\n\n')
            f.write(f'Train Batch Recognition Losses:\n{train_batch_recognition_loss}\n\n')
            f.write(f'Train Batch Translation Losses:\n{train_batch_translation_loss}\n\n')
            f.write(f'Dev Recognition Losses:\n{dev_recognition_loss}\n\n')
            f.write(f'Dev Translation Losses:\n{dev_translation_loss}\n\n')
            f.write(f'WER:\n{wer}\n\n')
            f.write(f'BLEU-4:\n{bleu_4}')


if __name__ == '__main__':
    # Extract the experiment information of the best model found for Phoenix14T dataset.
    extract_experiment_info(EXP_PATH + '/train.log')
