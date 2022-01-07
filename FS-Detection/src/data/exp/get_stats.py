def get_stats_of_stage(path):

    # Opening the requested file
    with open(path, 'r', encoding='utf-8') as file:
        # Ignoring the newline character (\n) at the end of each line
        log_file = [x[:-2] if x.endswith("\n") else x for x in file.readlines()]

    epochs = []
    steps = []
    train_losses = []
    dev_ap = []

    for i, line in enumerate(log_file, 0):
        if not line:
            continue
        else:
            if 'loss' in line:
                line = line.split(' ')
                epochs.append(int(line[1][:-1]))
                steps.append(int(line[3][:-1]))
                train_losses.append(float(line[6][:-1]))
            if 'AP' in line:
                line = line.split(' ')
                dev_ap.append(float(line[6]))

    print(epochs)
    print(steps)
    print(train_losses)
    print(dev_ap)


print('Stage1:')
get_stats_of_stage('./stage1/log')
print('\nStage2:')
get_stats_of_stage('./stage2/log')
