from create_exp import run_cmd

if __name__ == "__main__":

    # TODO: Update accordingly. V
    extra_args = """
        --num-classes 1233 \\
        --num_in_frames 16 \\
        --save_features 1 \\
        --include_embds 1 \\
        --test_set test \\
        --phoenix_path /home/nlp/dorink/project/bsl1k/data_phoenix \\
    """

    run_cmd(  # TODO: Update accordingly.   V
        dataset="phoenix2014",
        subfolder="bug",
        extra_args=extra_args,
        running_mode="test",
        modelno="_050",
        test_suffix="",
        num_gpus=4,
        jobsub=False,
        refresh=False,
    )

    # extra_args = """
    #     --num-classes 2000 \\
    #     --num_in_frames 64 \\
    #     --save_features 1 \\
    #     --include embds 1 \\
    #     --test_set test \\
    # """
    # run_cmd(
    #     dataset="wlasl",
    #     subfolder="my_experiment",
    #     extra_args=extra_args,
    #     running_mode="test",
    #     modelno="_050",
    #     test_suffix="",
    #     num_gpus=1,
    #     jobsub=False,
    #     refresh=False,
    # )
#         --checkpoint checkpoint/phoenix2014t_i3d_pkinetics
