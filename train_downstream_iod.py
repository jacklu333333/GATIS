import datetime
import faulthandler
import os
from argparse import ArgumentParser

import colored
import pytorch_lightning as pl
import torch

from myutils import *
from utils.mcallbacks import TimeProgressBar
from utils.weightFinder import WeightFinder

faulthandler.enable()

# from utils.complexstrategy import ComplexStrategy

if __name__ == "__main__":
    ###########################################################################################################################
    parser = ArgumentParser()
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--gpus", type=str, default="0")

    # parser.add_argument("--precision", type=int, default=1)
    # parser.add_argument("--precision", type=int, default=16)
    # parser.add_argument("--amp_level", type=str, default="O1")
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = parser.parse_args().gpus
    # torch.cuda.set_device(int(parser.parse_args().gpus))

    ###########################################################################################################################
    # num_gpus = torch.cuda.device_count()
    num_gpus = 1

    ###########################################################################################################################
    data_path = "datasets/migration/migration_wm"
    dm = IODDataModule(
        data_dir=f"{data_path}",
        # batch_size=BATCH_SIZE,
        num_workers=int(
            torch.get_num_threads()
            / torch.cuda.device_count()
            # / int(torch.cuda.mem_get_info()[1] / 24892145664)
        ),
        pin_memory=True,
        shuffle=True,
        label=True,
        whole_testing=False,
    )
    dm.setup("fit")
    print(
        f"{colored.Fore.red}Num Train Batch  {len(dm.train_dataloader())} {colored.Style.reset}"
    )

    ###########################################################################################################################
    model = IODModel(
        root=f"{data_path}",
        lr=0.07356422544596415,
        train_weight=dm.get_balanced_weight("train"),
        val_weight=dm.get_balanced_weight("val"),
        # TODO: find the solution to use these in the on_epoch_end without raise the error
        # num_batches=np.ceil(len(dm.train_dataloader()) / num_gpus),
        num_batches=len(dm.train_dataloader()) * 3,
        gradient_accumulation=False,
        # l1_lambda=0.0000005 / 5 * 100 ,
        l2_lambda=0.0005 * 10,
        # l2_lambda=0.0,
        include_top=False,
    )

    # # load the weight except for the mlp layer
    # model.load_from_checkpoint(
    #     "lightning_logs/iod-2024-01-23-04-05-32migration_wm_downstream/2024-01-23-04-05-32/checkpoints/iod-loss0.15136-F0.617-epoch172step00519.ckpt"
    # )

    # weight_finder = WeightFinder("./distance_logs/", "MotionID")
    weight_finder = WeightFinder(
        "./lightning_logs/", "iod-2024-06-11-15-55-58MotionID_upstream"
    )
    weight_index = 0
    model.load_upstream(
        weight_finder.weight_name[weight_index],
        # "lightning_logs/iod-2024-02-26-00-38-59multi_migration_unlabeled_upstream/2024-02-26-00-38-59/checkpoints/iod-loss_v1.39202-loss_t1.45062-epoch09step00829.ckpt"
    )

    print(
        colored.Fore.green
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f ")
        + f"complete model setup with datasets {data_path}"
        + colored.Style.reset
    )

    ###########################################################################################################################

    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    version = time
    logger = pl.loggers.TensorBoardLogger(
        save_dir="logs/IOD",
        name="iod-"
        + time
        + data_path.split("/")[-1].replace("datasets", "")
        + "_downstream",
        version=version,
        default_hp_metric=False,
    )

    print(
        colored.Fore.green
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f ")
        + f"complete logger setup with logdir {logger.log_dir}"
        + colored.Style.reset
    )

    ###########################################################################################################################
    # check whether has nvlink or not
    # get the output string of "nvidia-smi nvlink --status"
    nvlink = os.popen("nvidia-smi nvlink --status").read()
    if "Unable" in nvlink:
        nvlink = False
    else:
        nvlink = True
    # strategy = ComplexStrategy(num_gpus=num_gpus, nvlink=nvlink)

    ###########################################################################################################################
    trainer = pl.Trainer(
        # accelerator=args.accelerator,
        # strategy=args.strategy,
        strategy=pl.strategies.DDPStrategy(),
        devices=[int(i) for i in args.gpus.split(",")],
        num_nodes=int(os.getenv("WORLD_SIZE", "1")),
        # num_nodes=int(os.getenv("WORLD_SIZE", "1")),
        # max_epochs=10,
        # accumulate_grad_batches=4,
        callbacks=[
            # pl.callbacks.ProgressBar(),
            # add checkpoint callback only keep the best model
            pl.callbacks.ModelCheckpoint(
                auto_insert_metric_name=False,
                filename="iod-loss{loss/val:.5f}-F{f1/val:.3f}-epoch{epoch:02d}step{step:05d}",
                monitor="total_loss/val",
                mode="min",
                save_top_k=3,
                every_n_epochs=1,
                # every_n_train_steps=30,
                # train_time_interval=datetime.timedelta(minutes=15),
            ),
            # add early stopping callback
            pl.callbacks.EarlyStopping(
                monitor="f1/val", mode="max", patience=5  # model.hparams.patience
            ),
            # add learning rate decay callback
            pl.callbacks.ModelSummary(max_depth=-1),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            TimeProgressBar(),
        ],
        enable_progress_bar=True,
        # limit_train_batches=0.001,
        # limit_val_batches=0.001,
        # limit_test_batches=0.1,
        # limit_predict_batches=0.1,
        # configurethe logger name with the current time
        logger=logger,
        check_val_every_n_epoch=1,
        # val_check_interval=29,
        # gradient_clip_val=1.0,
        # gradient_clip_algorithm="norm",
        use_distributed_sampler=True,
        # max_time="00:07:00:00",
        min_epochs=1,
        # max_epochs=13,
        # max_steps=2,
        log_every_n_steps=1,
        # detect_anomaly=True,
        sync_batchnorm=True,
        # terminate_on_nan=True,
        # num_processes=3, ##TODO: test this function
        # profiler="advanced",
    )

    print(
        colored.Fore.green
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f ")
        + f"complete trainer setup with accelerator {args.accelerator}, strategy {args.strategy}, gpus {args.gpus}"
        + colored.Style.reset
    )

    ###########################################################################################################################
    # find lr
    tuner = pl.tuner.tuning.Tuner(trainer)
    lr_finder = tuner.lr_find(
        model=model,
        datamodule=dm,
        method="fit",
        max_lr=1e-1,
        min_lr=1e-5,
        num_training=100,
        # mode='linear',
        mode="exponential",
        # early_stop_threshold=None,
        update_attr=True,
        attr_name="lr",
    )
    fig = lr_finder.plot(suggest=True)
    # fig.show()
    new_lr = lr_finder.suggestion()
    if os.getenv("LOCAL_RANK", "0") == "0" and os.getenv("NODE_RANK", "0") == "0":
        fig.savefig("lightning_logs/" + logger.name + "/lr_finder.png")
        # model.save_hyperparameters()
    print(
        colored.Fore.green
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f ")
        + f"complete lr_finder with lr {new_lr}"
        + colored.Style.reset
    )
    # exit()
    ###########################################################################################################################
    trainer.fit(
        model,
        dm,
        # ckpt_path="lightning_logs/iod-2023-12-05-20-45-07MotionID_/checkpoints/iod-loss_v4.66598-loss_t4.62186-epoch00step00000.ckpt",
        # ckpt_path="lightning_logs/iod-2023-12-07-17-33-36MotionID/2023-12-07-17-33-36/checkpoints/iod-loss_v4.75767-loss_t4.76034-epoch28step00000.ckpt",
    )
    print(
        colored.Fore.green
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f ")
        + f"complete trainer.fit"
        + colored.Style.reset
    )

    ###########################################################################################################################

    # # set dm to test for migration_wm dataset
    # dm = IODDataModule(
    #     data_dir=f"/tmp/trainer/spectrums/migration_wm/",
    #     # batch_size=BATCH_SIZE,
    #     num_workers=0,
    #     pin_memory=False,
    #     shuffle=True,
    #     label=True,
    #     whole_testing=True,
    # )

    trainer.test(
        model,
        datamodule=dm,
        # ckpt_path='lightning_logs/iod-2023-12-25-08-25-30migration_wm/2023-12-25-08-25-30/checkpoints/iod-loss0.14014-F0.636-epoch15step00048.ckpt',
        ckpt_path=trainer.checkpoint_callback.best_model_path,
    )
    print(
        colored.Fore.green
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f ")
        + f"complete trainer.test"
        + colored.Style.reset
    )

    ###########################################################################################################################

    del dm
    data_path = "./datasets/ADVIO/advio_wm"
    dm_crossTest = IODDataModule(
        data_dir=f"{data_path}",
        # batch_size=BATCH_SIZE,
        num_workers=int(
            torch.get_num_threads()
            / torch.cuda.device_count()
            # / int(torch.cuda.mem_get_info()[1] / 24892145664)
        ),
        pin_memory=False,
        shuffle=True,
        label=True,
        whole_testing=True,
    )
    # data_path = "/tmp/trainer/spectrums/migration_wm"
    # dm_crossTest = IODDataModule(
    #     data_dir=f"{data_path}",
    #     # batch_size=BATCH_SIZE,
    #     num_workers=int(
    #         torch.get_num_threads()
    #         / torch.cuda.device_count()
    #         / int(torch.cuda.mem_get_info()[1] / 24892145664)
    #     ),
    #     pin_memory=True,
    #     shuffle=True,
    #     label=True,
    #     whole_testing=False,
    # )
    # dm_crossTest.setup("fit")

    logger_test = pl.loggers.TensorBoardLogger(
        save_dir=logger.save_dir + "/" + logger.name,
        name="test-" + data_path.split("/")[-1].replace("datasets", ""),
        version=version,
        default_hp_metric=False,
    )
    trainer.logger = logger_test
    print(
        colored.Fore.green
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f ")
        + f"complete dm_crossTest setup with {args.accelerator} and {args.strategy}"
        + colored.Style.reset
    )

    trainer.test(
        model,
        datamodule=dm_crossTest,
        ckpt_path=trainer.checkpoint_callback.best_model_path,
    )
    print(
        colored.Fore.green
        + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f ")
        + f"complete tester.test with {args.accelerator} and {args.strategy}"
        + colored.Style.reset
    )
    ###########################################################################################################################
    # del dm_crossTest
    # dm_test = NoiseModule()
    # logger_test = pl.loggers.TensorBoardLogger(
    #     save_dir=logger.save_dir + "/" + logger.name,
    #     name="test-" + "noise",
    #     version=version,
    # )
    # # swap the logger
    # trainer.logger = logger_test

    # dm_test.setup("test")
    # trainer.test(
    #     model,
    #     datamodule=dm_test,
    #     ckpt_path=trainer.checkpoint_callback.best_model_path,
    # )
