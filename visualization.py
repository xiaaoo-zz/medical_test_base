from datetime import datetime
import visdom


class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)

        self.loss_step_win = None
        self.loss_epoch_win = None

        self.precision_epoch_win = None
        self.recall_epoch_win = None
        self.accuracy_epoch_win = None

        self.precision_epoch_win2 = None
        self.recall_epoch_win2 = None
        self.accuracy_epoch_win2 = None

        self.precision_epoch_win3 = None
        self.recall_epoch_win3 = None
        self.accuracy_epoch_win3 = None

    def plot_loss(self, loss, step):
        self.loss_step_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_step_win,
            update='append' if self.loss_step_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss (mean per 100 steps)',
            )
        )

    def plot_epoch(self, loss, epoch):
        self.loss_epoch_win = self.vis.line(
            [loss],
            [epoch],
            win=self.loss_epoch_win,
            update='append' if self.loss_epoch_win else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Loss (mean-epoch)',
            )
        )

    # 2017
    def plot_precision(self, precision, epoch):
        self.precision_epoch_win = self.vis.line(
            [precision],
            [epoch],
            win=self.precision_epoch_win,
            update='append' if self.precision_epoch_win else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Precision',
                title='2017 Precision-Epoch',
            )
        )

    def plot_recall(self, recall, epoch):
        self.recall_epoch_win = self.vis.line(
            [recall],
            [epoch],
            win=self.recall_epoch_win,
            update='append' if self.recall_epoch_win else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Recall',
                title='2017 Recall-Epoch',
            )
        )

    def plot_accuracy(self, accuracy, epoch):
        self.accuracy_epoch_win = self.vis.line(
            [accuracy],
            [epoch],
            win=self.accuracy_epoch_win,
            update='append' if self.accuracy_epoch_win else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Accuracy',
                title='2017 Accuracy-Epoch',
            )
        )

    # 2018

    def plot_precision2(self, precision, epoch):
        self.precision_epoch_win2 = self.vis.line(
            [precision],
            [epoch],
            win=self.precision_epoch_win2,
            update='append' if self.precision_epoch_win2 else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Precision',
                title='2018 Precision-Epoch',
            )
        )

    def plot_recall2(self, recall, epoch):
        self.recall_epoch_win2 = self.vis.line(
            [recall],
            [epoch],
            win=self.recall_epoch_win2,
            update='append' if self.recall_epoch_win2 else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Recall',
                title='2018 Recall-Epoch',
            )
        )

    def plot_accuracy2(self, accuracy, epoch):
        self.accuracy_epoch_win2 = self.vis.line(
            [accuracy],
            [epoch],
            win=self.accuracy_epoch_win2,
            update='append' if self.accuracy_epoch_win2 else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Accuracy',
                title='2018 Accuracy-Epoch',
            )
        )

    # train_3000

    def plot_precision3(self, precision, epoch):
        self.precision_epoch_win3 = self.vis.line(
            [precision],
            [epoch],
            win=self.precision_epoch_win3,
            update='append' if self.precision_epoch_win3 else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Precision',
                title='Train_3000 Precision-Epoch',
            )
        )

    def plot_recall3(self, recall, epoch):
        self.recall_epoch_win3 = self.vis.line(
            [recall],
            [epoch],
            win=self.recall_epoch_win3,
            update='append' if self.recall_epoch_win3 else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Recall',
                title='Train_3000 Recall-Epoch',
            )
        )

    def plot_accuracy3(self, accuracy, epoch):
        self.accuracy_epoch_win3 = self.vis.line(
            [accuracy],
            [epoch],
            win=self.accuracy_epoch_win3,
            update='append' if self.accuracy_epoch_win3 else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Accuracy',
                title='Train_3000 Accuracy-Epoch',
            )
        )
