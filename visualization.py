#%%
from datetime import datetime
import visdom


class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        
        self.accuracy_epoch_win1 = None
        self.accuracy_epoch_win2 = None
        self.accuracy_epoch_win3 = None

    def plot_accuracy1(self, accuracy, epoch):
        self.accuracy_epoch_win1 = self.vis.line(
            [accuracy],
            [epoch],
            win=self.accuracy_epoch_win1,
            update='append' if self.accuracy_epoch_win1 else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Accuracy',
                title='2017 new Accuracy-Epoch',
            )
        )


    # 2018
    def plot_accuracy2(self, accuracy, epoch):
        self.accuracy_epoch_win2 = self.vis.line(
            [accuracy],
            [epoch],
            win=self.accuracy_epoch_win2,
            update='append' if self.accuracy_epoch_win2 else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Accuracy',
                title='2018 new Accuracy-Epoch',
            )
        )

    # train_3000
    def plot_accuracy3(self, accuracy, epoch):
        self.accuracy_epoch_win3 = self.vis.line(
            [accuracy],
            [epoch],
            win=self.accuracy_epoch_win3,
            update='append' if self.accuracy_epoch_win3 else None,
            opts=dict(
                xlabel='Epoch',
                ylabel='Accuracy',
                title='Train 3000 new Accuracy-Epoch',
            )
        )


#%%
