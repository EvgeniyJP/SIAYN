import datetime
import os


class Logger:
    """
    Logger class was build for control over log, more comfortable printing, time tags,
     different colors for each type of information and option to save the logs
    """
    def __init__(self, save=False, p_save='logs'):

        """
        Transfer parameters to Logger during initialization
        :param save: True - saving the logs in p_save, False - doesn't save. Default - False
        :param p_save: Path top save logs, default 'logs'
        """

        # Creating time tag for initialization of log
        self.created_date = datetime.datetime.now().strftime("%Y%M%d%H%M%S")
        self.save = save
        self.p_log = os.path.join(p_save, self.created_date + '.log')

        # Create output directory
        if self.save:
            os.makedirs(p_save, exist_ok=True)

    def title(self, *args):
        """
        Create log of type TITLE
        :param args: the arguments that needed to be logged

        """
        try:
            text = '[TITLE]:' + datetime.datetime.now().strftime("%H:%M:%S---") + '-' * 25
            for t in args:
                text += ' '
                text += str(t)

            text += ' '
            text += '-' * 25
            print('\033[94m' + text + '\033[0m')
            if self.save:
                with open(self.p_log, "a") as f:
                    f.write(text + '\n')
        except Exception as e:
            self.error('Previous TITLE message cannot be logged..', str(e))

    def info(self, *args):
        """
        Create log of type INFO
        :param args: the arguments that needed to be logged

        """

        try:
            text = '[INFO]:' + datetime.datetime.now().strftime("%H:%M:%S---")
            for t in args:
                text += ' '
                text += str(t)
            print('\033[96m' + text + '\033[0m')
            if self.save:
                with open(self.p_log, "a") as f:
                    f.write(text + '\n')
        except Exception as e:
            self.error('Previous INFO message cannot be logged..', str(e))

    def error(self, *args):
        """
        Create log of type ERROR
        :param args: the arguments that needed to be logged

        """
        try:
            text = '[ERROR]:' + datetime.datetime.now().strftime("%H:%M:%S---")
            for t in args:
                text += ' '
                text += str(t)
            print('\033[91m' + text + '\033[0m')
            if self.save:
                with open(self.p_log, "a") as f:
                    f.write(text + '\n')
        except Exception as e:
            self.error('Previous ERROR message cannot be logged..', str(e))

    def warning(self, *args):
        """
        Create log of type Warning
        :param args: the arguments that needed to be logged

        """
        try:
            text = '[WARNING]:' + datetime.datetime.now().strftime("%H:%M:%S---")
            for t in args:
                text += ' '
                text += str(t)
            print('\033[93m' + text + '\033[0m')
            if self.save:
                with open(self.p_log, "a") as f:
                    f.write(text + '\n')
        except Exception as e:
            self.error('Previous WARNING message cannot be logged..', str(e))