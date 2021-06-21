import os
import sys


class Config:

    def __init__(self, args=None, file=None):
        if args is not None:
            for k, v in args.__dict__.items():
                self.__setattr__(k, v)
            if self.__dict__["load_dir"] is not None:
                self.__setattr__("load_dir", self.get_ckpt(self.__dict__["load_dir"]))

        self.data_dir = self.get_data_dir()

    def get_ckpt(self, dir):
        name = os.listdir(dir)[0]
        return os.path.join(dir, name)

    def get_data_dir(self):
        return os.path.join(os.path.dirname(os.getcwd()), 'data')

    def show(self):
        for name, value in vars(self).items():
            print(f"{name}={value}")

    def add_display(self, name):
        if hasattr(self, name):
            return f'_{name}{getattr(self, name)}'
        else:
            return ''

    def get_generate_out_file_name(self):
        res = self.py_name
        names = ['version_num', 'sent']
        for name in names:
            res += self.add_display(name)
        return res

    def get_root_dir(self):
        res = './logs/' + self.py_name
        names = ['sent']
        for name in names:
            res += self.add_display(name)
        return res


if __name__ == '__main__':
    print(__file__)
    config = Config(file=__file__)
    print(config.data_dir)