from gui import Root
from modules import BertModule

if __name__ == '__main__':
    win = Root()
    win.add_module(BertModule("english"))
    win.add_module(BertModule("chinese"))
    win.mainloop()

