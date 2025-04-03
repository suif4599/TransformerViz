from gui import Root
from modules import BertModule, LlamaModule

if __name__ == '__main__':
    win = Root()
    win.add_module(BertModule("english"))
    win.add_module(BertModule("chinese"))
    win.add_module(LlamaModule("7b"))
    win.mainloop()

