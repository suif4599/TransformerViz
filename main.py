from catch_err import Catch, tkinter_show_error

with Catch(tkinter_show_error):
    from gui import Root
    from modules import BertModule, LlamaModule, T5Module

    if __name__ == '__main__':
        win = Root()
        win.add_module(BertModule("english"))
        win.add_module(BertModule("chinese"))
        win.add_module(LlamaModule())
        win.add_module(T5Module())
        win.mainloop()
