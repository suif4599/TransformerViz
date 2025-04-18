from gui import Root
from modules import BertModule, OpusModule

if __name__ == '__main__':
    win = Root()
    win.add_module(BertModule("english"))
    win.add_module(BertModule("chinese"))
    win.mainloop()

# model = OpusModule()
# model.load()
# print(model.model)
# model.forward("你好，请问你叫什么名字？")
