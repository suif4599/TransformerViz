from gui import Root
from modules import BertModule, LlamaModule, T5Module

# if __name__ == '__main__':
#     win = Root()
#     win.add_module(BertModule("english"))
#     win.add_module(BertModule("chinese"))
#     win.add_module(LlamaModule("7b"))
#     win.mainloop()

module = T5Module()
module.load()
print(module.model.decoder.block[0].layer[1].EncDecAttention)
module.forward("how are you today?")
print(module.get_sentence("encoder-decoder"))
