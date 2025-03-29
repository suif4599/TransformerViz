from gui import Root
from core import AbstractModule, BertModule

class TestModule(AbstractModule):
    def __init__(self, name="Test Module", description="This is a test module."):
        super().__init__()
        self.name = name
        self.description = description
    
    def get_name(self):
        return self.name
    
    def get_description(self):
        return self.description
    
    def forward(self, sentence):
        return
    
    def get_input(self):
        return
    
    def get_output(self):
        return
    
    def get_attention_weights(self, key, query, mode):
        return
    
    def get_layer_mix_mode_list(self):
        return ["first", "final", "average"]
    
    def get_head_mix_mode_list(self):
        return ["first", "all"]
    
    def load(self):
        return
    
    def unload(self):
        return 
    

if __name__ == '__main__':
    win = Root()
    win.add_module(TestModule())
    win.add_module(TestModule("Test Module 2", "This is another test module."))
    win.add_module(BertModule("english"))
    win.mainloop()

# model = BertModule()
# model.load()
# model.forward("I am a _.")
# model.get_attention_weights(0, "i", "am", "first")