import torch
import torch.nn as nn

# 残差ffn
class fc_act_drop(nn.Module):
    
    def __init__(self, input_size):
        super(fc_act_drop, self).__init__()
        self.fc = nn.Linear(input_size, input_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return x + self.dropout(self.act(self.fc(x)))


class MyModel(nn.Module):

    def __init__(self, n_block, input_size):
        super(MyModel, self).__init__()
        self.block_list = nn.ModuleList()
        for _ in range(n_block):
            self.block_list.append(fc_act_drop(input_size))
        self.cls = nn.Linear(input_size, 2)

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return self.cls(x)

class MixLayer(nn.Module):

    def __init__(self, layer_list):
        super(MixLayer, self).__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.model_weight = nn.Parameter(torch.randn(len(layer_list)))
        # print(f'init weight is :{self.model_weight}')
        # print(f'init softmax weight is :{nn.functional.softmax(self.model_weight)}')

    def forward(self, x):
        layer_output_list = []
        for layer in self.layer_list:
            layer_output_list.append(layer(x))
        layer_output_shape = layer_output_list[0].shape
        layer_output = torch.cat(layer_output_list, 0).reshape(len(self.layer_list), *layer_output_shape)
        weight = nn.functional.softmax(self.model_weight)
        print(f'weight:{weight}')
        weight = weight.reshape([-1] + [1] * len(layer_output_shape))
        return (layer_output * weight).sum(0)


class MixModel_modify_graph(nn.Module):

    def __init__(self, model_list):
        super(MixModel_modify_graph, self).__init__()
        self.mix_model = MyModel(2, 16)
        self.model_list = model_list
        self.para_dict = {}
        
        self.weight_list = nn.Parameter(torch.ones(len(self.model_list)))
        self.mix_model = self.replace_parameter(self.mix_model, model_list)

    def replace_parameter(self, mix_layer, layer_list):
        if isinstance(mix_layer, nn.Linear):
            return MixLayer(layer_list)
            
        mix_layer_module = mix_layer._modules
        layer_module_list = [layer._modules for layer in layer_list]
        for name in mix_layer._modules.keys():
            mix_layer_sub_module = mix_layer_module[name]
            layer_sub_module_list = [layer_sub_module[name] for layer_sub_module in layer_module_list]
            setattr(mix_layer, name, self.replace_parameter(mix_layer_sub_module, layer_sub_module_list))
        return mix_layer
    
    def forward(self, x):
        x = self.mix_model(x)
        return x


# n_layers mid_size
model_1 = MyModel(2, 16)
model_2 = MyModel(2, 16)
model_3 = MyModel(2, 16)

mix_model = MixModel_modify_graph([model_1, model_2, model_3])
print(mix_model)

opti_para_list = []
for k, v in mix_model.named_modules():
    if hasattr(v, 'model_weight') and isinstance(v.model_weight, nn.Parameter):
        opti_para_list.append({'params': v.model_weight})
print(f'opti_para_list:{opti_para_list}')

optimizer = torch.optim.AdamW(opti_para_list, 0.1)
criterion = nn.CrossEntropyLoss()

for _ in range(10):
    inp = torch.randn(512, 16)
    label = ((inp > 0).sum(1) > (inp < 0).sum(1)).type(torch.LongTensor)
    out = mix_model(inp)
    loss = criterion(out, label)
    print(f'loss:{loss}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
