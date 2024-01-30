import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.device))


class ConvLSTM(nn.Module):
    """
    ConvLSTM module that stacks multiple ConvLSTMCells.
    """
    def __init__(self, input_dim, hidden_dims, kernel_sizes, num_layers, batch_first=False, return_sequences=True, bias=True):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_sizes = kernel_sizes
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_sizes[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, x, hidden_state=None):
        """
        Forward pass of ConvLSTM.
        """
        b, _, _, h, w = x.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = x.size(1)
        cur_layer_input = x

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_sequences:
            layer_output_list = [layer_output[:, -1, :, :, :] for layer_output in layer_output_list]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class ConvLSTMModel(nn.Module):
    def __init__(self, num_classes,Im_height=64,Im_width=64):
        super(ConvLSTMModel, self).__init__()
        
        self.convlstm = ConvLSTM(input_dim=3, hidden_dims=[4, 8, 14, 16], kernel_sizes=[(3, 3), (3, 3), (3, 3), (3, 3)], num_layers=4, batch_first=True, return_sequences=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * (Im_height // 8) * (Im_width // 8), num_classes)

    def forward(self, x):
        convlstm_out, _ = self.convlstm(x)
        out = self.maxpool(convlstm_out[-1])
        out = self.flatten(out)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    ...