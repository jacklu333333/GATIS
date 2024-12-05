from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class Conv1dLSTMCell(nn.Module):
    def __init__(
        self,
        input_shape: Union[tuple, list],
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        bias: bool = False,
        cnn_dropout: bool = 0.0,
        rnn_dropout: bool = 0.0,
        layer_norm: bool = False,
    ):
        """
        Implementation of Convolutional1d LSTM cell
        Parameters
        ----------
        input_dim : int
                                        Number of channels in the input tensor
        hidden_dim : int
                                        Number of hidden units in the LSTM cell
        kernel_size : int
                                        Size of the convolutional kernel
        bias : bool
                                        Whether or not to add the bias
        cnn_dropout : float
                                        Dropout rate to be applied to the input sequence
        rnn_dropout : float
                                        Dropout rate to be applied to the output sequence
        layer_norm : bool
        """
        super().__init__()
        self.input_shape = list(input_shape)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.stride = 1

        # compute the padding size based on the kernel size and stride len so that the output size is the same as the input size
        self.padding = (self.kernel_size - self.stride) // 2
        self.cnn_dropout = cnn_dropout
        self.rnn_dropout = rnn_dropout
        self.layer_norm = layer_norm

        # print the padding size in cyan
        self.input_conv = nn.Conv1d(
            in_channels=self.input_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            padding_mode="circular",
        )
        nn.init.kaiming_normal_(self.input_conv.weight)
        if self.input_conv.bias is not None:
            nn.init.uniform_(self.input_conv.bias, -1e-5, 1e-5)

        self.rnn_conv = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
            padding_mode="replicate",
        )
        nn.init.kaiming_normal_(self.rnn_conv.weight)
        if self.rnn_conv.bias is not None:
            nn.init.uniform_(self.rnn_conv.bias, -1e-5, 1e-5)

        self.cnn_dropout = nn.Dropout(self.cnn_dropout)
        self.rnn_dropout = nn.Dropout(self.rnn_dropout)

        if self.layer_norm:
            self.layer_norm_x = nn.LayerNorm([4 * self.hidden_dim] + self.input_shape)
            self.layer_norm_h = nn.LayerNorm([4 * self.hidden_dim] + self.input_shape)
            self.layer_norm_c = nn.LayerNorm([self.hidden_dim] + self.input_shape)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        x = self.cnn_dropout(input_tensor)
        x_conv = self.input_conv(x)
        if self.layer_norm:
            x_conv = self.layer_norm_x(x_conv)
        x_i, x_f, x_c, x_o = torch.split(x_conv, self.hidden_dim, dim=1)

        h = self.rnn_dropout(h_cur)
        h_conv = self.rnn_conv(h)
        if self.layer_norm:
            h_conv = self.layer_norm_h(h_conv)
        h_i, h_f, h_c, h_o = torch.split(h_conv, self.hidden_dim, dim=1)

        f = torch.sigmoid((x_f + h_f))
        i = torch.sigmoid((x_i + h_i))

        g = torch.tanh((x_c + h_c))
        c_next = f * c_cur + i * g

        o = torch.sigmoid((x_o + h_o))

        if self.layer_norm:
            c_next = self.layer_norm_c(c_next)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        h = torch.empty(
            [batch_size] + [self.hidden_dim] + self.input_shape,
            device=self.input_conv.weight.device,
        )
        c = torch.empty(
            [batch_size] + [self.hidden_dim] + self.input_shape,
            device=self.input_conv.weight.device,
        )

        nn.init.constant_(h, 0)
        nn.init.constant_(c, 0)

        return h, c


class Conv1dLSTM(nn.Module):
    """
    Implementation of Convolutional LSTM
    """

    def __init__(
        self,
        input_shape: Union[tuple, list],
        input_dim: int,
        hidden_dim: list,
        kernel_size: list,
        num_layers: int,
        cnn_dropout: float = 0.0,
        rnn_dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = False,
        batch_first: bool = True,
        bidirectional: bool = False,
        return_all_layers: bool = False,
    ):
        super().__init__()
        assert num_layers == len(hidden_dim) == len(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        # store the parameters
        self.input_shape = list(input_shape)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.cnn_dropout = cnn_dropout
        self.rnn_dropout = rnn_dropout
        self.batch_first = batch_first
        self.layer_norm = layer_norm
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.bidirectional = bidirectional

        cell_list_fw = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list_fw.append(
                Conv1dLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                    cnn_dropout=self.cnn_dropout,
                    rnn_dropout=self.rnn_dropout,
                    layer_norm=self.layer_norm,
                    input_shape=self.input_shape,
                )
            )

        self.cell_list_fw = nn.ModuleList(cell_list_fw)

        if self.bidirectional:
            cell_list_bw = []
            for i in range(self.num_layers):
                cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

                cell_list_bw.append(
                    Conv1dLSTMCell(
                        input_dim=cur_input_dim,
                        hidden_dim=self.hidden_dim[i],
                        kernel_size=self.kernel_size[i],
                        bias=self.bias,
                        cnn_dropout=self.cnn_dropout,
                        rnn_dropout=self.rnn_dropout,
                        layer_norm=self.layer_norm,
                        input_shape=self.input_shape,
                    )
                )

            self.cell_list_bw = nn.ModuleList(cell_list_bw)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h) -> (b, t, c, h)
            input_tensor = input_tensor.permute(1, 0, 2, 3)

        b, seq_len, _, h = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state, hidden_state_inv = self._init_hidden(
                batch_size=b,
            )

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor
        cur_layer_input_inv = input_tensor.flip(1)

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list_fw[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            if self.bidirectional:
                h_inv, c_inv = hidden_state_inv[layer_idx]
                output_inner_inv = []
                for t in range(seq_len):
                    h_inv, c_inv = self.cell_list_bw[layer_idx](
                        input_tensor=cur_layer_input_inv[:, t, :, :],
                        cur_state=[h_inv, c_inv],
                    )
                    output_inner_inv.append(h_inv)

                output_inner_inv.reverse()
                layer_output_inv = torch.stack(output_inner_inv, dim=1)
                cur_layer_input_inv = layer_output_inv

            if self.bidirectional:
                layer_output_list.append((layer_output, layer_output_inv))
                last_state_list.append(([h, c], [h_inv, c_inv]))
            else:
                layer_output_list.append(layer_output)
                last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size: int):
        init_states_fw = []
        for i in range(self.num_layers):
            init_states_fw.append(self.cell_list_fw[i].init_hidden(batch_size))

        init_states_bw = []
        if self.bidirectional:
            for i in range(self.num_layers):
                init_states_bw.append(self.cell_list_bw[i].init_hidden(batch_size))

        return init_states_fw, init_states_bw

    @staticmethod
    def _extend_for_multilayer(param, num_layers: int):
        if isinstance(param, int):
            return (param,) * num_layers
        elif isinstance(param, list):
            return tuple(param)
        else:
            return param
