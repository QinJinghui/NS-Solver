import torch
import torch.nn as nn

class BaseRNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers,
                 embedding_dropout=0.5, rnn_dropout=0.5, rnn_cell_name="gru"):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        # embedding
        self.embedding_size = embedding_size
        self.embedding_dropout_rate = embedding_dropout
        self.embedding_dropout = nn.Dropout(self.embedding_dropout_rate)

        # rnn
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn_dropout_rate = rnn_dropout
        # self.rnn_dropout = nn.Dropout(self.rnn_dropout_rate)
        if rnn_cell_name.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell_name.lower() == 'gru':
            self.rnn_cell = nn.GRU
        elif rnn_cell_name.lower() == "rnn":
            self.rnn_cell = nn.RNN
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell_name))

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

class Attn(nn.Module):
    def __init__(self, hidden_size, batch_first=False, bidirectional_encoder=True):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional_encoder = bidirectional_encoder
        # if self.bidirectional_encoder:
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        # else:
        #     self.attn = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        if self.batch_first:  # B x S x H
            max_len = encoder_outputs.size(1)
            repeat_dims = [1] * hidden.dim()
            repeat_dims[1] = max_len
        else:  # S x B x H
            max_len = encoder_outputs.size(0)
            repeat_dims = [1] * hidden.dim()
            repeat_dims[0] = max_len
        # batch_first: False S x B x H
        # batch_first: True B x S x H
        hidden = hidden.repeat(*repeat_dims)  # Repeats this tensor along the specified dimensions

        # For each position of encoder outputs
        if self.batch_first:
            batch_size = encoder_outputs.size(0)
        else:
            batch_size = encoder_outputs.size(1)
        # (B x S) x (2 x H) or (S x B) x (2 x H)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1 or (B x S) x 1
        # print(attn_energies.size())
        attn_energies = attn_energies.squeeze(1)  # (S x B) or (B x S)
        if self.batch_first:
            attn_energies = attn_energies.view(batch_size, max_len)  # B x S
        else:
            attn_energies = attn_energies.view(max_len, batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        attn_energies = self.softmax(attn_energies)
        return attn_energies.unsqueeze(1)

class GeneralEncoderRNN(BaseRNN):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers,
                 embedding_dropout=0.5, embedding=None, update_embedding=True,
                 rnn_dropout=0.5, bidirectional=True, variable_lengths=False,
                 bias=True, batch_first=False, rnn_cell_name="gru", max_seq_len=512):
        super(GeneralEncoderRNN, self).__init__(vocab_size=vocab_size, embedding_size=embedding_size,
                                                hidden_size=hidden_size, n_layers=n_layers,
                                                embedding_dropout=embedding_dropout,
                                                rnn_dropout=rnn_dropout,
                                                rnn_cell_name=rnn_cell_name)
        self.max_seq_len = max_seq_len
        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        # embedding
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        elif isinstance(embedding, nn.Embedding):
            if embedding.num_embeddings == self.vocab_size and embedding.embedding_dim == self.embedding_size:
                self.embedding = embedding
            else:
                self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            try:
                self.embedding.weight = nn.Parameter(embedding)
                self.embedding.weight.requires_grad = update_embedding
            except:
                print("Embedding Init Exception: we use random init instead")

        # rnn
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers, dropout=self.rnn_dropout_rate,
                                 bidirectional=self.bidirectional, batch_first=self.batch_first, bias=self.bias)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        embedded = self.embedding_dropout(embedded)
        if self.variable_lengths:
            embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=self.batch_first)
        outputs, hidden = self.rnn(embedded, hidden)
        if self.variable_lengths:
            outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)  # unpack (back to padded)
        # fusion strategy
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # batch_first=False: S x B x H
        # batch_first=True: B x S x H
        return outputs, hidden


class GeneralDecoderRNN(BaseRNN):
    def __init__(self, vocab_size, classes_size, embedding_size, hidden_size, n_layers,
                 embedding_dropout=0.5, embedding=None, update_embedding=True,
                 rnn_dropout=0.5, bidirectional=False, variable_lengths=False,
                 bias=True, batch_first=False, rnn_cell_name="gru", max_seq_len=512):
        super(GeneralDecoderRNN, self).__init__(vocab_size=vocab_size, embedding_size=embedding_size,
                                                hidden_size=hidden_size, n_layers=n_layers,
                                                embedding_dropout=embedding_dropout,
                                                rnn_dropout=rnn_dropout,
                                                rnn_cell_name=rnn_cell_name)

        self.bidirectional_encoder = bidirectional
        self.max_seq_len = max_seq_len
        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.classes_size = classes_size
        # embedding
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        elif isinstance(embedding, nn.Embedding):
            if embedding.num_embeddings == self.vocab_size and embedding.embedding_dim == self.embedding_size:
                self.embedding = embedding
            else:
                self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            try:
                self.embedding.weight = nn.Parameter(embedding)
                self.embedding.weight.requires_grad = update_embedding
            except:
                print("Embedding Init Exception: we use random init instead")

        # rnn
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers, dropout=self.rnn_dropout_rate,
                                 bidirectional=False, batch_first=self.batch_first, bias=self.bias)

        self.out = nn.Linear(self.hidden_size, self.classes_size)

        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, hidden, encoder_outputs=None, seq_mask=None):
        # Get the embedding of the current input word (last output word)
        # batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)

        # Get current hidden state from input word and last hidden state
        last_hidden = hidden
        output, hidden = self.rnn(embedded, last_hidden)

        if self.batch_first:
            # output = self.softmax(self.out(torch.tanh(output.squeeze(1))))
            output = self.out(torch.tanh(output.squeeze(1)))
        else:
            # output = self.softmax(self.out(torch.tanh(output.squeeze(0))))
            output = self.out(torch.tanh(output.squeeze(0)))

        # Return final output, hidden state
        return output, hidden


# General RNN Decoder, we can customize its function with more options
class GeneralAttnDecoderRNN(BaseRNN):
    def __init__(self, vocab_size, classes_size, embedding_size, hidden_size, n_layers,
                 embedding_dropout=0.5, embedding=None, update_embedding=True,
                 rnn_dropout=0.5, bidirectional=False, variable_lengths=False,
                 bias=True, batch_first=False, rnn_cell_name="gru", max_seq_len=512):
        super(GeneralAttnDecoderRNN, self).__init__(vocab_size=vocab_size, embedding_size=embedding_size,
                                                    hidden_size=hidden_size, n_layers=n_layers,
                                                    embedding_dropout=embedding_dropout,
                                                    rnn_dropout=rnn_dropout,
                                                    rnn_cell_name=rnn_cell_name)
        self.bidirectional_encoder = bidirectional
        self.max_seq_len = max_seq_len
        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias
        self.classes_size = classes_size
        # embedding
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        elif isinstance(embedding, nn.Embedding):
            if embedding.num_embeddings == self.vocab_size and embedding.embedding_dim == self.embedding_size:
                self.embedding = embedding
            else:
                self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            try:
                self.embedding.weight = nn.Parameter(embedding)
                self.embedding.weight.requires_grad = update_embedding
            except:
                print("Embedding Init Exception: we use random init instead")

        # rnn
        self.rnn = self.rnn_cell(hidden_size + embedding_size, hidden_size, n_layers, dropout=self.rnn_dropout_rate,
                                 bidirectional=False, batch_first=self.batch_first, bias=self.bias)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(self.hidden_size, self.classes_size)
        # Choose attention model
        self.attn = Attn(hidden_size, batch_first=batch_first, bidirectional_encoder=bidirectional)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_seq, hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)

        last_hidden = hidden
        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        # Get current hidden state from input word and last hidden state
        # print(last_hidden[-1].unsqueeze(0).size())
        # print(encoder_outputs[0].size())
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        if self.batch_first:
            context = attn_weights.bmm(encoder_outputs) # B x 1 x S * B x S x H = B x 1 x H
            output, hidden = self.rnn(torch.cat((embedded, context), 2), last_hidden)
        else:
            context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x S * B x S x H = B x 1 x H  # B x S =1 x N
            output, hidden = self.rnn(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        if self.batch_first:
            output = self.softmax(self.out(torch.tanh(output.squeeze(1)))) # B x C
            output = self.out(torch.tanh(output.squeeze(1))) # B x C
        else:
            # output = self.softmax(self.out(torch.tanh(output.squeeze(0)))) # B x C
            output = self.out(torch.tanh(output.squeeze(0))) # B x C

        # Return final output, hidden state
        return output, hidden



class Seq2TreeEncoder(BaseRNN):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers=2, embedding_dropout=0.5, rnn_dropout=0.5):
        super(Seq2TreeEncoder, self).__init__(vocab_size, embedding_size, hidden_size, n_layers,
                                              embedding_dropout=embedding_dropout, rnn_dropout=rnn_dropout)
        # embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        # GRU
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers, dropout=self.rnn_dropout_rate,
                                 batch_first=False, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.embedding_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        pade_outputs, pade_hidden = self.rnn(packed, pade_hidden)
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)

        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:] # B x H
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        return pade_outputs, problem_output


######################################################################################
#####                              Seq2Tree                                      #####
######################################################################################
class TreeNode: # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag




# 对应Seq2Tree论文公式6的a中的score计算, 用于生成context向量
class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(TreeAttn, self).__init__()
        self.input_size = input_size # goal vector size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        if self.batch_first:
            # encoder_outputs: S x B x H
            max_len = encoder_outputs.size(1)
            # hidden:  B x 1 x H
            repeat_dims = [1] * hidden.dim()
            repeat_dims[1] = max_len
            batch_size = encoder_outputs.size(0)
        else:
            # encoder_outputs: S x B x H
            max_len = encoder_outputs.size(0)
            # hidden: 1 x B x H
            repeat_dims = [1] * hidden.dim()
            repeat_dims[0] = max_len
            batch_size = encoder_outputs.size(1)
        hidden = hidden.repeat(*repeat_dims)  # S x B x H or B x S x H

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size) # SBx2H or BSx2H

        score_feature = torch.tanh(self.attn(energy_in))  # SBxH or BSxH
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        if self.batch_first:
            attn_energies = attn_energies.view(batch_size, max_len) # B x S
        else:
            attn_energies = attn_energies.view(max_len, batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1) # B x 1 x S



# 用于选择数字
class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size  # goal vector size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        # 这里的hidden: B x 1 x H; num_embeddings: B x O x H
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -1e12)
        return score


class Seq2TreePrediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding
    def __init__(self, hidden_size, op_nums, vocab_size, dropout=0.5):
        super(Seq2TreePrediction, self).__init__()
        # Keep for reference
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.op_nums = op_nums  # 数字列表长度

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, vocab_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)  # left inner symbols generation
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)  # right inner symbols generation
        self.concat_lg = nn.Linear(hidden_size, hidden_size)   # left number generation
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)  # right number generation

        # 用于操作符选择
        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_children, encoder_outputs, padded_nums, padded_hidden, seq_mask, num_mask):
        current_embeddings = []
        # node_stacks: B
        # padded_hidden: B x 2H
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padded_hidden)
                # current_embeddings.append(padded_hidden[node_stacks.index(st)].unsqueeze(0))
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_children, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)
        current_embeddings = self.dropout(current_node)
        # print(current_embeddings.size())
        # print(encoder_outputs.size())
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, padded_nums), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, num_mask)

        op = self.ops(leaf_input)

        return num_score, op, current_node, current_context, embedding_weight


class Seq2TreeSubTreeMerge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Seq2TreeSubTreeMerge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class Seq2TreeNodeGeneration(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(Seq2TreeNodeGeneration, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class SelfSupervisionNumberPrediction(nn.Module):
    def __init__(self, classes_size, hidden_size, dropout=0.5):
        super(SelfSupervisionNumberPrediction, self).__init__()
        self.classes_size = classes_size
        self.hidden_size = hidden_size
        self.rnn_act = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, classes_size)

    def forward(self, problem_outputs):
        outputs = torch.mean(problem_outputs, dim=1)
        outputs = self.rnn_act(outputs)
        outputs = self.dropout(outputs)
        prediction_logits = self.linear(outputs)
        return prediction_logits


class SelfSupervisionNumberLocationPrediction(nn.Module):
    def __init__(self, classes_size, hidden_size, dropout=0.5):
        super(SelfSupervisionNumberLocationPrediction, self).__init__()
        self.classes_size = classes_size
        self.hidden_size = hidden_size
        self.rnn_act = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, classes_size)

    def forward(self, problem_outputs):
        outputs = torch.mean(problem_outputs, dim=1)
        outputs = self.rnn_act(outputs)
        outputs = self.dropout(outputs)
        prediction_logits = self.linear(outputs)
        return prediction_logits


class ConstantReasoning(nn.Module):
    def __init__(self, classes_size, hidden_size, dropout=0.5):
        super(ConstantReasoning, self).__init__()
        self.classes_size = classes_size
        self.hidden_size = hidden_size
        self.rnn_act = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, classes_size)

    def forward(self, problem_outputs):
        outputs = torch.mean(problem_outputs, dim=1)
        outputs = self.rnn_act(outputs)
        outputs = self.dropout(outputs)
        prediction_logits = self.linear(outputs)
        return prediction_logits


#############################################################################################
##### TreeEncoder
#############################################################################################
class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding  # the size = hidden size
        self.terminal = terminal


class TreeNode: # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class TreeNodeGeneration(nn.Module):
    def __init__(self, hidden_size, embedding, dropout=0.5):
        super(TreeNodeGeneration, self).__init__()
        self.embedding = embedding
        self.embedding_size = self.embedding.embedding_dim
        self.hidden_size = hidden_size
        # self.ops_idx_list = ops_idx_list
        self.dropout = nn.Dropout(dropout)

        self.input_encoding = nn.Linear(self.embedding_size, hidden_size)
        # for node embedding
        self.ne_lg1 = nn.Linear(hidden_size*2, hidden_size)
        self.ne_lg2 = nn.Linear(hidden_size*2, hidden_size)

        self.ne_rg1 = nn.Linear(hidden_size * 3, hidden_size)
        self.ne_rg2 = nn.Linear(hidden_size * 3, hidden_size)

        # for child hidden
        self.left_child = nn.Linear(hidden_size, hidden_size)
        self.left_child_g = nn.Linear(hidden_size, hidden_size)
        self.right_child = nn.Linear(hidden_size, hidden_size)
        self.right_child_g = nn.Linear(hidden_size, hidden_size)

    def forward(self, node_idx, parent_hidden, left_sibling_tree_embedding=None):
        # generate node_embedding and child hidden
        node_input_emb_ = self.embedding(node_idx)
        node_input_emb = self.dropout(node_input_emb_)
        node_input_encoding = self.input_encoding(node_input_emb)
        # parent_hidden = torch.unsqueeze(parent_hidden,0)
        # parent_hidden = self.dropout(parent_hidden)
        if isinstance(left_sibling_tree_embedding,torch.Tensor):
            left_sibling_tree_embedding = left_sibling_tree_embedding
            left_sibling_tree_embedding = self.dropout(left_sibling_tree_embedding)

        # generate node embedding
        if isinstance(left_sibling_tree_embedding,torch.Tensor):
            # print(node_input_encoding.size())
            # print(parent_hidden.size())
            # print(left_sibling_tree_embedding.size())
            ne_rg1 = torch.tanh(self.ne_rg1(torch.cat((node_input_encoding, parent_hidden,left_sibling_tree_embedding), 1)))
            ne_rg2 = torch.sigmoid(self.ne_rg2(torch.cat((node_input_encoding, parent_hidden,left_sibling_tree_embedding), 1)))
            node_embedding = ne_rg1 * ne_rg2
        else:
            # print(node_input_encoding)
            # print(parent_hidden)
            ne_lg1 = torch.tanh(self.ne_lg1(torch.cat((node_input_encoding, parent_hidden), 1)))
            ne_lg2 = torch.sigmoid(self.ne_lg2(torch.cat((node_input_encoding, parent_hidden), 1)))
            node_embedding = ne_lg1 * ne_lg2

        # generate child
        node_embedding_ = self.dropout(node_embedding)
        l_child = torch.tanh(self.left_child(node_embedding_))
        l_child_g = torch.sigmoid(self.left_child_g(node_embedding_))
        r_child = torch.tanh(self.right_child(node_embedding_))
        r_child_g = torch.sigmoid(self.right_child_g(node_embedding_))
        left_child_hidden = l_child * l_child_g
        right_child_hidden = r_child * r_child_g

        # return node_input_emb_, node_embedding, left_child_hidden, right_child_hidden
        return node_input_emb_, node_input_encoding, node_embedding, left_child_hidden, right_child_hidden


class SubTreeEncoding(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout=0.5):
        super(SubTreeEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_l, sub_tree_r):
        sub_tree_l = self.dropout(sub_tree_l)
        sub_tree_r = self.dropout(sub_tree_r)
        node_embedding = self.dropout(node_embedding)
        # print(node_embedding.size())
        # print(sub_tree_l.size())
        # print(sub_tree_r.size())
        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_l, sub_tree_r), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_l, sub_tree_r), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree


class ExplicitTreeEncoder(nn.Module):
    def __init__(self, output_lang, embedding_size, hidden_size, dropout=0.5):
        super(ExplicitTreeEncoder, self).__init__()

        vocab_size = output_lang.n_words
        self.output_lang = output_lang
        self.ops_idx_list = output_lang.get_ops_idx()
        self.pad_idx = output_lang.get_pad_token()

        self.embedding_size = embedding_size
        self.embedding_dropout_rate = dropout
        self.embedding_dropout = nn.Dropout(self.embedding_dropout_rate)
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=output_lang.get_pad_token())

        self.dropout = dropout
        self.hidden_size = hidden_size

        self.node_generation = TreeNodeGeneration(hidden_size=self.hidden_size,
                                                  embedding=self.embedding, dropout=self.dropout)
        self.subtree_enc = SubTreeEncoding(embedding_size=self.embedding_size, hidden_size=self.hidden_size, dropout=self.dropout)

    def forward(self, input_seqs, input_lengths, hiddens=None):
        # input_seqs: # S x B
        # input_seqs = input_seqs.transpose(0,1)  # B x S
        batch_size, seq_len = input_seqs.size()
        # zero_state = torch.zeros(1, self.hidden_size, dtype=input_seqs.dtype, device=input_seqs.device)
        if hiddens == None:
            hiddens = torch.zeros(batch_size, self.hidden_size,
                                  dtype=torch.float32, device=input_seqs.device)

        encoder_outputs = []
        encoder_hiddens = []

        for idx_bs in range(batch_size):
            input_seq = input_seqs[idx_bs]
            hidden = torch.unsqueeze(hiddens[idx_bs],0)

            all_node_outputs = [] # node embedding for each node for encoder outputs
            final_tree_embedding = None # for encoder final hidden
            node_stack = [TreeNode(hidden),]  # stack for prefix traverse
            left_childs = [None,]
            subtree_stack = []
            real_seq_length = input_lengths[idx_bs]
            # print(real_seq_length)
            for t in range(real_seq_length):
                # print(t)
                # print(input_seq[t])
                node_idx = input_seq[t].unsqueeze(0)
                # print(self.output_lang.index2word[input_seq[t]])
                node_input_emb_, node_input_encoding, node_embedding, left_child_hidden, right_child_hidden = self.node_generation(node_idx, node_stack[-1].embedding, left_sibling_tree_embedding=left_childs[-1])
                if len(node_stack) != 0:
                    node_stack.pop()
                else:
                    left_childs.append(None)
                    continue
                all_node_outputs.append(node_embedding)
                if node_idx in self.ops_idx_list: # 非数字
                    node_stack.append(TreeNode(right_child_hidden))
                    node_stack.append(TreeNode(left_child_hidden, left_flag=True))
                    # print('NN',node_input_emb_.size())
                    subtree_stack.append(TreeEmbedding(node_input_emb_, terminal=False))
                else: # 数字
                    current_num = node_input_encoding
                    while len(subtree_stack) > 0 and subtree_stack[-1].terminal:
                        sub_tree = subtree_stack.pop()
                        op = subtree_stack.pop()
                        # print('N', op.embedding.size())
                        # print('N',sub_tree.embedding.size())
                        # print('N',current_num.size())
                        current_num = self.subtree_enc(op.embedding, sub_tree.embedding, current_num)
                        # print('N',current_num.size())
                    subtree_stack.append(TreeEmbedding(current_num,terminal=True))
                    final_tree_embedding = current_num
                if len(subtree_stack) > 0 and subtree_stack[-1].terminal:
                    left_childs.append(subtree_stack[-1].embedding)
                else:
                    left_childs.append(None)

            for t in range(real_seq_length, seq_len):
                node_idx = input_seq[t].unsqueeze(0)
                _, _, node_embedding, _, _ = self.node_generation(node_idx, final_tree_embedding)
                # all_node_outputs.append(self.embedding(node_idx))
                all_node_outputs.append(node_embedding)

            encoder_outputs.append(torch.cat(all_node_outputs,dim=0))
            encoder_hiddens.append(final_tree_embedding)

        encoder_hiddens = torch.cat(encoder_hiddens, dim=0)
        encoder_outputs = torch.stack(encoder_outputs, dim=0)

        # S x B x H, B x H
        # print(encoder_outputs.size())
        # print(encoder_hiddens.size())
        return encoder_outputs.transpose(0,1), encoder_hiddens