import torch
import torch.nn.functional as F

vocab_size = 8
input_ids = [[1, 2], [1, 3]]


def decode(input_ids, vocab_size):
    batch_size = len(input_ids)
    return F.softmax(torch.randn((batch_size, vocab_size)), dim=-1)


# sample
def sample(scores):
    return torch.multinomial(scores, 1)


scores = decode(input_ids, vocab_size)
# print(scores)
# print(sample(scores))


# beam search
beam_size = 5
batch_scores = decode(input_ids, vocab_size)
beam_hyps_dict = dict()
eos_idx = 0
max_len = 10
for i in range(len(input_ids)):
    beam_hyps_dict[i] = list()
    scores = batch_scores[i]
    topk_scores, topk_indices = torch.topk(scores, k=beam_size)
    for score, token_idx in zip(topk_scores, topk_indices):
        beam_hyps_dict[i].append([score.item(), [token_idx.item()]])


def beam_step(beam_hyps, beam_size, vocab_size):
    input_ids = list(map(lambda x: x[1], beam_hyps))
    batch_scores = decode(input_ids, vocab_size)
    new_beam_hyps = []
    for i in range(len(input_ids)):
        beam_hyp = beam_hyps[i]
        if beam_hyp[1][-1] == eos_idx:
            new_beam_hyps.append(beam_hyp)
        else:
            scores = batch_scores[i]
            topk_scores, topk_indices = torch.topk(scores, k=beam_size)
            for score, token_idx in zip(topk_scores, topk_indices):
                new_beam_hyps.append([score.item() * beam_hyp[0], beam_hyp[1] + [token_idx.item()]])
    new_beam_hyps.sort(reverse=True)
    return new_beam_hyps[:beam_size]


def beam_search(beam_hyps, beam_size, vocab_size):
    while True:
        stop = True
        for beam_hyp in beam_hyps:
            if len(beam_hyp[1]) < max_len and beam_hyp[1][-1] != eos_idx:
                stop = False
                break
        if not stop:
            beam_hyps = beam_step(beam_hyps, beam_size, vocab_size)
        else:
            break
    return beam_hyps


for i in range(len(input_ids)):
    beam_hyps_dict[i] = beam_search(beam_hyps_dict[i], beam_size, vocab_size)
    print(beam_hyps_dict[i])
