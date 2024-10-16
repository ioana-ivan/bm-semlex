import codecs
import os

import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          AutoModelForMaskedLM)
from evaluate import load
from transformers import LlamaTokenizer, LlamaForCausalLM
from hf_olmo import OLMoTokenizerFast # NOQA
from datetime import datetime
import time
from argparse import ArgumentParser
from huggingface_hub import login
import gc


# https://stackoverflow.com/questions/70464428/how-to-calculate-perplexity-of-a-sentence-using-huggingface-masked-language-mode
def score_masked(sentence_list, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    list_scores = []
    print('Scoring sentences ...')
    for sentence in sentence_list:
        tensor_input = tokenizer.encode(sentence, return_tensors='pt')
        repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1,
                                                tokenizer.mask_token_id)
        labels = repeat_input.masked_fill(
            masked_input != tokenizer.mask_token_id,
            -100)

        with torch.inference_mode():
            loss = model(masked_input, labels=labels).loss

        pp = np.exp(loss.item())
        list_scores.append(pp)
    return list_scores


# https://datascience.stackexchange.com/questions/38540/are-there-any-good-out-of-the-box-language-models-for-python
# same result as huggingface
def score(sentence, model, tokenizer, device):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer
                                 .convert_tokens_to_ids(
                                     tokenize_input
                                     )]).to(device)

    with torch.no_grad():
        outputs = model(tensor_input, labels=tensor_input)
    entropy = outputs[0].item()
    perplexity = np.exp(entropy)
    return perplexity


def get_device():
    print(f'Is cuda available: {torch.cuda.is_available()}')
    print(f'Device count: {torch.cuda.device_count()}')

    device = 'cpu'
    device_map = 'auto'
    if torch.cuda.device_count() != 0:
        device = 'cuda'
        device_map = 'cuda'
        print(f'Device current: {torch.cuda.current_device()}')

        # additional Info when using cuda
        giga = 1024**3
        if device == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:',
                  round(torch.cuda.memory_allocated(0)/giga, 1), 'GB')
            print('Cached:   ',
                  round(torch.cuda.memory_reserved(0)/giga, 1), 'GB')

    print('Using device:', device)
    print('Using device map:', device_map)
    return device, device_map


def score_causal_old(sentence_list, model_name):
    """
    Parameters
    ----------
    sentence : string

    model : model

    tokenizer : tokenizer
    """

    device, device_map = get_device()

    if 'OLMo' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            device=device)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            offload_folder="offload",
            device_map=device_map)
    elif 'Amber' in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_name,
            device=device)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            device_map=device_map)
    elif 'Mistral' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device=device)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map)
    elif 'Llama' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device=device)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map)
    else:
        print(f"Model {model_name} not found")
        raise Exception("Model not found")

    print(f'Model {model_name} loaded')
    perplexities = []
    i = 0
    for sentence in sentence_list:
        if i % 10 == 0:
            print(f'Scoring sentence {i}')
        pp = score(sentence, model, tokenizer, device)
        perplexities.append(pp)
        i += 1
    return perplexities


# https://huggingface.co/spaces/evaluate-metric/perplexity
def score_causal(sentence_list, model_name):
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(model_id=model_name,
                                 add_start_token=False,
                                 predictions=sentence_list)
    print(results['perplexities'])
    return results['perplexities']


def get_sentences_to_eval(prompt_file):
    list_sent = []
    with codecs.open(prompt_file, encoding='utf-8', mode='r') as pfile:
        for line in pfile:
            elems = line.split('\t')
            context = elems[3]
            list_sent.append(context)

    return list_sent


def create_score_filename(prompt_file, model_name_file, output_dir):
    prompt_filename = os.path.split(prompt_file)[1]
    prompt_scored_filename = prompt_filename + "." + model_name_file
    prompt_scored_filename = prompt_scored_filename.replace('/', '.')
    score_file = output_dir + prompt_scored_filename
    return score_file


def write_scores_to_file(prompt_file, pp_list, score_file):
    i = 0
    with codecs.open(prompt_file, encoding='utf-8', mode='r') as pfile:
        with codecs.open(score_file, encoding='utf-8', mode='w') as sfile:
            for line in pfile:
                newline = line.strip('\n') + '\t' + str(pp_list[i]) + '\n'
                i += 1
                sfile.write(newline)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename",
                        help="input file name (path hardcoded for now)",
                        metavar="FILE", type=str)
    parser.add_argument("-m", "--model", dest="model",
                        help="the huggingface model name or local checkpoint path if checkpoint saved on disk",
                        type=str, metavar="MODEL")
    parser.add_argument("-p", "--masked", dest="masked",
                        help="if masked, compute pseudo-perplexity",
                        default=False, type=bool)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # read args
    args = parse_args()
    model_name = args.model
    prompt_filename = args.filename

    # if masked = True than we compute pseudo-perplexity
    masked = args.masked

    # login to huggingface
    login(token='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')

    test_dir = "output/synonyms/test/lm/"
    output_dir = "output/synonyms/eval/"

    prompt_file = test_dir + prompt_filename

    # datetime object containing current date and time
    dformat = "%d%m%Y_%H%M%S"
    start = time.time()

    print(f"{datetime.now().strftime(dformat)}: Starting evaluation")
    print(f"{datetime.now().strftime(dformat)}: Using model_name {model_name}")
    print(f"{datetime.now().strftime(dformat)}: \
          Evaluating examples from {prompt_file}")

    model_name_last = model_name.split('/')[-3:]
    model_name_list = list(filter(None, model_name_last))
    model_name_file = '.'.join(model_name_list)

    sentences_list = get_sentences_to_eval(prompt_file)

    if masked:
        print(f"{datetime.now().strftime(dformat)}: Computing pseudo-perplexity for masked")
    else:
        print(f"{datetime.now().strftime(dformat)}: Computing perplexity for causal")

    # for masked we compute pseudo perplexity, not really the same thing
    if masked:
        pp_list = score_masked(sentences_list, model_name)
    else:
        # pp_list = score_causal(sentences_list, model_name)
        pp_list = score_causal_old(sentences_list, model_name)

    print(f"{datetime.now().strftime(dformat)}: \
          Evaluation ended, time elapsed: {time.time() - start}")

    score_file = create_score_filename(prompt_file,
                                       model_name_file,
                                       output_dir)

    print(score_file)

    write_scores_to_file(prompt_file, pp_list, score_file)
    print(f"{datetime.now().strftime(dformat)}: Write to file {score_file}")

    torch.cuda.empty_cache()
    gc.collect()

