#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
from collections import defaultdict
import random
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
from copy import deepcopy
import datasets
from datasets import load_dataset, Dataset, DatasetDict

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.utils.versions import require_version

from transformers.trainer_utils import get_last_checkpoint, has_length, seed_worker
from transformers.utils import check_min_version, send_example_telemetry, is_datasets_available

from peft import get_peft_model,LoraConfig
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")

require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


import wandb
# wandb.init(project="semformer", resume="allow")

from mytrainer import MyTrainer as Trainer

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default='gpt2',
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    
    """
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )
    """

    # added
    model_name_or_path_ae: Optional[str] = field(default=None)

    ztokens: Optional[int] = field(default=32)  # ztoken 的长度
    shallow_decoder_n_layer: Optional[int] = field(default=12)
    
    z_from_layer: Optional[int] = field(default=-1)
    predictor: Optional[str] = field(default="linear")

    # for ablation study
    main_decoder_n_layer: Optional[int] = field(default=12)

    zdim: Optional[int] = field(default=32)
    znorm: Optional[int] = field(default=0)
    alpha: Optional[float] = field(default=1.0)
    beta: Optional[float] = field(default=1.0)
    regloss: Optional[str] = field(default="l2")

    large_path: Optional[str] = field(default='gpt2-large')

    from_scratch: Optional[bool] = field(default=True)

    use_flash_attention: Optional[bool] = field(default=True)
    use_lora: Optional[bool] = field(default=False)

    load_pretrained_enc_and_fix: Optional[int] = field(default=1)
    fix_aedecoder: Optional[int] = field(default=0)
    only_tunelayers: Optional[int] = field(default=0)

    share_emb_decoders: Optional[int] = field(default=1)

    prefix_generator: Optional[str] = field(default='mlp')

    beams: Optional[int] = field(default=4)

    # sae config
    use_sae: Optional[bool] = field(default=False)
    sae_architecture: Optional[str] = field(default="jumprelu",metadata={"choices": ["jumprelu", "standard"]})
    sae_use_pre_enc_bias: Optional[bool] = field(default=True)
    sae_activation_size: Optional[int] = field(default=1024)
    sae_bandwidth: Optional[float] = field(default=0.1)
    sae_sparsity_coefficient: Optional[float] = field(default=0.5)

    sae_kl_temperature: Optional[float] = field(default=1.0)
    sae_kl_sparse: Optional[int] = field(default=0)

    zorth_reg:Optional[float]=field(default=0)

    mlanguages_planning:Optional[bool] = field(default=False)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={
                                      "help": "The input training data file (a text file)."})

    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={
                            "help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    # added
    encstr: str = field(
        default='suffix',
        metadata={"help": "suffix or all"},
    )
    ptae: str = field(
        default=None,
        metadata={"help": "suffix or all"},
    )

    group_doc: bool = field(default=True)
    sent_split: bool = field(default=True)
    
    # DAE, random delete 
    mask_ratio: float = field(default=0)

    model_backbone: str = field(
        default="gpt2",
        metadata={"help": ""},
    )
    from_disk: Optional[str] = field(default=None, metadata={"help": "load from disk."})
    already_tokened: Optional[bool] = field(default=False, metadata={"help": "load from disk."})

    append_z_aeenc: int = field(default=0)
    ae_min_length: int = field(default=32)

    mlanguages: Optional[str] = field(default=None, metadata={"help": "e.g., en_de_fr"})
    
    decode_srclang: Optional[str] = field(default=None, metadata={"help": "e.g., en"})
    decode_tgtlang: Optional[str] = field(default=None, metadata={"help": "e.g., fr"})

    open_dp: Optional[bool] = field(default=False, metadata={"help": "open dropout"})

    
    output_file: Optional[str] = field(default=None, metadata={"help": ""})
    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0",
                            "The streaming feature requires `datasets>=2.0.0`")
    
    enc_mode: Optional[str] = field(
        default="target",
        metadata={"help": "The encoding mode: 'target', 'source', or 'source+target'"}
    )

    trans_direc: Optional[str] = field(default=None, metadata={"help": "e.g., en-de|en-zh|en-ru"})

        # if self.dataset_name is None and self.train_file is None and self.validation_file is None:
        #     raise ValueError(
        #         "Need either a dataset name or a training/validation file.")
        # else:
        #     if self.train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension in [
        #             "csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
        #     if self.validation_file is not None:
        #         extension = self.validation_file.split(".")[-1]
        #         assert extension in [
        #             "csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def split_doc_by_sent(data_args, model_args, doc, tokenizer, special_seq):
    sentences = sent_tokenize(doc)
    sent_num = len(sentences)

    if sent_num < 3:
        return {}
    
    where_to_add = random.randint(1, sent_num-1)

    prefix = sentences[:where_to_add]
    suffix = sentences[where_to_add:]

    if data_args.encstr == "suffix":
        if data_args.append_z_aeenc:
            ae_encstr = ' '.join(suffix) + tokenizer.eos_token + special_seq
        else:
            ae_encstr = ' '.join(suffix) + tokenizer.eos_token
        
        ae_decstr = ' '.join(suffix) + tokenizer.eos_token
    elif data_args.encstr == "all":
        ae_encstr = doc + tokenizer.eos_token
        ae_decstr = doc + tokenizer.eos_token
    else:
        exit("encstr type")

    decstr = ' '.join(prefix) + special_seq + ' '.join(suffix) + tokenizer.eos_token

    return {"ae_encstr":ae_encstr, "ae_decstr":ae_decstr, "decstr":decstr}


def split_doc_(data_args, model_args, doc, tokenizer, special_seq):
    predict_len_min = 32
    
    doc_split = doc.split()
    
    if len(doc_split) < predict_len_min * 2 + model_args.ztokens:
        return {}

    doc_len = len(doc_split)
    
    # print("doc_len", doc_len)

    insert_posi = random.randint(predict_len_min, doc_len - predict_len_min - model_args.ztokens)
    
    prefix = doc_split[:insert_posi]
    suffix = doc_split[insert_posi:]

    if data_args.encstr == "suffix":
        if data_args.append_z_aeenc:
            ae_encstr = ' '.join(suffix) + tokenizer.eos_token + special_seq
            ae_decstr = special_seq + ' '.join(suffix) + tokenizer.eos_token
        else:
            ae_encstr = ' '.join(suffix) + tokenizer.eos_token
            ae_decstr = ' '.join(suffix) + tokenizer.eos_token

    elif data_args.encstr == "all":
        ae_encstr = doc + tokenizer.eos_token
        ae_decstr = doc + tokenizer.eos_token
    else:
        exit("encstr type")
    
    decstr = ' '.join(prefix) + special_seq + ' '.join(suffix) + tokenizer.eos_token
    
    return {"ae_encstr":ae_encstr, "ae_decstr":ae_decstr, "decstr":decstr}


PROMPT_DICT = {
    "prompt_input": (
        "Write a response that appropriately completes the request.\n\n"
        "### Request:\n{instruction}\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Write a response that appropriately completes the request.\n\n"
        "### Request:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_instruct": (
        "{input} "
    ),
}

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_flash_attention:
        check_min_version("4.35.0")
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    # added
    if data_args.ptae is not None:
        config = AutoConfig.from_pretrained(data_args.ptae, **config_kwargs)
    else:
        if model_args.config_name:
            config = AutoConfig.from_pretrained(
                model_args.config_name, **config_kwargs)
        elif model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path, **config_kwargs)
        else:
            config = CONFIG_MAPPING[model_args.model_type]()
            logger.warning(
                "You are instantiating a new config instance from scratch.")
            if model_args.config_overrides is not None:
                logger.info(f"Overriding config: {model_args.config_overrides}")
                config.update_from_string(model_args.config_overrides)
                logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.

    if model_args.mlanguages_planning:

        languages = data_args.trans_direc.split('|')

        special_seq = {}
        tholist.l = {}
        thoid.l = {}

        z_start_id = {}

        for lang_pair in languages:

            special_list = [f'<THO{idx}_{lang_pair}>' for idx in range(model_args.ztokens)]
            special_seq[lang_pair] = ''.join(special_list)
            tokenizer.add_special_tokens({'additional_special_tokens': special_list})
            
            tholist.l[lang_pair] = [tokenizer.convert_tokens_to_ids(
                    f'<THO{idx}_{lang_pair}>') for idx in range(model_args.ztokens)]
            
            thoid.l[lang_pair] = tokenizer.convert_tokens_to_ids(f'<THO0_{lang_pair}>')

            z_start_id[lang_pair] = tokenizer.convert_tokens_to_ids(f'<THO0_{lang_pair}>')
            # print(tholist.l)
            # print(thoid.l)
            # print(z_start_id)

    else:
        special_list = [f'<THO{idx}>' for idx in range(model_args.ztokens)]
        special_seq = ''.join(special_list)
        tokenizer.add_special_tokens({'additional_special_tokens': special_list})
        # print(tokenizer)
    
        tholist = [tokenizer.convert_tokens_to_ids(
                f'<THO{i}>') for i in range(model_args.ztokens)]

        thoid = tokenizer.convert_tokens_to_ids('<THO0>')
        # sepid = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)

        z_start_id = tokenizer.convert_tokens_to_ids('<THO0>')

    tokenizer.pad_token_id = tokenizer.eos_token_id
    # print(tokenizer.pad_token_id)

    from datasets import concatenate_datasets, interleave_datasets, load_dataset

    if data_args.mlanguages is not None:
        logging.info("load multiple languages")

        mdatasets = []
        for lang in data_args.mlanguages.split('_'):
            mdataset = load_dataset(f"{data_args.dataset_name}/c4_{lang}", data_args.dataset_config_name, streaming=True)
            mdatasets.append(mdataset)

        column_names = list(mdatasets[0]["train"].features)
        column_names.remove("timestamp")
        
        # Concatenate both datasets
        # Or interleave them (alternates between one and the other)
        # raw_datasets = interleave_datasets([mdataset for mdataset in mdatasets])
        
        raw_datasets = DatasetDict()
        raw_datasets["train"] = interleave_datasets([mdataset["train"].remove_columns("timestamp") for mdataset in mdatasets])
        raw_datasets["validation"] = interleave_datasets([mdataset["validation"].remove_columns("timestamp") for mdataset in mdatasets])
    else:
        if data_args.from_disk is not None:
            raw_datasets = datasets.load_from_disk(data_args.from_disk)
        else:
            if data_args.dataset_name is not None:
                raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                streaming=data_args.streaming,
                )
            else:
                data_files = {}
                dataset_args = {}
                
                if data_args.train_file is not None:
                    data_files["train"] = data_args.train_file
                
                if data_args.validation_file is not None:
                    data_files["validation"] = data_args.validation_file
                
                extension = (
                    data_args.train_file.split(".")[-1]
                    if data_args.train_file is not None
                    else data_args.validation_file.split(".")[-1]
                )
                
                # added
                extension = "txt"

                if extension == "txt":
                    extension = "text"
                    dataset_args["keep_linebreaks"] = data_args.keep_linebreaks

                raw_datasets = load_dataset(
                    extension,
                    data_files=data_files,
                    cache_dir=model_args.cache_dir,
                    **dataset_args,
                )
        
        if training_args.do_train:
            column_names = list(raw_datasets["train"].features)
        else:
            column_names = list(raw_datasets["validation"].features)
    
    # logger.info("split a val set")
    # split_dataset = raw_datasets["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    # split_dataset['val'] = split_dataset.pop('test') # rename the test split to val    
    # raw_datasets = split_dataset

    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base")
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
    
    enc_block_size = block_size // 2
    
    # for baseline
    if not hasattr(config, "ztokens"):
        config.update({"ztokens": model_args.ztokens, "zdim": model_args.zdim, "task_ztokens":model_args.ztokens,
                    "z_start_id":z_start_id, "len_tokenizer":len(tokenizer)})

    if data_args.open_dp:
        config.update({"dropout": 0.1})
    
    config.update({"predictor_nn": model_args.predictor})

    logger.info(f"model args: {model_args}")
    
    from semformer_hfmodels import SemformerHF

    if model_args.use_flash_attention:
        config._attn_implementation = "flash_attention_2"

    lora_config = None
    if model_args.use_lora:
        if config.architectures[0] == "GPT2LMHeadModel":
            # for gpt2
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["c_attn","c_proj","c_fc"],
                bias="none",
                task_type="CAUSAL_LM"
            )
        else:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                bias="none",
                task_type="CAUSAL_LM"
            )

    print(config)
    print("*************")
    if data_args.ptae is not None:
        logger.info(f"loading pretrained ae model {data_args.ptae}")
        tmodel = SemformerHF.from_pretrained(data_args.ptae, config=config, model_args=model_args, 
                                             ignore_mismatched_sizes=True,
                                             lora_config=lora_config)
    else:
        tmodel = SemformerHF(config=config, model_args=model_args, lora_config=lora_config)
    
    if model_args.only_tunelayers:
        tmodel.only_tunelayers()

    if model_args.fix_aedecoder:
        tmodel.freeze_aedecoer()

    if model_args.load_pretrained_enc_and_fix:
        tmodel.load_encoder_and_fix(model_args.model_name_or_path)

    if model_args.beta == 0:
        tmodel.freeze_ae()
    elif model_args.alpha == 0:
        tmodel.freeze_lm()


    if training_args.do_predict:
        # import nltk
        # def postprocess_text(preds, labels):
        #     preds = [pred.strip() for pred in preds]
        #     labels = [label.strip() for label in labels]
        #     # rougeLSum expects newline after each sentence
        #     preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        #     labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        #     return preds, labels

        tmodel.eval().half().cuda()

        gen_model = tmodel.main_decoder

        from torch.utils.data import DataLoader

        # test_dataset = raw_datasets["test"].select(range(300))
        test_dataset = raw_datasets["validation"]

        ld = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size)
        samples_seen  = 0
        
        print(len(test_dataset))
        
        safe_length = block_size - 1 - config.ztokens - enc_block_size
         # sep*1, ztokens, summary
        
        tokenizer.padding_side = 'left'

        # Prepare everything with our `accelerator`.
        from accelerate import Accelerator
        accelerator = Accelerator()

        gen_model, ld = accelerator.prepare(gen_model, ld)
        
        from tqdm import tqdm
        import torch
        
        fw = open(f"{training_args.output_dir}/{data_args.output_file}", "w")

        lang_dict = {"en":"English", "de":"German", "zh":"Chinese", "fr": "French", "es": "Spanish", "ru": "Russian", "fi": "Finnish"}

        for idx, cur in tqdm(enumerate(ld), desc = "generating"):
            instruction = f"Translate the sentence from {lang_dict[data_args.decode_srclang]} to {lang_dict[data_args.decode_tgtlang]}."
            
            # src_sent = [PROMPT_DICT['prompt_input'].format_map({"instruction": ins[i], "input":src_sent[i]}) for i in range(len(ins))]
            src_sents = cur["text"]
            x = [PROMPT_DICT['prompt_input'].format_map({"instruction": instruction, "input":src_sent}) for src_sent in src_sents]

            xx = tokenizer(
                x,
                padding = "longest",
                max_length = safe_length,
                truncation = True,
                return_tensors="pt"
            )

            # labels = tokenizer(
            #     y,
            #     padding = "longest",
            #     return_tensors="pt"
            # )
            # labels = labels.input_ids.cuda()
            lang_pair_i = data_args.decode_srclang + "_" + data_args.decode_tgtlang
            if model_args.mlanguages_planning:
                appseq = [special_seq[lang_pair_i]] * len(x)
        
            else:
                appseq = [special_seq] * len(x)
            appxx = tokenizer(
                appseq,
                return_tensors="pt"
            )

            input_ids = torch.cat((xx.input_ids, appxx.input_ids), dim = -1)
            attention_mask = torch.cat((xx.attention_mask, appxx.attention_mask), dim = -1)

            with torch.inference_mode():
                pred = accelerator.unwrap_model(gen_model).generate(
                    input_ids.cuda(),
                    attention_mask = attention_mask.cuda(),
                    num_beams = model_args.beams,
                    min_new_tokens = 1,
                    max_new_tokens = block_size,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                )

            generated_tokens = pred[:, input_ids.size(1):]

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            # labels = accelerator.pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id)
            # generated_tokens, labels = accelerator.gather_for_metrics((generated_tokens, labels))
            
            generated_tokens = accelerator.gather_for_metrics(generated_tokens)
            generated_tokens = generated_tokens.cpu().numpy()
            # labels = labels.cpu().numpy()

            # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # logger.info("see a sample, pred, label")
            # accelerator.print(json.dumps({"src": x[0], "pred":decoded_preds[0], "target":decoded_labels[0]}, indent=4))
            
            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                for pred in decoded_preds:
                    pred = pred.replace("\n", "")

                    fw.write(pred + "\n")

            samples_seen += len(decoded_preds)
        
        logger.info(f'done, samples_seen {samples_seen}')
        fw.close()
        exit(0)


    def safe_encode(srclist, tarlist):
        # ae-encoder
        encres = {
            'input_ids': [],
            'attention_mask': []
        }
        ae_decres = {
            'input_ids': [],
            'attention_mask': []
        }

        for src, tar in zip(srclist, tarlist):
            # generating full encstrs:
            if data_args.enc_mode == "target":
                enctail = tar
            elif data_args.enc_mode == "source":
                enctail = src
            elif data_args.enc_mode == "source+target":
                enctail = src + " " + tar
            enctail_ids = tokenizer.encode(enctail)

            if data_args.encstr == "all":
                exit("not ready")
            else:
                assert data_args.encstr == "suffix"
                # if len(enctail_ids) >= block_size // 1.5: # ill tail
                enctail_ids = enctail_ids[:enc_block_size]
                # another version
                # enctail_ids = enctail_ids[:enc_block_size-config.ztokens] + tholist
                encres_app = enctail_ids + \
                    [tokenizer.pad_token_id] * \
                    max(enc_block_size-len(enctail_ids), 0)

            encres_atm = [1 if i != tokenizer.pad_token_id else 0 for i in encres_app]

            assert len(encres_app) == len(encres_atm) == enc_block_size

            encres["input_ids"].append(encres_app)
            encres["attention_mask"].append(encres_atm)

            # generating  ae-decoder:
            ae_dec_ids = tokenizer.encode(tar)
            ae_dec_ids = ae_dec_ids[:enc_block_size]

            ae_dec_app = ae_dec_ids + \
                    [tokenizer.pad_token_id] * \
                    max(enc_block_size-len(ae_dec_ids), 0)
            
            ae_decres["input_ids"].append(ae_dec_app)
            ae_dec_atm = [1 if i != tokenizer.pad_token_id else 0 for i in ae_dec_app]
            ae_decres["attention_mask"].append(ae_dec_atm)

            assert len(ae_dec_app) == len(ae_dec_atm) == enc_block_size

            # generating  decstrs:
            # dectail = tokenizer.sep_token + special_seq + tar + tokenizer.eos_token
            # dectail_ids = tokenizer.encode(dectail)
            # dectail_ids = [tokenizer.sep_token_id] + tholist + ae_dec_ids + [tokenizer.eos_token_id]
            if model_args.mlanguages_planning:
                # main decoder
                decres_l = {}

                for lang_pair in languages:

                    dectail_ids = tholist.l[lang_pair] + ae_dec_ids + [tokenizer.eos_token_id]

                    dechead = src
                    dechead_ids = tokenizer.encode(dechead)[:block_size - len(dectail_ids)]

                    decres_app = dechead_ids + dectail_ids + \
                        [tokenizer.pad_token_id] * \
                        max(block_size -len(dechead_ids)-len(dectail_ids), 0)
            
                    # the eosid == padid
                    decres_atm = [1] * (len(dechead_ids) + len(dectail_ids)) + [0] * (block_size-len(dechead_ids)-len(dectail_ids))
            
                    assert len(decres_app) == len(decres_atm) == block_size

                    decres_l[lang_pair] = {
                        'input_ids': decres_app,
                        'attention_mask': decres_atm,
                    }

                    return encres,ae_decres, decres_l


            else:
                # main decoder
                decres = {
                'input_ids': [],
                'attention_mask': [],
                }

                dectail_ids = tholist + ae_dec_ids + [tokenizer.eos_token_id]

                dechead = src
                dechead_ids = tokenizer.encode(dechead)[:block_size - len(dectail_ids)]

                decres_app = dechead_ids + dectail_ids + \
                    [tokenizer.pad_token_id] * \
                    max(block_size -len(dechead_ids)-len(dectail_ids), 0)
            
                # the eosid == padid
                decres_atm = [1] * (len(dechead_ids) + len(dectail_ids)) + [0] * (block_size-len(dechead_ids)-len(dectail_ids))
            
                assert len(decres_app) == len(decres_atm) == block_size

                decres["input_ids"].append(decres_app)
                decres["attention_mask"].append(decres_atm)

                return encres,ae_decres, decres

    def tokenize_function(examples, prefix_loss):

        with CaptureLogger(tok_logger) as cl:
            src_sent, tar_sent = examples["input"], examples["output"]
            ins = examples["instruction"]
            src_sent = [PROMPT_DICT['prompt_input'].format_map({"instruction": ins[i], "input":src_sent[i]}) for i in range(len(ins))]
            lang_pair_i = data_args.decode_srclang + "_" + data_args.decode_tgtlang

            # print(src_sent[0:2])
            # print(tar_sent[0:2])
            # exit()

            bs = len(src_sent)
            if model_args.mlanguages_planning:
                encres, ae_decres, decres.l = safe_encode(srclist=src_sent, tarlist=tar_sent)
                decres = decres.l[lang_pair_i]
                tholist = tholist.l[lang_pair_i]
                thoid = thoid.l[lang_pair_i]
            else:
                encres, ae_decres, decres = safe_encode(srclist=src_sent, tarlist=tar_sent)
                tholist = tholist
                thoid = thoid               

            input_ids = decres['input_ids']
            attention_mask = decres['attention_mask']
            labels = input_ids.copy()

            for i in range(bs):
                pad_label = [-100 if label_token in tholist or label_token == tokenizer.pad_token_id else label_token
                             for label_token in labels[i]]
                
                start_pos = input_ids[i].index(thoid)

                for first_pad_posi in range(start_pos + len(tholist) + 1 + 1, len(pad_label)):
                    if pad_label[first_pad_posi] == -100:
                        break
                
                pad_label[first_pad_posi] = tokenizer.eos_token_id

                if not prefix_loss:
                    pad_label[:start_pos] = [-100] * len(pad_label[:start_pos])

                    labels [i] = pad_label

            input_ids_enc = encres['input_ids']
            attention_mask_enc = encres['attention_mask']

            input_ids_ae_dec = ae_decres['input_ids']
            attention_mask_ae_dec = ae_decres['attention_mask']
            labels_ae = [
                [-100 if i in tholist or i ==
                    tokenizer.pad_token_id else i for i in j]
                for j in input_ids_ae_dec
            ]

        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,

            'input_ids_enc': input_ids_enc,
            'attention_mask_enc': attention_mask_enc,

            'input_ids_ae_dec': input_ids_ae_dec,
            'attention_mask_ae_dec': attention_mask_ae_dec,

            'labels_ae': labels_ae
        }


    if data_args.already_tokened:
        logger.info("already_tokened")
        lm_datasets = raw_datasets
    else:
        with training_args.main_process_first(desc="dataset map tokenization and save to disk"):
            if not data_args.streaming:
                lm_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                    fn_kwargs={"prefix_loss": False}
                )
            else:
                lm_datasets = DatasetDict()
                lm_datasets["train"] = raw_datasets["train"].map(
                    tokenize_function,
                    batched=True,
                    remove_columns=column_names,
                    fn_kwargs={"prefix_loss": False}
                )
                if "validation" in raw_datasets:
                    lm_datasets["validation"] = raw_datasets["validation"].map(
                        tokenize_function,
                        batched=True,
                        remove_columns=column_names,
                        fn_kwargs={"prefix_loss": False}
                    )

    trainable_parameters = 0
    all_param = 0
    # logger.info(f"{tmodel.main_decoder.lm_head.weight.requires_grad},")
    # logger.info(f"{tmodel.aemodel.decoder.lm_head.weight.requires_grad},")
    for pname, param in tmodel.named_parameters():
        all_param += param.numel()
        logger.info(f"{pname}, {param.requires_grad}")

        if param.requires_grad:
            trainable_parameters += param.numel()
    
    logger.info(f"trainable params: {trainable_parameters} || all params: {all_param} || trainable%: {100 * trainable_parameters / all_param}")

    """
    try:
        print(tokenizer.batch_decode(
            lm_datasets['train']["input_ids"][:5],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ))
        print(tokenizer.batch_decode(
            lm_datasets['train']["input_ids_enc"][:5],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ))
        print(tokenizer.batch_decode(
            lm_datasets['train']["input_ids_enc_z"][:5],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ))
    except:
        print("ERROR during decoding...")
    """

    if training_args.do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]

        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in lm_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        def calc_acc(preds, labels):
            totals = 0
            exm = 0
            for idx, i in enumerate(labels):
                if i.item() != -100:
                    totals += 1
                    if i.item() == preds[idx].item():
                        exm += 1
            res = exm / totals if totals else 0
            return {'accuracy': res}

        def compute_metrics(eval_preds):

            preds, labels = eval_preds
            # print(labels)
            labels = labels[0]
            # print(type(labels))
            for idx, i in enumerate(labels):
                pos = 0
                while pos < len(i):
                    if i[pos] == tokenizer.sep_token_id:
                        labels[idx][pos] = -100
                        break
                    else:
                        labels[idx][pos] = -100
                        pos += 1

            # torch.set_printoptions(profile='full', precision=1)
            # print(torch.tensor(labels))

            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels

            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return calc_acc(preds, labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=tmodel,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        if not data_args.streaming:
            max_train_samples = (
               data_args.max_train_samples if data_args.max_train_samples is not None else len(
                   train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_train:
        kwargs = {"finetuned_from": model_args.model_name_or_path,
                  "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        if training_args.push_to_hub:
            trainer.push_to_hub(**kwargs)
        else:
            trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

