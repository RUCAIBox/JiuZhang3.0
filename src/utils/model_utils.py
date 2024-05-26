import torch


def load_hf_lm_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    device_map="auto",
    load_in_8bit=False,
    load_in_half=False,
    gptq_model=False,
    use_fast_tokenizer=False,
    padding_side="left",
    model_max_length=2048,
    cache_dir=None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path

    if "llama" in model_name_or_path:
        model_class = LlamaForCausalLM
        tokenizer_class = LlamaTokenizer
    else:
        model_class = AutoModelForCausalLM
        tokenizer_class = AutoTokenizer

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        use_auth_token=None,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        model_max_length=model_max_length,
        padding_side=padding_side,  # set padding side to left for batch generation
        use_fast=False,
        cache_dir=cache_dir,
    )
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    # if tokenizer.pad_token is None:
    #     print(f"Set tokenizer pad token to {tokenizer.eos_token} {tokenizer.eos_token_id}")
    #     tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.pad_token_id = tokenizer.eos_token_id
    #     config.pad_token_id = config.bos

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    # if gptq_model:
    #     from auto_gptq import AutoGPTQForCausalLM

    #     model_wrapper = AutoGPTQForCausalLM.from_quantized(model_name_or_path, device="cuda:0", use_triton=True)
    #     model = model_wrapper.model
    # elif load_in_8bit:
    #     model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map, load_in_8bit=True)
    # else:
    #     if device_map:
    #         model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=device_map)
    #     else:
    #         model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    #         if torch.cuda.is_available():
    #             model = model.cuda()
    #     if load_in_half:
    #         model = model.half()
    model.eval()
    return model, tokenizer
