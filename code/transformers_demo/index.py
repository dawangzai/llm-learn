from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'uer/gpt2-chinese-cluecorpussmall'
cache_dir = 'model/gpt2-chinese-cluecorpussmall'

AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

JointIntentWithSlotCRFin.from_pretrained
print(f'模型已经下载到{cache_dir}')