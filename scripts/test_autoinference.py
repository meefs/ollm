# ollm AutoInference
# pip install peft 
# supported model families: llama3, gemma3

from ollm import TextStreamer, AutoInference, file_get_contents

def get_sample():
	return f"Extract short summary from document: {file_get_contents("./temp/xsum_sample.txt")}\nSummary:\n",
	#completion: Clean-up operations are continuing across the Scottish Borders and Dumfries and Galloway after flooding caused by Storm Frank.

o = AutoInference("/media/mega4alik/ssd/models/gemma3-12B", # any llama3 or gemma3 model
	adapter_dir="/home/mega4alik/Desktop/python/peftee/model_temp/checkpoint-20", #PEFT adapter checkpoint if available
	device="cuda:0", multimodality=False, logging=False)
past_key_values = o.DiskCache(cache_dir="./kv_cache/") #set None if context is small
text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

#messages = [{"role":"system", "content":"You are helpful AI assistant"}, {"role":"user", "content":"List planets"}]
#input_ids = o.tokenizer.apply_chat_template(messages, reasoning_effort="minimal", tokenize=True, add_generation_prompt=True, return_tensors="pt").to(o.device)
input_ids = o.tokenizer(get_sample(), add_special_tokens=False, return_tensors="pt").input_ids.to(o.device)

outputs = o.model.generate(input_ids=input_ids,  past_key_values=past_key_values, max_new_tokens=500, streamer=text_streamer).cpu()
answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
print(answer)