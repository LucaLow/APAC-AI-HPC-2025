yo ngl I still don't really get what the fuck is going on with the tp dp ep and all that shit so here's my documentation from researching what the fuck is going on

atm sglang does not support pipeline parallelism (wouldn't help anyway)

[Broad Overview of All Methods](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
# [Tensor Parallelism](https://huggingface.co/docs/text-generation-inference/en/conceptual/tensor_parallelism#tensor-parallelism)
![Image depicting tensor parallelism or some shit](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tgi/TP.png)
Ngl this is actually super simple. So when your multiplying the input tensor (user input, shown as $X$) by the weight tensor (shown as $A$), you'll often discover that with large models, the weights tensor will be to heavy (haha get it) to be stored on a single GPU, so split it into columns instead, store 1/Nth of the weights on N GPU's, perform the matrix multiplication, and then concatonate the outputs and you end up with output!

# Data Parallelism
Scale across GPU's, so DP=2 would mean that across all GPU's there are two complete copies of the model, and requets would be split between each model instance, so 20 requests and dp=2 would mean 10 go to one and 10 to the other. It is highly recommended to look into sglang router when investigating this. **Data Parallelism does not appear possible due to the model size being too large for 16 H100's**