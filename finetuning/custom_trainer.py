import sys
from transformers import Seq2SeqTrainer

class CustomTrainer(Seq2SeqTrainer):


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        with open("debug.log", "a") as f:
            f.write(f"DEBUG: CustomTrainer.compute_loss called.\n")
            f.write(f"DEBUG: Inputs Type: {type(inputs)}\n")
            f.write(f"DEBUG: Inputs Dir: {dir(inputs)}\n")
            f.write(f"DEBUG: Inputs Keys (list): {list(inputs.keys())}\n")
        
        # AGGRESSIVE REMOVAL
        if "input_ids" in inputs:
            with open("debug.log", "a") as f:
                f.write(f"DEBUG: Found input_ids via 'in'. Removing.\n")
            del inputs["input_ids"]
        
        # Check attribute access
        if hasattr(inputs, "input_ids"):
             with open("debug.log", "a") as f:
                f.write(f"DEBUG: Found input_ids via hasattr. Removing.\n")
             try:
                 delattr(inputs, "input_ids")
             except:
                 inputs.pop("input_ids", None)


        import inspect
        with open("debug.log", "a") as f:
            f.write(f"DEBUG: Model Forward Signature: {inspect.signature(model.forward)}\n")
            # f.write(f"DEBUG: Base Model Forward Signature: {inspect.signature(model.base_model.forward)}\n")

        # Cast to clean dict
        inputs_dict = dict(inputs)
        with open("debug.log", "a") as f:
             f.write(f"DEBUG: Converted to dict. Keys: {list(inputs_dict.keys())}\n")

        
        # Custom manual calls to debug
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        
        # MANUAL CALL
        # if "input_ids" in inputs:
        #      del inputs["input_ids"]
             
        # outputs = model(**inputs)
        # loss = outputs.loss
        
        # return (loss, outputs) if return_outputs else loss


    def training_step(self, model, inputs, num_items_in_batch=None):
        with open("debug.log", "a") as f:
             f.write(f"DEBUG: CustomTrainer.training_step called.\n")
        
        if "input_ids" in inputs:
             del inputs["input_ids"]
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
