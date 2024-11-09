from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline


class PredictionPipeline:
    def _init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        
        
    def predict(self,text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {"length_penalty": 0.8, "num_beams":18, "max_length" : 128}
        
        pipe = pipeline("Summarization", model=self.config.model_path,tokenizer=tokenizer)
        
        print("Dialogue:")
        print(text)
        
        output = pipe(text, **gen_kwargs)[0]["Summary_text"]
        print("\nModel Summary:")
        print(output)
        
        return output