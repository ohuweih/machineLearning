## An example of How to use kserve to front this model with http.  

#import kserve
#from typing import Dict
#import generator
#import os
#import ast
#
#
#class AIServiceModel(kserve.Model):
#    """Wrap Generator with kserve."""
#
#    def __init__(self, name: str):
#        super().__init__(name)
#        self.name = name
#        self.load()
#
#    def load(self):
#        pass
#
#    def predict(self, request: Dict) -> Dict:
#        """Execute generator class and return the result."""
#        payload = []
#        print("Request recieved")
#        print(request)
#        for instance in request["instances"]:
#            print("Digging through response")
#            g = generator.Generator()
#            if instance["numGenerations"]:
#                g.num_generations = instance["numGenerations"]
#            if instance["generationLength"]:
#                g.generation_length = instance["generationLength"]
#            if instance["temperature"]:
#                g.temperature = instance["temperature"]
#            if instance["numBeams"]:
#                g.num_beams = instance["numBeams"]
#            if instance["doSampling"]:
#                g.do_sampling = instance["doSampling"]
#            if instance["noRepeatNGramSize"]:
#                g.no_repeat_ngram_size = instance["noRepeatNGramSize"]
#            if instance["saveGenerator"]:
#                g.save_generator = instance["saveGenerator"]
#            print("Running generation.")
#            _payload = g.generate()
#            payload.extend(_payload)
#            print("Returning:\n {0}\n".format(payload))
#        return dict(predictions=payload)
#
#
#if __name__ == "__main__":
#    print("Starting model")
#    model_name = "gan-model"
#    print("Model name: {0}".format(model_name))
#    model = AIServiceModel(model_name)
#    kserve.ModelServer().start([model])