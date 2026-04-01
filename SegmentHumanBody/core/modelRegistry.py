import slicer

#This should live only in CPU space
class ModelRegistry:
    model_cache={}

    def get_model(key):

        if key not in ModelRegistry:
            ModelRegistry.check_dependencies(key)
            model = ModelRegistry.instantiate_model(key)
            ModelRegistry.model_cache[key] = model
            return model 
        
        return ModelRegistry.model_cache[key]


    def check_dependencies(key):
        pass
    def instantiate_model(key):
        pass
    
    