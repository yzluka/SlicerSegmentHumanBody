import slicer

#This should live only in CPU space
class ModelRegistry:
    model_cache={}

    def get_model(key):

        if key not in ModelRegistry.model_cache:
            ModelRegistry.check_dependencies(key)
            model = ModelRegistry.instantiate_model(key)
            ModelRegistry.model_cache[key] = model
            return model 
        
        return ModelRegistry.model_cache[key]


    def check_dependencies(key):
        print(f"[Dependencies] Checking for {key}")
        pass
    def instantiate_model(key):
        print(f"[Instantiation] Fetching model for {key}")
        pass
    
    