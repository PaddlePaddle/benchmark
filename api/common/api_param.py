import json

class BaseParamInfo(object):
    def __init__(self, name, type, value):
        self.name = name
        self.type = type
        self.value = value

    def convert_params_to_str(self):
         param = self.name + '--' + self.type + '| '+ self.value +'\n '
         return param
 
class VarParamInfo(object):
    def __init__(self, name, type, dtype, shape, lod_level=0):
        self.name = name
        self.type = type
        self.dtype = dtype
        self.shape = shape
        self.lod_level = lod_level

    def convert_params_to_str(self):
         param = self.name + '--' + self.type + '| '+ self.dtype + '| shape:'+ self.shape +'\n '
         return param

class APIParam(object):
    def __init__(self, op_type, params):
        self.name = op_type
        self.params = params

    def init_from_json(self, filename, pos=0):
        with open(filename, 'r') as f:
            data = json.load(f) 
            self.name = data[pos]["op"]
            self.params = data[pos]["param_info"]
        return self

    def _convert_params_to_str(self):
        params=""
        for p_key, p_value in self.params.items():
            params_str=""
            param_name=p_key
            dtype=p_value["dtype"]
            type=""
            v_type = 0
            for v_k in p_value.keys():
                if v_k == "type":
                    v_type = 1
            if v_type ==1:
                type=p_value["type"]
                shape=p_value["shape"]
                var_ = VarParamInfo(param_name, type, dtype, shape)
            else:
                value=p_value["value"]
                var_ = BaseParamInfo(param_name, dtype, value)
            params_str=var_.convert_params_to_str()
            params=params+params_str
        return params
